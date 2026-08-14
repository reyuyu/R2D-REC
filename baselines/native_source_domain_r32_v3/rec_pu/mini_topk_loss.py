"""Multi-positive Set-NLL with dynamic same-level Top-K coverage.

This module is deliberately independent from the Trainer.  It receives the
already-located final SID logits and prefix-conditioned positive token ids.
All reductions are float32; callers multiply the returned scalar by the
existing SID8 loss weight exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniTopKConfig:
    enabled: bool = False
    slack_a: int = 4
    slack_b: int = 8
    slack_c: int = 12
    rank_weight: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if min(self.slack_a, self.slack_b, self.slack_c) < 0:
            raise ValueError("MiniTopK slacks must be non-negative.")
        if self.rank_weight < 0 or self.temperature <= 0:
            raise ValueError("MiniTopK rank_weight must be >=0 and temperature >0.")

    def slack_for(self, level: str) -> int:
        return {"a": self.slack_a, "b": self.slack_b, "c": self.slack_c}[level]


@dataclass
class MiniTopKPositionResult:
    loss: torch.Tensor
    set_nll: torch.Tensor
    rank_loss: torch.Tensor
    native_ce: torch.Tensor
    all_in: torch.Tensor
    weakest_gap: torch.Tensor
    boundary_violation: torch.Tensor
    positive_count: int
    k: int
    multi: bool


@dataclass
class MiniTopKBatchResult:
    loss: torch.Tensor
    set_nll: torch.Tensor
    rank_loss: torch.Tensor
    native_ce: torch.Tensor
    all_in: torch.Tensor
    weakest_gap: torch.Tensor
    boundary_violation: torch.Tensor
    positive_count: torch.Tensor
    k: torch.Tensor
    multi: torch.Tensor


def _ids(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(int(v) for v in values), dtype=torch.long, device=device)


def mini_topk_position_loss(
    logits: torch.Tensor,
    *,
    gold_id: int | None = None,
    positive_ids: Sequence[int],
    same_level_ids: Sequence[int],
    level: str,
    config: MiniTopKConfig,
) -> MiniTopKPositionResult:
    """Return one final-component replacement loss and detached-safe facts.

    For a singleton positive set this is *exactly* full-vocabulary one-hot CE;
    no Top-K term is evaluated.  Multi-positive positions use set NLL plus a
    smooth weakest-positive coverage term against the (r+1)-th same-level
    negative only.
    """
    if logits.ndim != 1:
        raise ValueError("Expected one-dimensional logits.")
    positives = tuple(sorted(set(int(v) for v in positive_ids)))
    level_ids = tuple(sorted(set(int(v) for v in same_level_ids)))
    if not positives or gold_id is None:
        raise ValueError("Gold must be present in a non-empty positive set.")
    if not isinstance(gold_id, torch.Tensor) and int(gold_id) not in positives:
        raise ValueError("Gold must be present in the positive set.")
    if not set(positives).issubset(level_ids):
        raise ValueError("Positive ids must belong to the same-level vocabulary.")
    if len(positives) == 1:
        native = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([positives[0]], device=logits.device))
        zero = native * 0.0
        return MiniTopKPositionResult(native, native, zero, native, native.new_tensor(1.0), zero, zero, 1, 1, False)

    positive_index = _ids(positives, logits.device)
    level_index = _ids(level_ids, logits.device)
    value = logits.float()
    positive_logits = value.index_select(0, positive_index)
    set_nll = torch.logsumexp(value, dim=0) - torch.logsumexp(positive_logits, dim=0)

    positive_mask = torch.zeros(level_index.numel(), dtype=torch.bool, device=logits.device)
    # component vocab is sorted and positives are a subset; equality is robust
    # to non-contiguous added-token ids.
    positive_mask = (level_index[:, None] == positive_index[None, :]).any(dim=1)
    negative_index = level_index[~positive_mask]
    slack = config.slack_for(level)
    k = len(positives) + slack
    ranked_level = value.index_select(0, level_index)
    top = ranked_level.topk(min(k, ranked_level.numel())).indices
    all_in = positive_mask.index_select(0, top).sum().eq(len(positives)).to(dtype=value.dtype)
    if negative_index.numel() <= slack:
        rank_loss = value.sum() * 0.0
        gap = value.sum() * 0.0
        violation = value.sum() * 0.0
    else:
        negative_logits = value.index_select(0, negative_index)
        # r=4 -> fifth-highest negative (zero-indexed r).
        boundary = negative_logits.topk(slack + 1).values[-1]
        tau = float(config.temperature)
        weakest = -tau * (torch.logsumexp(-positive_logits / tau, dim=0) - torch.log(torch.tensor(float(len(positives)), device=value.device)))
        gap = weakest - boundary
        violation = gap.lt(0).to(dtype=value.dtype)
        rank_loss = F.softplus(-gap)
    gold = gold_id.reshape(1).to(device=logits.device, dtype=torch.long) if isinstance(gold_id, torch.Tensor) else torch.tensor([gold_id], device=logits.device)
    native = F.cross_entropy(value.unsqueeze(0), gold)
    loss = set_nll + float(config.rank_weight) * rank_loss
    return MiniTopKPositionResult(loss, set_nll, rank_loss, native, all_in, gap, violation, len(positives), k, True)


def mini_topk_batched_position_loss(
    logits: torch.Tensor,
    *,
    gold_ids: torch.Tensor,
    positive_sets: Sequence[Sequence[int]],
    same_level_ids: Sequence[int],
    level: str,
    config: MiniTopKConfig,
) -> MiniTopKBatchResult:
    """Vectorized equivalent of :func:`mini_topk_position_loss`.

    One full-vocabulary tensor is retained per A/B/C group instead of one per
    recommendation position.  This is essential for 8K packed backward memory
    while preserving the exact scalar objective and singleton CE route.
    """
    if logits.ndim != 2 or logits.size(0) != len(positive_sets):
        raise ValueError("Expected [positions, vocab] logits and aligned positive sets.")
    device = logits.device
    sets = [tuple(sorted(set(int(v) for v in values))) for values in positive_sets]
    if not sets or any(not values for values in sets):
        raise ValueError("Every batched position requires a non-empty positive set.")
    level_ids = tuple(sorted(set(int(v) for v in same_level_ids)))
    component_pos = {token: index for index, token in enumerate(level_ids)}
    if any(not set(values).issubset(component_pos) for values in sets):
        raise ValueError("Positive ids must belong to the same-level vocabulary.")
    value = logits.float()
    count = len(sets)
    max_positive = max(map(len, sets))
    pos_token_ids = torch.zeros((count, max_positive), dtype=torch.long, device=device)
    pos_component_ids = torch.zeros_like(pos_token_ids)
    pos_mask = torch.zeros((count, max_positive), dtype=torch.bool, device=device)
    for row, values in enumerate(sets):
        width = len(values)
        pos_token_ids[row, :width] = torch.tensor(values, dtype=torch.long, device=device)
        pos_component_ids[row, :width] = torch.tensor([component_pos[item] for item in values], dtype=torch.long, device=device)
        pos_mask[row, :width] = True
    p_count = pos_mask.sum(dim=1)
    if not bool(((gold_ids[:, None] == pos_token_ids) & pos_mask).any(dim=1).all()):
        # This branch is only a fail-closed assertion; normal metadata was
        # validated before loss routing.
        raise ValueError("Teacher labels must occur in their positive sets.")
    native = F.cross_entropy(value, gold_ids.to(device=device, dtype=torch.long), reduction="none")
    positive_logits = value.gather(1, pos_token_ids).masked_fill(~pos_mask, float("-inf"))
    set_nll = torch.logsumexp(value, dim=1) - torch.logsumexp(positive_logits, dim=1)

    component_ids = _ids(level_ids, device)
    component_logits = value.index_select(1, component_ids)
    same_level_positive = torch.zeros_like(component_logits, dtype=torch.bool)
    same_level_positive.scatter_(1, pos_component_ids, pos_mask)
    slack = config.slack_for(level)
    negative_logits = component_logits.masked_fill(same_level_positive, float("-inf"))
    negative_count = component_ids.numel() - p_count
    applicable = negative_count > slack
    # A/B/C component vocabularies are large; use a safe fallback for any toy
    # test whose component vocabulary has <= slack negatives.
    boundary = negative_logits.topk(min(slack + 1, negative_logits.size(1)), dim=1).values[:, -1]
    tau = float(config.temperature)
    smooth_min = -tau * (
        torch.logsumexp((-positive_logits / tau).masked_fill(~pos_mask, float("-inf")), dim=1)
        - torch.log(p_count.to(dtype=value.dtype))
    )
    gap = smooth_min - boundary
    rank_loss = F.softplus(-gap) * applicable.to(dtype=value.dtype)
    gap = gap * applicable.to(dtype=value.dtype)
    violation = gap.lt(0).to(dtype=value.dtype) * applicable.to(dtype=value.dtype)

    k = p_count + slack
    max_k = min(len(level_ids), max(map(len, sets)) + slack)
    top_indices = component_logits.topk(max_k, dim=1).indices
    top_positive = same_level_positive.gather(1, top_indices)
    within_k = torch.arange(max_k, device=device).unsqueeze(0) < k.unsqueeze(1)
    all_in = ((top_positive & within_k).sum(dim=1) == p_count).to(dtype=value.dtype)
    multi = p_count > 1
    replacement = set_nll + float(config.rank_weight) * rank_loss
    loss = torch.where(multi, replacement, native)
    zeros = torch.zeros_like(loss)
    return MiniTopKBatchResult(
        loss=loss,
        set_nll=torch.where(multi, set_nll, zeros),
        rank_loss=torch.where(multi, rank_loss, zeros),
        native_ce=native,
        all_in=all_in,
        weakest_gap=torch.where(multi, gap, zeros),
        boundary_violation=torch.where(multi, violation, zeros),
        positive_count=p_count,
        k=k,
        multi=multi,
    )
