"""Cheap detached statistics for the native Set-PU packed loss path.

The helpers deliberately operate only on tensors that the loss path already
selected.  They never form a full-vocabulary probability tensor and their
outputs are detached sum/count pairs suitable for DDP reduction at log time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


_COMPONENT_INDEX_CACHE: dict[tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class CandidateRankOutcome:
    """Teacher-forced candidate results for aligned final SID positions."""

    hit8: torch.Tensor
    hit32: torch.Tensor
    coverage8: torch.Tensor
    coverage32: torch.Tensor


def finite_sum_count(values: torch.Tensor) -> torch.Tensor:
    """Return detached ``[sum, count]`` for finite values only."""

    flat = values.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(flat)
    return torch.stack((torch.where(finite, flat, torch.zeros_like(flat)).sum(), finite.sum().to(torch.float64)))


def positive_entropy_sum_count(
    selected_logits: torch.Tensor, positive_ids_by_position: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Return normalized positive-set entropy ``[sum, count]``.

    This gathers only the observed positive IDs. Singleton sets intentionally
    contribute neither a value nor a denominator count.
    """

    if selected_logits.ndim != 2 or len(positive_ids_by_position) != selected_logits.size(0):
        raise ValueError("Expected [positions, vocab] logits and one positive set per position.")
    if not positive_ids_by_position:
        return torch.zeros(2, device=selected_logits.device, dtype=torch.float64)

    width = max(len(ids) for ids in positive_ids_by_position)
    widths = torch.tensor([len(ids) for ids in positive_ids_by_position], device=selected_logits.device)
    # One compact transfer per level, never one transfer or scalar sync per
    # position. Padding IDs are masked before use.
    positive_ids = torch.tensor(
        [list(ids) + [0] * (width - len(ids)) for ids in positive_ids_by_position],
        dtype=torch.long,
        device=selected_logits.device,
    )
    valid = torch.arange(width, device=selected_logits.device).unsqueeze(0) < widths.unsqueeze(1)

    # Detach before gathering so metrics cannot extend the backward graph.
    positives = selected_logits.detach().float().gather(1, positive_ids).masked_fill(~valid, float("-inf"))
    log_q = torch.log_softmax(positives, dim=1)
    q = torch.exp(log_q)
    entropy = -(torch.where(valid, q * log_q, torch.zeros_like(q))).sum(dim=1)
    eligible = widths > 1
    log_cardinality = widths.to(dtype=entropy.dtype).log()
    normalized = entropy / torch.where(eligible, log_cardinality, torch.ones_like(log_cardinality))
    return finite_sum_count(normalized[eligible])


def pack_effective_share_sum_count(
    sample_task_ids: torch.Tensor, sample_row_ids: torch.Tensor, batch_size: int, num_tasks: int
) -> torch.Tensor:
    """Mean each packed row equally, independent of segment token length.

    The returned ``[task, sum/count]`` values implement the required
    pack-mean semantics, not a global segment-count ratio.
    """

    if sample_task_ids.shape != sample_row_ids.shape:
        raise ValueError("sample_task_ids and sample_row_ids must align.")
    stats = torch.zeros((num_tasks, 2), device=sample_task_ids.device, dtype=torch.float64)
    for row in range(int(batch_size)):
        row_tasks = sample_task_ids[sample_row_ids == row]
        counts = torch.bincount(row_tasks, minlength=num_tasks)[:num_tasks].to(dtype=torch.float64)
        total = counts.sum()
        present = (total > 0).to(dtype=torch.float64)
        stats[:, 0].add_(counts / total.clamp_min(1.0))
        stats[:, 1].add_(present)
    return stats


def candidate_window_active(global_step: int, enabled: bool, interval: int) -> bool:
    """Whether the current pre-update GA window belongs to the sampling step."""

    if interval < 1:
        raise ValueError("Candidate metric interval must be >= 1.")
    return bool(enabled) and (int(global_step) + 1) % int(interval) == 0


def _component_indices(vocab_size: int, component_ids: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tuple(int(value) for value in component_ids)
    key = (str(device), int(vocab_size), ids)
    cached = _COMPONENT_INDEX_CACHE.get(key)
    if cached is None:
        global_ids = torch.tensor(ids, dtype=torch.long, device=device)
        # The dense lookup is cached once per component/device. It maps a
        # global token id into the small legal component-vocab index.
        local_index = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        local_index[global_ids] = torch.arange(global_ids.numel(), device=device)
        cached = (global_ids, local_index)
        _COMPONENT_INDEX_CACHE[key] = cached
    return cached


def candidate_rank_outcome(
    selected_logits: torch.Tensor,
    positive_ids_by_position: Sequence[Sequence[int]],
    component_ids: Sequence[int],
    *,
    max_k: int,
) -> CandidateRankOutcome:
    """Rank only legal same-level SID tokens, entirely in batched tensors.

    Returned values are detached and aligned with ``positive_ids_by_position``.
    They are teacher-forced candidate proxies, never autoregressive metrics.
    """

    if selected_logits.ndim != 2 or len(positive_ids_by_position) != selected_logits.size(0):
        raise ValueError("Expected [positions, vocab] logits and one positive set per position.")
    if max_k < 1 or not component_ids:
        raise ValueError("max_k and component_ids must be non-empty.")
    positions = len(positive_ids_by_position)
    if positions == 0:
        zero = torch.zeros(0, dtype=torch.float32, device=selected_logits.device)
        return CandidateRankOutcome(zero, zero, zero, zero)

    global_ids, local_index = _component_indices(selected_logits.size(1), component_ids, selected_logits.device)
    width = max(len(ids) for ids in positive_ids_by_position)
    if width < 1:
        raise ValueError("Every candidate position needs at least one observed positive.")
    positive_global = torch.tensor(
        [list(ids) + [0] * (width - len(ids)) for ids in positive_ids_by_position],
        dtype=torch.long,
        device=selected_logits.device,
    )
    positive_widths = torch.tensor([len(ids) for ids in positive_ids_by_position], device=selected_logits.device)
    positive_valid = torch.arange(width, device=selected_logits.device).unsqueeze(0) < positive_widths.unsqueeze(1)
    # Phase-2/Set-PU has already validated this subset relation. Avoid adding
    # a diagnostic-only device-to-host synchronization here.
    positive_local = local_index[positive_global]

    # No full-vocabulary top-k: select the legal component logits first.
    # Gather the legal component slice before fp32 promotion. This avoids a
    # transient [positions, full_vocab] fp32 diagnostic tensor.
    component_logits = selected_logits.detach().index_select(1, global_ids).float()
    top_k = min(int(max_k), component_logits.size(1))
    top32 = torch.topk(component_logits, k=top_k, dim=1).indices

    def _hit_and_coverage(top_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membership = (top_local.unsqueeze(2) == positive_local.unsqueeze(1)) & positive_valid.unsqueeze(1)
        found_positive = membership.any(dim=1)
        hit = found_positive.any(dim=1).to(dtype=torch.float32)
        coverage = found_positive.to(dtype=torch.float32).sum(dim=1) / positive_widths.to(dtype=torch.float32)
        return hit, coverage

    hit32, coverage32 = _hit_and_coverage(top32)
    hit8, coverage8 = _hit_and_coverage(top32[:, : min(8, top_k)])
    return CandidateRankOutcome(hit8, hit32, coverage8, coverage32)


def candidate_chain_sum_count(
    a_outcome: CandidateRankOutcome, b_outcome: CandidateRankOutcome, c_outcome: CandidateRankOutcome
) -> torch.Tensor:
    """Return segment-aligned TF chain hits for a32/b8/c8."""

    lengths = {a_outcome.hit32.numel(), b_outcome.hit8.numel(), c_outcome.hit8.numel()}
    if len(lengths) != 1:
        raise ValueError("Candidate chain outcomes must stay aligned by RecPUPackedTarget.")
    chain = a_outcome.hit32.bool() & b_outcome.hit8.bool() & c_outcome.hit8.bool()
    return finite_sum_count(chain.to(dtype=torch.float32))
