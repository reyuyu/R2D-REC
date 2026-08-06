"""Normalized SID-token weighted causal SFT cross-entropy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...data.action_select import build_action_semantic_token_ids
from ...extras.constants import IGNORE_INDEX


SID_WEIGHT_STAT_NAMES = (
    "sid_token_count",
    "supervised_token_count",
    "sid_ce_sum",
    "text_token_count",
    "text_ce_sum",
    "sid_weighted_mass",
    "total_weighted_mass",
    "weighted_loss_sum",
    "weighted_loss_count",
)
SID_WEIGHT_STAT_INDEX = {name: index for index, name in enumerate(SID_WEIGHT_STAT_NAMES)}
SID_WEIGHT_STAT_SIZE = len(SID_WEIGHT_STAT_NAMES)


@dataclass
class SidTokenWeightingResult:
    loss: torch.Tensor
    statistics: torch.Tensor


class SidTokenWeightingController:
    """Apply one fixed normalized CE weight to supervised SID component tokens."""

    def __init__(self, tokenizer, data_args) -> None:
        semantic_ids = build_action_semantic_token_ids(tokenizer)
        # A full SID has a preceding domain marker plus a/b/c item tokens. The
        # reference ITEM_WEIGHT applies to the three item tokens, not the domain.
        self._sid_token_ids = tuple(sorted(set().union(semantic_ids["a"], semantic_ids["b"], semantic_ids["c"])))
        if not self._sid_token_ids:
            raise ValueError("SID token weighting found no <s_a_*>, <s_b_*>, or <s_c_*> token IDs.")
        self.sid_weight = float(data_args.sid_token_weight)
        self.text_weight = float(data_args.sid_text_weight)
        if self.sid_weight <= 0 or self.text_weight <= 0:
            raise ValueError("sid_token_weight and sid_text_weight must both be positive.")
        self._lookup_by_device: dict[tuple[torch.device, int], torch.Tensor] = {}

    def _sid_lookup(self, device: torch.device, vocab_size: int) -> torch.Tensor:
        key = (device, int(vocab_size))
        lookup = self._lookup_by_device.get(key)
        if lookup is None:
            largest_id = self._sid_token_ids[-1]
            if largest_id >= vocab_size:
                raise ValueError(
                    "The active tokenizer has SID token IDs outside the model vocabulary "
                    f"(largest={largest_id}, vocab_size={vocab_size})."
                )
            lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            lookup[torch.tensor(self._sid_token_ids, dtype=torch.long, device=device)] = True
            self._lookup_by_device[key] = lookup
        return lookup

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> SidTokenWeightingResult:
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError("SID token weighting expects logits [batch, sequence, vocab] and labels [batch, sequence].")
        if logits.shape[:2] != labels.shape:
            raise ValueError("SID token weighting logits and labels must have identical batch and sequence dimensions.")

        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(IGNORE_INDEX)
        vocab_size = shift_logits.shape[-1]
        lookup = self._sid_lookup(shift_labels.device, vocab_size)
        safe_labels = shift_labels.clamp(min=0, max=vocab_size - 1)
        sid_mask = valid_mask & lookup[safe_labels]
        text_mask = valid_mask & ~sid_mask

        per_token_ce = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(shift_labels)
        weights = torch.full_like(per_token_ce, self.text_weight)
        weights = torch.where(sid_mask, torch.full_like(weights, self.sid_weight), weights)
        valid_float = valid_mask.to(per_token_ce.dtype)
        weighted_mass = weights * valid_float
        total_weighted_mass = weighted_mass.sum()
        loss = (per_token_ce * weighted_mass).sum() / total_weighted_mass.clamp_min(1.0)

        statistics = torch.zeros(SID_WEIGHT_STAT_SIZE, dtype=torch.float32, device=logits.device)
        sid_float = sid_mask.to(per_token_ce.dtype)
        text_float = text_mask.to(per_token_ce.dtype)
        statistics[SID_WEIGHT_STAT_INDEX["sid_token_count"]] = sid_float.sum()
        statistics[SID_WEIGHT_STAT_INDEX["supervised_token_count"]] = valid_float.sum()
        statistics[SID_WEIGHT_STAT_INDEX["sid_ce_sum"]] = (per_token_ce * sid_float).sum().detach()
        statistics[SID_WEIGHT_STAT_INDEX["text_token_count"]] = text_float.sum()
        statistics[SID_WEIGHT_STAT_INDEX["text_ce_sum"]] = (per_token_ce * text_float).sum().detach()
        statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] = (self.sid_weight * sid_float.sum()).detach()
        statistics[SID_WEIGHT_STAT_INDEX["total_weighted_mass"]] = total_weighted_mass.detach()
        statistics[SID_WEIGHT_STAT_INDEX["weighted_loss_sum"]] = loss.detach()
        statistics[SID_WEIGHT_STAT_INDEX["weighted_loss_count"]] = 1
        return SidTokenWeightingResult(loss=loss, statistics=statistics)


def sid_weight_statistics_to_metrics(statistics: torch.Tensor) -> dict[str, float]:
    """Convert globally summed SID statistics into the compact monitoring fields."""
    values = statistics.detach().float().cpu()
    sid_count = float(values[SID_WEIGHT_STAT_INDEX["sid_token_count"]].item())
    supervised_count = float(values[SID_WEIGHT_STAT_INDEX["supervised_token_count"]].item())
    sid_ce_sum = float(values[SID_WEIGHT_STAT_INDEX["sid_ce_sum"]].item())
    text_count = float(values[SID_WEIGHT_STAT_INDEX["text_token_count"]].item())
    text_ce_sum = float(values[SID_WEIGHT_STAT_INDEX["text_ce_sum"]].item())
    sid_mass = float(values[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]].item())
    total_mass = float(values[SID_WEIGHT_STAT_INDEX["total_weighted_mass"]].item())
    weighted_loss_sum = float(values[SID_WEIGHT_STAT_INDEX["weighted_loss_sum"]].item())
    weighted_loss_count = float(values[SID_WEIGHT_STAT_INDEX["weighted_loss_count"]].item())
    return {
        "sid_weighted_token_count": sid_count,
        "sid_supervised_token_ratio": sid_count / supervised_count if supervised_count else 0.0,
        "sid_effective_weight": total_mass / supervised_count if supervised_count else 0.0,
        "sid_token_ce": sid_ce_sum / sid_count if sid_count else 0.0,
        "text_token_ce": text_ce_sum / text_count if text_count else 0.0,
        "sid_weighted_mass_share": sid_mass / total_mass if total_mass else 0.0,
        "sid_weighted_total_loss": weighted_loss_sum / weighted_loss_count if weighted_loss_count else 0.0,
    }
