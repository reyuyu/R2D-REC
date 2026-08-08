"""Normalized SID-token weighted causal SFT cross-entropy.

The optional recommendation role-aware mode changes only the token weight map:
SID component tokens inside a supervised ``<think>...</think>`` span use the
text weight, while final answer SID components retain the configured SID
weight.  It never adds another model forward or loss term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    "rec_think_sid_token_count",
    "rec_final_sid_token_count",
    "rec_think_sid_ce_sum",
    "rec_final_sid_ce_sum",
    "rec_think_sid_weighted_mass",
    "rec_final_sid_weighted_mass",
)
SID_WEIGHT_STAT_INDEX = {name: index for index, name in enumerate(SID_WEIGHT_STAT_NAMES)}
SID_WEIGHT_STAT_SIZE = len(SID_WEIGHT_STAT_NAMES)


@dataclass
class SidTokenWeightingResult:
    loss: torch.Tensor
    statistics: torch.Tensor


class SidTokenWeightingController:
    """Apply normalized SID weights, optionally separated by recommendation role."""

    def __init__(self, tokenizer, data_args) -> None:
        semantic_ids = build_action_semantic_token_ids(tokenizer)
        self._sid_token_ids = tuple(sorted(set().union(semantic_ids["a"], semantic_ids["b"], semantic_ids["c"])))
        if not self._sid_token_ids:
            raise ValueError("SID token weighting found no <s_a_*>, <s_b_*>, or <s_c_*> token IDs.")
        self.sid_weight = float(data_args.sid_token_weight)
        self.text_weight = float(data_args.sid_text_weight)
        self.role_aware_enabled = bool(getattr(data_args, "recommendation_role_aware_sid_weighting_enabled", False))
        self.recommendation_think_sid_weight = float(
            getattr(data_args, "recommendation_think_sid_weight", self.text_weight)
        )
        self.recommendation_final_sid_weight = float(
            getattr(data_args, "recommendation_final_sid_weight", self.sid_weight)
        )
        if self.sid_weight <= 0 or self.text_weight <= 0:
            raise ValueError("sid_token_weight and sid_text_weight must both be positive.")
        if self.recommendation_think_sid_weight <= 0 or self.recommendation_final_sid_weight <= 0:
            raise ValueError("Recommendation role-aware SID weights must both be positive.")
        self._lookup_by_device: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._think_open_ids = self._encode_marker(tokenizer, "<think>")
        self._think_close_ids = self._encode_marker(tokenizer, "</think>")

    @staticmethod
    def _encode_marker(tokenizer: Any, marker: str) -> tuple[int, ...]:
        """Encode a marker with the active tokenizer, without decoding batches."""
        try:
            if hasattr(tokenizer, "encode"):
                value = tokenizer.encode(marker, add_special_tokens=False)
                if isinstance(value, dict):
                    value = value.get("input_ids", [])
                if torch.is_tensor(value):
                    value = value.flatten().tolist()
                if value:
                    return tuple(int(item) for item in value)
        except Exception:
            pass
        try:
            added_vocab = tokenizer.get_added_vocab()
            if marker in added_vocab:
                return (int(added_vocab[marker]),)
        except Exception:
            pass
        return ()

    @staticmethod
    def _find(values: list[int], sequence: tuple[int, ...], start: int = 0) -> int | None:
        if not sequence or len(sequence) > len(values):
            return None
        limit = len(values) - len(sequence) + 1
        for index in range(start, limit):
            if tuple(values[index : index + len(sequence)]) == sequence:
                return index
        return None

    def _role_masks(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build independent think/final masks for each valid label run.

        In a packed row, prompt labels and segment boundaries are IGNORE_INDEX.
        Splitting on those runs prevents a marker in one segment from affecting
        the next segment. If a segment has no complete marker pair, all of its
        supervised positions conservatively use the final-answer role.
        """
        think = torch.zeros_like(labels, dtype=torch.bool)
        final = torch.zeros_like(labels, dtype=torch.bool)
        if not self._think_open_ids or not self._think_close_ids:
            final.copy_(labels.ne(IGNORE_INDEX))
            return think, final
        for row_index in range(labels.shape[0]):
            row = labels[row_index]
            valid = row.ne(IGNORE_INDEX).tolist()
            position = 0
            while position < len(valid):
                while position < len(valid) and not valid[position]:
                    position += 1
                start = position
                while position < len(valid) and valid[position]:
                    position += 1
                end = position
                if start >= end:
                    continue
                values = [int(value) for value in row[start:end].tolist()]
                open_at = self._find(values, self._think_open_ids)
                close_at = self._find(
                    values,
                    self._think_close_ids,
                    (open_at + len(self._think_open_ids)) if open_at is not None else 0,
                )
                if open_at is None or close_at is None:
                    final[row_index, start:end] = True
                    continue
                think_start = start + open_at + len(self._think_open_ids)
                think_end = start + close_at
                final_start = start + close_at + len(self._think_close_ids)
                if think_start < think_end:
                    think[row_index, think_start:think_end] = True
                if final_start < end:
                    final[row_index, final_start:end] = True
        return think, final

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

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task_name: str | None = None,
        subtask_name: str | None = None,
    ) -> SidTokenWeightingResult:
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

        role_route = task_name == "recommendation" and subtask_name in {"cot", "nocot"}
        if role_route:
            think_full, final_full = self._role_masks(labels)
            rec_think_mask = sid_mask & think_full[:, 1:]
            rec_final_mask = sid_mask & final_full[:, 1:]
            # A malformed or truncated marker pair must never make a SID lose
            # supervision; unclassified recommendation SID positions stay final.
            rec_final_mask = rec_final_mask | (sid_mask & ~(rec_think_mask | rec_final_mask))
        else:
            rec_think_mask = torch.zeros_like(sid_mask)
            rec_final_mask = torch.zeros_like(sid_mask)

        per_token_ce = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(shift_labels)
        weights = torch.full_like(per_token_ce, self.text_weight)
        weights = torch.where(sid_mask, torch.full_like(weights, self.sid_weight), weights)
        if self.role_aware_enabled and role_route and subtask_name == "cot":
            weights = torch.where(
                rec_think_mask,
                torch.full_like(weights, self.recommendation_think_sid_weight),
                weights,
            )
            weights = torch.where(
                rec_final_mask,
                torch.full_like(weights, self.recommendation_final_sid_weight),
                weights,
            )
        elif self.role_aware_enabled and role_route and subtask_name == "nocot":
            weights = torch.where(
                rec_final_mask,
                torch.full_like(weights, self.recommendation_final_sid_weight),
                weights,
            )
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
        statistics[SID_WEIGHT_STAT_INDEX["sid_weighted_mass"]] = (weights * sid_float).sum().detach()
        statistics[SID_WEIGHT_STAT_INDEX["total_weighted_mass"]] = total_weighted_mass.detach()
        statistics[SID_WEIGHT_STAT_INDEX["weighted_loss_sum"]] = loss.detach()
        statistics[SID_WEIGHT_STAT_INDEX["weighted_loss_count"]] = 1
        if role_route:
            think_float = rec_think_mask.to(per_token_ce.dtype)
            final_float = rec_final_mask.to(per_token_ce.dtype)
            statistics[SID_WEIGHT_STAT_INDEX["rec_think_sid_token_count"]] = think_float.sum()
            statistics[SID_WEIGHT_STAT_INDEX["rec_final_sid_token_count"]] = final_float.sum()
            statistics[SID_WEIGHT_STAT_INDEX["rec_think_sid_ce_sum"]] = (per_token_ce * think_float).sum().detach()
            statistics[SID_WEIGHT_STAT_INDEX["rec_final_sid_ce_sum"]] = (per_token_ce * final_float).sum().detach()
            statistics[SID_WEIGHT_STAT_INDEX["rec_think_sid_weighted_mass"]] = (weights * think_float).sum().detach()
            statistics[SID_WEIGHT_STAT_INDEX["rec_final_sid_weighted_mass"]] = (weights * final_float).sum().detach()
        return SidTokenWeightingResult(loss=loss, statistics=statistics)


def sid_weight_statistics_to_metrics(statistics: torch.Tensor) -> dict[str, float]:
    """Convert globally summed SID statistics into compact monitoring fields."""
    values = statistics.detach().float().cpu()
    get = lambda name: float(values[SID_WEIGHT_STAT_INDEX[name]].item())
    sid_count = get("sid_token_count")
    supervised_count = get("supervised_token_count")
    sid_ce_sum = get("sid_ce_sum")
    text_count = get("text_token_count")
    text_ce_sum = get("text_ce_sum")
    sid_mass = get("sid_weighted_mass")
    total_mass = get("total_weighted_mass")
    weighted_loss_sum = get("weighted_loss_sum")
    weighted_loss_count = get("weighted_loss_count")
    think_count = get("rec_think_sid_token_count")
    final_count = get("rec_final_sid_token_count")
    return {
        "sid_weighted_token_count": sid_count,
        "sid_supervised_token_ratio": sid_count / supervised_count if supervised_count else 0.0,
        "sid_effective_weight": total_mass / supervised_count if supervised_count else 0.0,
        "sid_token_ce": sid_ce_sum / sid_count if sid_count else 0.0,
        "text_token_ce": text_ce_sum / text_count if text_count else 0.0,
        "sid_weighted_mass_share": sid_mass / total_mass if total_mass else 0.0,
        "sid_weighted_total_loss": weighted_loss_sum / weighted_loss_count if weighted_loss_count else 0.0,
        "rec_think_sid_token_count": think_count,
        "rec_final_sid_token_count": final_count,
        "rec_think_sid_ce": get("rec_think_sid_ce_sum") / think_count if think_count else 0.0,
        "rec_final_sid_ce": get("rec_final_sid_ce_sum") / final_count if final_count else 0.0,
        "rec_think_sid_weighted_mass": get("rec_think_sid_weighted_mass"),
        "rec_final_sid_weighted_mass": get("rec_final_sid_weighted_mass"),
    }
