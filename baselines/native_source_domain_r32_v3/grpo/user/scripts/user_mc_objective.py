"""Pure unit-level marginal-credit objective for MC_USER_v1."""

from __future__ import annotations

import math
import operator
from collections import Counter
from typing import Any, Mapping, Sequence

import torch


class MCObjectiveError(ValueError):
    pass


def _unit_indices(unit: Mapping[str, Any], width: int) -> list[int]:
    raw_indices = unit.get("generated_token_indices")
    if raw_indices is None:
        raise MCObjectiveError("credit unit is missing generated_token_indices")
    try:
        indices = [operator.index(index) for index in raw_indices]
    except TypeError as exc:
        raise MCObjectiveError("generated token indices must be integers") from exc
    if not indices:
        raise MCObjectiveError("generated token indices must be non-empty")
    if len(indices) != len(set(indices)):
        raise MCObjectiveError("generated token indices must not contain duplicates")
    if any(index < 0 or index >= width for index in indices):
        raise MCObjectiveError("generated token index is outside [0,T)")
    return indices


def mc_unit_credit_loss(
    per_token_logps: torch.Tensor,
    credit_units_per_candidate: Sequence[Sequence[Mapping[str, Any]]],
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute mean-over-candidates, sum-over-units MC credit loss.

    Cross-unit token overlap is algebraic: each unit retains its independent
    span-mean loss, so shared-token logp coefficients add naturally.
    """

    if not torch.is_tensor(per_token_logps) or not torch.is_tensor(completion_mask):
        raise TypeError("per_token_logps and completion_mask must be tensors")
    if per_token_logps.ndim != 2 or per_token_logps.numel() == 0:
        raise MCObjectiveError("per_token_logps must be a non-empty [B,T] tensor")
    if not per_token_logps.is_floating_point():
        raise MCObjectiveError("per_token_logps must use a floating dtype")
    if completion_mask.shape != per_token_logps.shape:
        raise MCObjectiveError("completion_mask must match per_token_logps [B,T]")

    batch_size, width = per_token_logps.shape
    if len(credit_units_per_candidate) != batch_size:
        raise MCObjectiveError("credit-unit candidate count must equal batch size")

    candidate_losses = []
    unit_records = []
    type_counts: Counter[str] = Counter()
    positive_credit_mass = 0.0
    negative_credit_mass = 0.0
    active_unit_count = 0
    active_token_count = 0
    active_token_assignment_count = 0
    overlap_records = []
    same_sign_overlap_token_count = 0
    mixed_sign_overlap_token_count = 0
    max_active_units_per_token = 0

    for candidate_index, units in enumerate(credit_units_per_candidate):
        candidate_loss = per_token_logps[candidate_index].sum() * 0.0
        active_indices: set[int] = set()
        active_units_by_token: dict[int, list[dict[str, Any]]] = {}
        for unit_index, unit in enumerate(units):
            indices = _unit_indices(unit, width)
            mask_values = completion_mask[candidate_index, indices]
            if not bool(torch.all(mask_values == 1)):
                raise MCObjectiveError("credit unit touches masked or padded completion tokens")

            delta = float(unit["delta"])
            if not math.isfinite(delta):
                raise MCObjectiveError("credit unit delta must be finite")
            if delta > 0.0:
                credit_type = "positive"
                positive_credit_mass += delta
            elif delta < 0.0:
                credit_type = "negative"
                negative_credit_mass += abs(delta)
            else:
                credit_type = "zero"
            type_counts[credit_type] += 1

            if delta != 0.0:
                active_indices.update(indices)
                active_unit_count += 1
                active_token_assignment_count += len(indices)
                coefficient = -delta / len(indices)
                for token_index in indices:
                    active_units_by_token.setdefault(token_index, []).append(
                        {
                            "unit_index": unit_index,
                            "delta": delta,
                            "per_unit_logp_coefficient": coefficient,
                        }
                    )

            unit_mean_logp = per_token_logps[candidate_index, indices].mean()
            unit_loss = -delta * unit_mean_logp
            candidate_loss = candidate_loss + unit_loss
            unit_records.append(
                {
                    "candidate_index": candidate_index,
                    "unit_index": unit_index,
                    "delta": delta,
                    "credit_type": credit_type,
                    "generated_token_indices": list(indices),
                    "token_count": len(indices),
                    "unit_mean_logp": unit_mean_logp,
                    "unit_loss": unit_loss,
                }
            )
        # Count token instances uniquely within each candidate; assignments
        # retain multiplicity across independently scored units.
        active_token_count += len(active_indices)
        max_active_units_per_token = max(
            max_active_units_per_token,
            max(
                (len(assignments) for assignments in active_units_by_token.values()),
                default=0,
            ),
        )
        for token_index in sorted(active_units_by_token):
            assignments = active_units_by_token[token_index]
            if len(assignments) < 2:
                continue
            deltas = [assignment["delta"] for assignment in assignments]
            coefficients = [
                assignment["per_unit_logp_coefficient"] for assignment in assignments
            ]
            if all(delta > 0.0 for delta in deltas) or all(
                delta < 0.0 for delta in deltas
            ):
                same_sign_overlap_token_count += 1
            else:
                mixed_sign_overlap_token_count += 1
            overlap_records.append(
                {
                    "candidate_index": candidate_index,
                    "token_index": token_index,
                    "unit_indices": [
                        assignment["unit_index"] for assignment in assignments
                    ],
                    "deltas": deltas,
                    "per_unit_logp_coefficients": coefficients,
                    "net_logp_coefficient": sum(coefficients),
                }
            )
        candidate_losses.append(candidate_loss)

    candidate_losses_tensor = torch.stack(candidate_losses)
    loss = candidate_losses_tensor.mean()
    metadata = {
        "candidate_losses": candidate_losses_tensor,
        "positive_unit_count": type_counts["positive"],
        "negative_unit_count": type_counts["negative"],
        "zero_unit_count": type_counts["zero"],
        "positive_credit_mass": positive_credit_mass,
        "negative_credit_mass": negative_credit_mass,
        "active_unit_count": active_unit_count,
        "active_token_count": active_token_count,
        "active_token_assignment_count": active_token_assignment_count,
        "overlap_token_count": len(overlap_records),
        "same_sign_overlap_token_count": same_sign_overlap_token_count,
        "mixed_sign_overlap_token_count": mixed_sign_overlap_token_count,
        "max_active_units_per_token": max_active_units_per_token,
        "overlap_records": overlap_records,
        "unit_records": unit_records,
    }
    return loss, metadata


__all__ = ["MCObjectiveError", "mc_unit_credit_loss"]
