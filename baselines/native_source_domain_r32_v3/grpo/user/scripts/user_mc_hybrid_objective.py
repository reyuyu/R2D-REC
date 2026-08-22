"""Pure sequence-GRPO plus marginal-credit objective for MC_USER ablations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from user_mc_objective import MCObjectiveError, mc_unit_credit_loss


ADVANTAGE_EPSILON = 1e-6
DEFAULT_SEQUENCE_WEIGHT = 1.0
DEFAULT_LOCAL_WEIGHT = 0.3


def mc_hybrid_objective(
    per_token_logps: torch.Tensor,
    rewards: Sequence[float] | torch.Tensor,
    credit_units_per_candidate: Sequence[Sequence[Mapping[str, Any]]],
    completion_mask: torch.Tensor,
    *,
    sequence_weight: float = DEFAULT_SEQUENCE_WEIGHT,
    local_weight: float = DEFAULT_LOCAL_WEIGHT,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine group-normalized sequence loss with the existing local loss."""

    if not torch.is_tensor(per_token_logps) or per_token_logps.ndim != 2:
        raise MCObjectiveError("per_token_logps must be a [B,T] tensor")
    if not per_token_logps.is_floating_point() or per_token_logps.numel() == 0:
        raise MCObjectiveError("per_token_logps must be non-empty and floating point")
    if not torch.is_tensor(completion_mask) or completion_mask.shape != per_token_logps.shape:
        raise MCObjectiveError("completion_mask must match per_token_logps [B,T]")
    if not math.isfinite(sequence_weight) or not math.isfinite(local_weight):
        raise MCObjectiveError("hybrid objective weights must be finite")

    batch_size = per_token_logps.shape[0]
    reward_tensor = torch.as_tensor(
        rewards,
        dtype=per_token_logps.dtype,
        device=per_token_logps.device,
    ).detach()
    if reward_tensor.ndim != 1 or reward_tensor.shape[0] != batch_size:
        raise MCObjectiveError("rewards must contain one scalar per candidate")
    if not bool(torch.isfinite(reward_tensor).all()):
        raise MCObjectiveError("rewards must be finite")

    real_token_mask = completion_mask.to(dtype=per_token_logps.dtype)
    if not bool(torch.all((real_token_mask == 0) | (real_token_mask == 1))):
        raise MCObjectiveError("completion_mask must contain only 0/1 values")
    real_token_counts = real_token_mask.sum(dim=1)
    if bool(torch.any(real_token_counts == 0)):
        raise MCObjectiveError("each candidate must contain at least one real completion token")

    reward_mean = reward_tensor.mean()
    reward_std = reward_tensor.std(unbiased=False)
    if float(reward_std) < ADVANTAGE_EPSILON:
        advantages = torch.zeros_like(reward_tensor)
    else:
        advantages = (reward_tensor - reward_mean) / (reward_std + ADVANTAGE_EPSILON)

    completion_mean_logps = (
        per_token_logps * real_token_mask
    ).sum(dim=1) / real_token_counts
    candidate_sequence_losses = -advantages * completion_mean_logps
    sequence_loss = candidate_sequence_losses.mean()

    local_loss, local_metadata = mc_unit_credit_loss(
        per_token_logps,
        credit_units_per_candidate,
        completion_mask,
    )
    total_loss = sequence_weight * sequence_loss + local_weight * local_loss
    metadata = {
        "group_reward_mean": float(reward_mean),
        "group_reward_std": float(reward_std),
        "group_reward_min": float(reward_tensor.min()),
        "group_reward_max": float(reward_tensor.max()),
        "group_reward_spread": float(reward_tensor.max() - reward_tensor.min()),
        "sequence_advantages": advantages,
        "candidate_sequence_losses": candidate_sequence_losses,
        "sequence_loss": sequence_loss,
        "local_loss": local_loss,
        "total_loss": total_loss,
        "sequence_weight": float(sequence_weight),
        "local_weight": float(local_weight),
        "local_objective_metadata": local_metadata,
    }
    return total_loss, metadata


__all__ = [
    "ADVANTAGE_EPSILON",
    "DEFAULT_LOCAL_WEIGHT",
    "DEFAULT_SEQUENCE_WEIGHT",
    "mc_hybrid_objective",
]
