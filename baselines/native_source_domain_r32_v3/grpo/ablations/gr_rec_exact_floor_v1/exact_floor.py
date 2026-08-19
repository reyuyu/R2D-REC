"""Pure CPU/GPU-agnostic math for the ExactFloor advantage ablation."""

from __future__ import annotations

import torch


EXACT_FLOOR = 8.0
FIXED_SCALE = 8.0


def exact_floor_advantages(
    rewards: torch.Tensor,
    group_size: int,
    exact_floor: float = EXACT_FLOOR,
    fixed_scale: float = FIXED_SCALE,
) -> torch.Tensor:
    """Return ``(R - min(group_mean, exact_floor)) / fixed_scale`` per group."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape={tuple(rewards.shape)}")
    if not rewards.is_floating_point():
        raise TypeError("rewards must use a floating dtype")
    if group_size <= 0 or rewards.numel() % group_size:
        raise ValueError(
            f"reward count {rewards.numel()} must be divisible by group_size={group_size}"
        )
    if exact_floor <= 0 or fixed_scale <= 0:
        raise ValueError("exact_floor and fixed_scale must be positive")
    grouped = rewards.reshape(-1, group_size)
    baselines = grouped.mean(dim=1).clamp(max=exact_floor)
    return ((grouped - baselines.unsqueeze(1)) / fixed_scale).reshape_as(rewards)
