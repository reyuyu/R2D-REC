"""Pure math for the Think-only Centered Exact-Clamp rule."""

from __future__ import annotations

import torch


THINK_GROUP_SIZE = 4
FIXED_SCALE = 8.0
EXACT_THRESHOLD = 8.0


def think_exact_clamp_advantages(
    rewards: torch.Tensor,
    group_size: int = THINK_GROUP_SIZE,
    fixed_scale: float = FIXED_SCALE,
    exact_threshold: float = EXACT_THRESHOLD,
) -> torch.Tensor:
    """Center G4 rewards and clamp negative advantages only for reward >= 8."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape={tuple(rewards.shape)}")
    if not rewards.is_floating_point():
        raise TypeError("rewards must use a floating dtype")
    if group_size != THINK_GROUP_SIZE:
        raise ValueError(f"Think ExactClamp requires group_size=4, got {group_size}")
    if rewards.numel() % group_size:
        raise ValueError(f"reward count {rewards.numel()} must be divisible by 4")
    if fixed_scale <= 0:
        raise ValueError("fixed_scale must be positive")

    grouped = rewards.reshape(-1, group_size)
    raw = (grouped - grouped.mean(dim=1, keepdim=True)) / fixed_scale
    clamp_mask = (grouped >= exact_threshold) & (raw < 0)
    return torch.where(clamp_mask, torch.zeros_like(raw), raw).reshape_as(rewards)
