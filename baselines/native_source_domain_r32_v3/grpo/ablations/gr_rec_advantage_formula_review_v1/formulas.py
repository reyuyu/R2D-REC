"""Advantage formulas used by the CPU-only comparison audit."""

from __future__ import annotations

import torch


def centered_exact_clamp_advantages(
    rewards: torch.Tensor,
    group_size: int,
    fixed_scale: float = 8.0,
    exact_threshold: float = 8.0,
) -> torch.Tensor:
    """Center by group mean, then clamp negative high-quality advantages to zero."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape={tuple(rewards.shape)}")
    if not rewards.is_floating_point():
        raise TypeError("rewards must use a floating dtype")
    if group_size <= 0 or rewards.numel() % group_size:
        raise ValueError(
            f"reward count {rewards.numel()} must be divisible by group_size={group_size}"
        )
    if fixed_scale <= 0:
        raise ValueError("fixed_scale must be positive")

    grouped = rewards.reshape(-1, group_size)
    raw = (grouped - grouped.mean(dim=1, keepdim=True)) / fixed_scale
    clamp_mask = (grouped >= exact_threshold) & (raw < 0)
    return torch.where(clamp_mask, torch.zeros_like(raw), raw).reshape_as(rewards)
