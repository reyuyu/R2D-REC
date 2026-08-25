"""Frontier PPO and multi-positive HPR loss contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


HPR_LAMBDA = 0.02
FRONTIER_REDUCTION = "candidate credited-token sum -> G8 candidate mean -> business-group batch mean"
HPR_SITE_REDUCTION = "mean over on-policy positions within site"
HPR_GROUP_REDUCTION = "mean over sites within business group; HPR_NONE=0"


@dataclass(frozen=True)
class FrontierLoss:
    loss: torch.Tensor
    candidate_losses: torch.Tensor
    group_losses: torch.Tensor
    ratios: torch.Tensor
    clipped_ratios: torch.Tensor
    token_losses: torch.Tensor


@dataclass(frozen=True)
class TotalLoss:
    frontier_loss: torch.Tensor
    hpr_loss_raw: torch.Tensor
    hpr_loss_weighted: torch.Tensor
    total_loss: torch.Tensor


def frontier_ppo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    token_credits: torch.Tensor,
    token_credit_mask: torch.Tensor,
    epsilon: float,
) -> FrontierLoss:
    expected = current_logps.shape
    if current_logps.ndim != 3 or expected[-1] != 3:
        raise ValueError("Frontier tensors must have shape [B,G,3]")
    if old_logps.shape != expected or token_credits.shape != expected or token_credit_mask.shape != expected:
        raise ValueError("Frontier tensor shapes must match")
    if token_credit_mask.dtype != torch.bool:
        raise TypeError("token_credit_mask must be bool")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0,1)")
    ratios = torch.exp(current_logps - old_logps)
    clipped = ratios.clamp(1.0 - epsilon, 1.0 + epsilon)
    unclipped_objective = ratios * token_credits
    clipped_objective = clipped * token_credits
    token_losses = -torch.minimum(unclipped_objective, clipped_objective)
    token_losses = torch.where(token_credit_mask, token_losses, torch.zeros_like(token_losses))
    candidate_losses = token_losses.sum(dim=-1)
    group_losses = candidate_losses.mean(dim=-1)
    return FrontierLoss(group_losses.mean(), candidate_losses, group_losses, ratios, clipped, token_losses)


def multi_positive_log_mass_loss(logits: torch.Tensor, target_token_ids: Sequence[int]) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("site logits must be one vocabulary vector")
    unique_targets = sorted({int(token_id) for token_id in target_token_ids})
    if not unique_targets:
        raise ValueError("multi-positive target set cannot be empty")
    if unique_targets[0] < 0 or unique_targets[-1] >= logits.shape[0]:
        raise IndexError("target token ID outside vocabulary")
    targets = torch.tensor(unique_targets, dtype=torch.long, device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_probs.index_select(0, targets), dim=0)


def compose_total_loss(frontier_loss: torch.Tensor, hpr_loss_raw: torch.Tensor) -> TotalLoss:
    weighted = hpr_loss_raw * HPR_LAMBDA
    return TotalLoss(frontier_loss, hpr_loss_raw, weighted, frontier_loss + weighted)
