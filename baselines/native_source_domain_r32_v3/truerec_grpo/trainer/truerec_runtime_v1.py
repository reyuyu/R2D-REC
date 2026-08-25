"""Runtime adapters that reuse Phase 0.8 Frontier/HPR plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from action_alignment import align_action
from frontier_credit_v1 import GATED, NEGATIVE, POSITIVE, plan_frontier_credit
from hpr_plan_v1 import HPRGroupPlan, plan_hpr, validate_plan
from truerec_loss_v1 import HPR_GROUP_REDUCTION, HPR_SITE_REDUCTION, multi_positive_log_mass_loss


@dataclass(frozen=True)
class RuntimeHPRSite:
    level: str
    target_token_ids: tuple[int, ...]
    onpolicy_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RuntimeHPRPlan:
    trigger: str
    sites: tuple[RuntimeHPRSite, ...]


@dataclass(frozen=True)
class GroupRuntimePlan:
    token_credits: torch.Tensor
    token_credit_mask: torch.Tensor
    hpr: RuntimeHPRPlan
    monitoring: dict[str, float | int]


def build_runtime_hpr_plan(
    candidates: Sequence[dict[str, Any]],
    all_gold_abc: Sequence[str],
    token_to_id: Callable[[str], int],
) -> RuntimeHPRPlan:
    source: HPRGroupPlan = plan_hpr(candidates, all_gold_abc)
    validate_plan(source)
    sites = []
    for site in source.sites:
        targets = tuple(sorted({int(token_to_id(token)) for token in site.target_tokens}))
        if not targets:
            raise ValueError("runtime HPR target set cannot be empty")
        sites.append(RuntimeHPRSite(site.level, targets, site.onpolicy_positions))
    return RuntimeHPRPlan(source.trigger, tuple(sites))


def build_group_runtime_plan(
    candidates: Sequence[dict[str, Any]],
    all_gold_abc: Sequence[str],
    token_to_id: Callable[[str], int],
) -> GroupRuntimePlan:
    frontier = plan_frontier_credit(candidates)
    credits = torch.tensor([candidate.credits for candidate in frontier.candidates], dtype=torch.float32)
    mask = torch.tensor([[kind != GATED for kind in candidate.kinds] for candidate in frontier.candidates], dtype=torch.bool)
    hpr = build_runtime_hpr_plan(candidates, all_gold_abc, token_to_id)
    kinds = [candidate.kinds for candidate in frontier.candidates]
    monitoring = {
        "candidate_count": len(candidates),
        "A_hit_rate": sum(item["A_hit"] for item in candidates) / len(candidates),
        "AB_hit_rate": sum(item["AB_hit"] for item in candidates) / len(candidates),
        "exact_rate": sum(item["exact"] for item in candidates) / len(candidates),
        "frontier_A_positive": sum(row[0] == POSITIVE for row in kinds),
        "frontier_A_negative": sum(row[0] == NEGATIVE for row in kinds),
        "frontier_B_positive": sum(row[1] == POSITIVE for row in kinds),
        "frontier_B_negative": sum(row[1] == NEGATIVE for row in kinds),
        "frontier_C_positive": sum(row[2] == POSITIVE for row in kinds),
        "frontier_C_negative": sum(row[2] == NEGATIVE for row in kinds),
        "wrong_history_copy_rate": sum(item.get("wrong_history_copy", False) for item in candidates) / len(candidates),
    }
    return GroupRuntimePlan(credits, mask, hpr, monitoring)


def sampled_action_logps(policy_logits: torch.Tensor, context_length: int, completion_ids: torch.Tensor) -> torch.Tensor:
    if policy_logits.ndim != 3 or completion_ids.ndim != 2 or completion_ids.shape[1] != 3:
        raise ValueError("expected logits [G,L,V] and completion_ids [G,3]")
    if policy_logits.shape[0] != completion_ids.shape[0]:
        raise ValueError("candidate batch mismatch")
    alignment = align_action(context_length, completion_ids[0].tolist())
    positions = torch.tensor(alignment.logit_indices, device=policy_logits.device)
    action_logits = policy_logits.index_select(1, positions)
    log_probs = torch.log_softmax(action_logits, dim=-1)
    return log_probs.gather(-1, completion_ids.to(policy_logits.device).unsqueeze(-1)).squeeze(-1)


def hpr_loss_from_shared_logits(policy_logits: torch.Tensor, context_length: int, plan: RuntimeHPRPlan) -> torch.Tensor:
    if not plan.sites:
        return policy_logits.sum() * 0.0
    site_losses = []
    for site in plan.sites:
        position_losses = []
        for sample_index, action_position in site.onpolicy_positions:
            if action_position not in (0, 1, 2):
                raise ValueError("HPR position must be A/B/C")
            causal_position = context_length - 1 + action_position
            position_losses.append(multi_positive_log_mass_loss(policy_logits[sample_index, causal_position], site.target_token_ids))
        if not position_losses:
            raise ValueError("HPR site lacks on-policy positions")
        site_losses.append(torch.stack(position_losses).mean())
    return torch.stack(site_losses).mean()
