"""Pure DSR objective math. No model calls or generation live here."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import torch

from .dsr_parser import InterestParseResult, Sid


EPS = 1e-4


def interest_count_gate(grounded_count: int) -> float:
    if grounded_count < 2:
        return 0.0
    if grounded_count <= 4:
        return 1.0
    return 0.5


def mean_pairwise_jaccard(evidence_sets: Sequence[set[Sid]]) -> float:
    if len(evidence_sets) < 2:
        return 1.0
    values = []
    for left, right in combinations(evidence_sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return sum(values) / len(values)


def cot_score(parsed: InterestParseResult) -> dict[str, float | int]:
    grounded = parsed.grounded_bullets
    grounded_count = len(grounded)
    if grounded_count < 2:
        overlap = 1.0
        diversity = 0.0
    else:
        # Only evidence verified in the prompt affects the score. Fake SIDs
        # cannot be added to manufacture low overlap.
        evidence_sets = [set(item.grounded_evidence) for item in grounded]
        overlap = mean_pairwise_jaccard(evidence_sets)
        diversity = 1.0 - overlap
    gate = interest_count_gate(grounded_count)
    score = gate * (0.5 + 0.5 * diversity)
    return {
        "bullet_count": parsed.bullet_count,
        "grounded_count": grounded_count,
        "count_gate": gate,
        "evidence_overlap": overlap,
        "evidence_diversity": diversity,
        "s_cot": score,
    }


def prefix_support(
    beam_sids: Sequence[Sid | None], gold_sids: Iterable[Sid], beam_size: int = 32
) -> dict[str, float]:
    golds = set(gold_sids)
    gold_a = {(domain, a) for domain, a, _b, _c in golds}
    gold_ab = {(domain, a, b) for domain, a, b, _c in golds}
    valid = [sid for sid in beam_sids if sid is not None]
    s_a = (
        sum(math.sqrt(sum(sid[:2] == prefix for sid in valid) / beam_size) for prefix in gold_a)
        / len(gold_a)
        if gold_a else 0.0
    )
    s_ab = (
        sum(math.sqrt(sum(sid[:3] == prefix for sid in valid) / beam_size) for prefix in gold_ab)
        / len(gold_ab)
        if gold_ab else 0.0
    )
    return {"s_a": s_a, "s_ab": s_ab, "s_prefix": 0.33 * s_a + 0.67 * s_ab}


def exploration_score(beam_sids: Sequence[Sid | None], target_domain: str) -> dict[str, float | int]:
    values = [sid[1] for sid in beam_sids if sid is not None and sid[0] == target_domain]
    unique_a = len(set(values))
    if len(values) <= 1:
        entropy_norm = 0.0
    else:
        counts = Counter(values)
        entropy = -sum((count / len(values)) * math.log(count / len(values)) for count in counts.values())
        entropy_norm = entropy / math.log(32)
    score = min(unique_a / 8.0, 1.0) * entropy_norm
    return {
        "valid_same_domain_beams": len(values),
        "unique_a": unique_a,
        "a_entropy_norm": entropy_norm,
        "s_explore": score,
    }


def choose_think_aux_scores(candidates: Sequence[dict]) -> tuple[list[float], str]:
    """Select the group branch and return one auxiliary score per candidate."""
    primary = torch.tensor([float(item["primary_reward"]) for item in candidates])
    primary_std = float(primary.std(correction=0))
    all_zero = bool(torch.equal(primary, torch.zeros_like(primary)))
    any_exact = any(bool(item.get("has_exact")) for item in candidates)
    any_prefix = any(float(item.get("s_prefix", 0.0)) > 0.0 for item in candidates)
    if primary_std > 0.0:
        branch = "primary_variance"
        scores = [float(item["s_cot"]) for item in candidates]
    elif all_zero:
        branch = "dead_zero"
        scores = [
            0.60 * float(item["s_cot"])
            + 0.40 * float(item["s_cot"]) * float(item["s_explore"])
            for item in candidates
        ]
    elif not any_exact and any_prefix:
        branch = "prefix_rescue"
        scores = [
            0.40 * float(item["s_cot"]) + 0.60 * float(item["s_prefix"])
            for item in candidates
        ]
    else:
        branch = "cot_only_saturated_or_other"
        scores = [float(item["s_cot"]) for item in candidates]
    return scores, branch


def group_aux_advantages(scores: Sequence[float], group_size: int, eps: float = EPS) -> torch.Tensor:
    values = torch.as_tensor(scores, dtype=torch.float32)
    if values.numel() % group_size:
        raise ValueError("auxiliary scores are not divisible by group_size")
    grouped = values.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, correction=0, keepdim=True)
    advantages = (grouped - means) / (stds + eps)
    advantages = torch.where(stds == 0, torch.zeros_like(advantages), advantages)
    return advantages.reshape(-1)


@dataclass(frozen=True)
class NoThinkRescuePlan:
    active: bool
    concentration: float
    gold_unique_a: int
    lambda_a: float
    predicted_as: tuple[int | None, ...]
    frequency_weights: tuple[float, ...]
    positions: tuple[int, ...]

    @property
    def coefficient(self) -> float:
        return self.lambda_a * self.concentration if self.active else 0.0


def build_nothink_rescue_plan(
    rewards: Sequence[float],
    predicted_as: Sequence[int | None],
    positions: Sequence[int],
    gold_as: Iterable[int],
    rescue_scale: float = 1.0,
) -> NoThinkRescuePlan:
    if len(rewards) != 8 or len(predicted_as) != 8 or len(positions) != 8:
        raise ValueError("NoThink rescue requires one complete G=8 group")
    valid_a = [value for value in predicted_as if value is not None]
    counts = Counter(valid_a)
    concentration = max(counts.values(), default=0) / 8.0
    weights = tuple(counts.get(value, 0) / 8.0 if value is not None else 0.0 for value in predicted_as)
    gold_unique_a = len(set(gold_as))
    lambda_a = (0.10 if gold_unique_a < 3 else 0.20) * float(rescue_scale)
    active = (
        rescue_scale > 0
        and all(float(value) == 0.0 for value in rewards)
        and all(value is not None and position >= 0 for value, position in zip(predicted_as, positions))
    )
    return NoThinkRescuePlan(
        active=active,
        concentration=concentration,
        gold_unique_a=gold_unique_a,
        lambda_a=lambda_a,
        predicted_as=tuple(predicted_as),
        frequency_weights=weights,
        positions=tuple(positions),
    )


def clipped_surrogate_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    delta: float | None = None,
) -> torch.Tensor:
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
    if delta is not None:
        ratio = torch.clamp(ratio, max=delta)
    loss1 = ratio * advantages.unsqueeze(1)
    loss2 = clipped * advantages.unsqueeze(1)
    per_token = -torch.min(loss1, loss2)
    return (per_token * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)


def nothink_unlikelihood_loss(
    per_token_logps: torch.Tensor,
    positions: torch.Tensor,
    frequency_weights: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Token-local wrong-A unlikelihood, averaged over all local candidates."""
    valid = positions >= 0
    if not bool(valid.any()):
        return per_token_logps.sum() * 0.0
    rows = torch.arange(per_token_logps.size(0), device=per_token_logps.device)[valid]
    selected = per_token_logps[rows, positions[valid]]
    probabilities = torch.exp(selected).clamp(max=1.0 - 1e-6)
    unlikelihood = -torch.log1p(-probabilities)
    weighted = coefficients[valid] * frequency_weights[valid] * unlikelihood
    # Inactive/invalid candidates are exact zeros in the requested group mean.
    return weighted.sum() / per_token_logps.size(0)
