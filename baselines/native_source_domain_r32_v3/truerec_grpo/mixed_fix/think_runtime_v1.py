"""Runtime token-ID adapter for the frozen Mixed-Fix Think credit plan."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (Path(__file__).resolve().parent, ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import think_credit_v1  # noqa: E402
from batch_collator_v1 import PaddedBusinessGroup  # noqa: E402


@dataclass(frozen=True)
class RuntimeARescue:
    triggered: bool
    reason: str
    target_token_ids: tuple[int, ...]
    reduction: str


@dataclass(frozen=True)
class RuntimeHPRSite:
    level: str
    target_token_ids: tuple[int, ...]
    onpolicy_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RuntimeBCHPR:
    trigger: str
    sites: tuple[RuntimeHPRSite, ...]


@dataclass(frozen=True)
class ThinkRuntimePlan:
    token_credits: torch.Tensor
    token_credit_mask: torch.Tensor
    a_rescue: RuntimeARescue
    bc_hpr: RuntimeBCHPR
    monitoring: dict[str, float | int | bool]


def _map_tokens(tokens: Sequence[str], token_to_id: Callable[[str], int]) -> tuple[int, ...]:
    source = tuple(sorted(set(str(token) for token in tokens)))
    if not source:
        raise ValueError("runtime target token set cannot be empty")
    mapped = tuple(sorted(set(int(token_to_id(token)) for token in source)))
    if len(mapped) != len(source):
        raise ValueError("runtime token mapping is not one-to-one")
    if any(token_id < 0 for token_id in mapped):
        raise ValueError("runtime target token ID must be nonnegative")
    return mapped


def build_think_runtime_plan(
    candidates: Sequence[dict[str, Any]],
    all_gold_abc: Sequence[str],
    token_to_id: Callable[[str], int],
    *,
    dtype: torch.dtype = torch.float32,
) -> ThinkRuntimePlan:
    """Adapt, without re-planning, the frozen Phase-1 Think plan for loss code."""
    source = think_credit_v1.plan_think_credit(candidates, all_gold_abc)
    if source.bc_hpr.trigger == "HPR_A":
        raise ValueError("Think runtime must never produce HPR_A")
    a_targets = _map_tokens(source.a_rescue.target_a_tokens, token_to_id)
    sites = tuple(
        RuntimeHPRSite(
            site.level,
            _map_tokens(site.target_tokens, token_to_id),
            tuple((int(sample), int(position)) for sample, position in site.onpolicy_positions),
        )
        for site in source.bc_hpr.sites
    )
    if source.bc_hpr.trigger not in {"HPR_B", "HPR_C", "HPR_NONE", "NONE"}:
        raise ValueError(f"unsupported Think HPR trigger: {source.bc_hpr.trigger}")
    monitoring = dict(source.monitoring)
    monitoring.update({
        "A_RESCUE_TRIGGERED": int(source.a_rescue.triggered),
        "A_RESCUE_NO_GOLD_A": int(source.a_rescue.reason == "NO_GOLD_A"),
        "A_RESCUE_MODE_COLLAPSE": int(source.a_rescue.reason == "A_MODE_COLLAPSE"),
        "HPR_B": int(source.bc_hpr.trigger == "HPR_B"),
        "HPR_C": int(source.bc_hpr.trigger == "HPR_C"),
        "HPR_NONE": int(source.bc_hpr.trigger in {"HPR_NONE", "NONE"}),
        "HPR_DISABLED_NO_A": int(source.bc_hpr.trigger == "NONE"),
    })
    return ThinkRuntimePlan(
        torch.tensor(source.token_credits, dtype=dtype),
        torch.tensor(source.semantic_active_mask, dtype=torch.bool),
        RuntimeARescue(
            source.a_rescue.triggered, source.a_rescue.reason, a_targets,
            source.a_rescue.reduction,
        ),
        RuntimeBCHPR(source.bc_hpr.trigger, sites),
        monitoring,
    )


def assert_shared_a_prefix(batch: PaddedBusinessGroup) -> tuple[int, ...]:
    """Fail closed unless every A causal logit sees the identical context prefix."""
    prefixes = []
    for row in range(batch.input_ids.shape[0]):
        causal_index = int(batch.causal_logit_indices[row, 0])
        visible = batch.input_ids[row, : causal_index + 1]
        visible_mask = batch.attention_mask[row, : causal_index + 1]
        prefix = tuple(int(value) for value in visible[visible_mask].tolist())
        if len(prefix) != int(batch.context_lengths[row]):
            raise ValueError("A causal position does not terminate at context boundary")
        prefixes.append(prefix)
    if not prefixes or any(prefix != prefixes[0] for prefix in prefixes[1:]):
        raise ValueError("G8 A causal positions do not share an identical context prefix")
    return prefixes[0]
