"""Pure planner and loss primitives for the NoThink teacher bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


BRIDGE_OFF = "off"
BRIDGE_DEAD_A = "dead_zero_a_bridge"
BRIDGE_COLLAPSE_AB = "a_collapse_ab_bridge"
BRIDGE_LAMBDA = 0.02


@dataclass(frozen=True)
class BridgePlan:
    branch: str = BRIDGE_OFF
    gold_a_targets: tuple[int, ...] = ()
    current_a: int | None = None
    missing_a_targets: tuple[int, ...] = ()
    current_b_targets: tuple[int, ...] = ()

    @property
    def active(self) -> bool:
        return self.branch != BRIDGE_OFF


def _sid_tuple(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    domain, a, b, c = value
    try:
        return str(domain), int(a), int(b), int(c)
    except (TypeError, ValueError):
        return None


def plan_nothink_bridge(
    rewards: Sequence[float],
    predicted_sids: Sequence[Sequence | None],
    gold_sids: Iterable[Sequence],
    target_domain: str,
) -> BridgePlan:
    """Choose one mutually exclusive bridge branch for one NoThink G8 group."""
    reward_values = tuple(float(value) for value in rewards)
    if len(reward_values) != 8 or len(predicted_sids) != 8:
        raise ValueError("NoThink bridge planner requires exactly one G8 group")

    gold = tuple(sorted({sid for value in gold_sids if (sid := _sid_tuple(value)) is not None}))
    gold_a = tuple(sorted({a for domain, a, _b, _c in gold if domain == target_domain}))
    if reward_values == (0.0,) * 8:
        return BridgePlan(branch=BRIDGE_DEAD_A, gold_a_targets=gold_a) if gold_a else BridgePlan()

    predicted = tuple(_sid_tuple(value) for value in predicted_sids)
    if any(sid is None for sid in predicted):
        return BridgePlan()
    if any(sid[0] != target_domain for sid in predicted):
        return BridgePlan()
    predicted_a = {sid[1] for sid in predicted}
    if len(gold_a) < 3 or len(predicted_a) != 1:
        return BridgePlan()
    current_a = next(iter(predicted_a))
    if current_a not in gold_a:
        return BridgePlan()
    if max(reward_values) > 0.5 or 0.5 not in reward_values:
        return BridgePlan()

    missing_a = tuple(a for a in gold_a if a != current_a)
    current_b = tuple(sorted({
        b for domain, a, b, _c in gold
        if domain == target_domain and a == current_a
    }))
    if not missing_a or not current_b:
        return BridgePlan()
    return BridgePlan(
        branch=BRIDGE_COLLAPSE_AB,
        gold_a_targets=gold_a,
        current_a=current_a,
        missing_a_targets=missing_a,
        current_b_targets=current_b,
    )


def uniform_multi_positive_ce(logits: torch.Tensor, target_ids: Sequence[int]) -> torch.Tensor:
    """Uniform mean negative log-probability over unique teacher targets."""
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector")
    unique_targets = tuple(sorted({int(target) for target in target_ids}))
    if not unique_targets:
        raise ValueError("at least one teacher target is required")
    index = torch.tensor(unique_targets, dtype=torch.long, device=logits.device)
    return -torch.log_softmax(logits, dim=-1).index_select(0, index).mean()


def ddp_group_weight(world_size: int, global_group_count: int, ranks_for_group: int) -> float:
    """Per-rank factor whose DDP mean equals a mean over global groups."""
    if min(world_size, global_group_count, ranks_for_group) <= 0:
        raise ValueError("DDP group-weight dimensions must be positive")
    return world_size / (global_group_count * ranks_for_group)


def bridge_token_text(kind: str, value: int | str) -> str:
    if kind == "domain":
        return f"<|{value}_begin|>"
    if kind in {"a", "b"}:
        return f"<s_{kind}_{int(value)}>"
    raise ValueError(f"unknown bridge token kind: {kind}")


def require_single_token(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"bridge token must encode to one token: {text!r} -> {token_ids}")
    return int(token_ids[0])
