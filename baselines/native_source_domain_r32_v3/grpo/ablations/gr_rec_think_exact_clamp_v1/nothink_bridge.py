"""Dead-zero Gold-A teacher bridge primitives for NoThink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


BRIDGE_OFF = "off"
BRIDGE_DEAD_A = "dead_zero_a_bridge"
BRIDGE_LAMBDA = 0.02


@dataclass(frozen=True)
class BridgePlan:
    branch: str = BRIDGE_OFF
    gold_a_targets: tuple[int, ...] = ()

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


def plan_dead_zero_bridge(
    rewards: Sequence[float], gold_sids: Iterable[Sequence], target_domain: str,
) -> BridgePlan:
    """Activate only for exact G8 all-zero reward and teach unique Gold A."""
    reward_values = tuple(float(value) for value in rewards)
    if len(reward_values) != 8:
        raise ValueError("dead-zero bridge planner requires exactly one G8")
    if reward_values != (0.0,) * 8:
        return BridgePlan()
    gold = {sid for value in gold_sids if (sid := _sid_tuple(value)) is not None}
    targets = tuple(sorted({a for domain, a, _b, _c in gold if domain == target_domain}))
    return BridgePlan(BRIDGE_DEAD_A, targets) if targets else BridgePlan()


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
    if min(world_size, global_group_count, ranks_for_group) <= 0:
        raise ValueError("DDP group-weight dimensions must be positive")
    return world_size / (global_group_count * ranks_for_group)


def bridge_token_text(kind: str, value: int | str) -> str:
    if kind == "domain":
        return f"<|{value}_begin|>"
    if kind in {"a", "b", "c"}:
        return f"<s_{kind}_{int(value)}>"
    raise ValueError(f"unknown bridge token kind: {kind}")


def require_single_token(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"bridge token must encode to one token: {text!r} -> {token_ids}")
    return int(token_ids[0])
