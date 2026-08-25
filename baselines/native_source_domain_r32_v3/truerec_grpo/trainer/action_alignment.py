"""Exact three-token TrueRec action and causal-logit alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ACTION_LEVELS = ("A", "B", "C")
DOMAIN_IS_RL_ACTION = False
DOMAIN_DIRECT_LOSS_TERMS = 0
CONTEXT_NONACTION_DIRECT_LOSS_TERMS = 0


@dataclass(frozen=True)
class ActionAlignment:
    context_length: int
    action_indices: tuple[int, int, int]
    logit_indices: tuple[int, int, int]


def align_action(context_length: int, completion_ids: Sequence[int]) -> ActionAlignment:
    if context_length < 1:
        raise ValueError("context_length must include at least one token")
    if len(completion_ids) != 3:
        raise ValueError("TrueRec action must contain exactly A/B/C tokens")
    return ActionAlignment(
        context_length=context_length,
        action_indices=(context_length, context_length + 1, context_length + 2),
        logit_indices=(context_length - 1, context_length, context_length + 1),
    )
