"""Evaluator-aligned marginal credit for GR_USER Action and Chain outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from user_action_reward import score_action
from user_chain_reward import score_chain


def _credit_type(delta: float) -> str:
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "negative"
    return "zero"


def _violation_kinds(result: Any) -> List[str]:
    return [violation.kind for violation in result.violations]


def action_marginal_credit(
    completion: str,
    sample: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute leave-one-unique-SID-out Action reward deltas.

    A repeated SID is removed in full for its first occurrence. Later duplicate
    occurrences receive zero credit because removing only those occurrences does
    not change the evaluator's unique prediction set.
    """

    full = score_action(completion, sample)
    if not full.format_valid:
        return {
            "valid": False,
            "full_reward": float(full.reward),
            "credits": [],
            "diagnostics": {
                "parser_errors": list(full.parser_errors),
                "violation_kinds": _violation_kinds(full),
            },
        }

    raw_sids = list(full.pred_sids_raw)
    gold_sids = set(full.gold_sids)
    reward_without_sid: Dict[str, float] = {}
    for sid in full.predicted_sids:
        reduced_completion = json.dumps(
            [candidate for candidate in raw_sids if candidate != sid],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        reward_without_sid[sid] = float(
            score_action(reduced_completion, sample).reward
        )

    credits: List[Dict[str, Any]] = []
    for occurrence in full.sid_occurrences:
        sid = occurrence["sid"]
        occurrence_index = int(occurrence["occurrence"])
        if occurrence_index == 1:
            without_reward = reward_without_sid[sid]
            delta = float(full.reward) - without_reward
        else:
            without_reward = float(full.reward)
            delta = 0.0

        char_start = int(occurrence["char_start"])
        char_end = int(occurrence["char_end"])
        credits.append(
            {
                "sid": sid,
                "occurrence": occurrence_index,
                "is_gold": sid in gold_sids,
                "char_start": char_start,
                "char_end": char_end,
                "char_span": [char_start, char_end],
                "full_reward": float(full.reward),
                "reward_without_sid": without_reward,
                "delta": delta,
                "credit_type": _credit_type(delta),
            }
        )

    return {
        "valid": True,
        "full_reward": float(full.reward),
        "credits": credits,
        "diagnostics": {
            "parser_errors": list(full.parser_errors),
            "violation_kinds": _violation_kinds(full),
        },
    }


def _serialize_chain(events: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {"logic_chain": {"name": "marginal-credit", "events": list(events)}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def chain_marginal_credit(
    completion: str,
    sample: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute leave-one-event-out Chain reward deltas via the real scorer."""

    full = score_chain(completion, sample)
    if not full.format_valid:
        return {
            "valid": False,
            "full_reward": float(full.total_reward),
            "credits": [],
            "diagnostics": {
                "parser_errors": list(full.parser_errors),
                "violation_kinds": _violation_kinds(full),
            },
        }

    events = list(full.predicted_events)
    credits: List[Dict[str, Any]] = []
    for event_index in range(len(events)):
        reduced_events = events[:event_index] + events[event_index + 1 :]
        without = score_chain(_serialize_chain(reduced_events), sample)

        delta_action = float(full.action_f1) - float(without.action_f1)
        delta_logic = float(full.logic_f1) - float(without.logic_f1)
        delta_total = 0.5 * delta_action + 0.5 * delta_logic
        reward_delta = float(full.total_reward) - float(without.total_reward)

        span = full.event_spans[event_index]["event"]
        char_start = int(span["char_start"])
        char_end = int(span["char_end"])
        credits.append(
            {
                "event_index": event_index,
                "char_start": char_start,
                "char_end": char_end,
                "char_span": [char_start, char_end],
                "full_reward": float(full.total_reward),
                "reward_without_event": float(without.total_reward),
                "delta": delta_total,
                "delta_total": delta_total,
                "reward_delta": reward_delta,
                "credit_type": _credit_type(delta_total),
                "full_action_alignment": float(full.action_f1),
                "without_action_alignment": float(without.action_f1),
                "delta_action_alignment": delta_action,
                "full_logic_alignment": float(full.logic_f1),
                "without_logic_alignment": float(without.logic_f1),
                "delta_logic_alignment": delta_logic,
            }
        )

    return {
        "valid": True,
        "full_reward": float(full.total_reward),
        "full_action_alignment": float(full.action_f1),
        "full_logic_alignment": float(full.logic_f1),
        "credits": credits,
        "diagnostics": {
            "parser_errors": list(full.parser_errors),
            "violation_kinds": _violation_kinds(full),
        },
    }


__all__ = ["action_marginal_credit", "chain_marginal_credit"]
