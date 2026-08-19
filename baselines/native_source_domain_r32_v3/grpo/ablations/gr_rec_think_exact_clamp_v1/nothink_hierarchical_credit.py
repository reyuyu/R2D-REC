"""Conditional hierarchy-state credit for NoThink final SID tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


HIERARCHY_SCALE = 8.0
STAGE_INCREMENTS = (0.5, 1.5, 6.0)


@dataclass(frozen=True)
class HierarchyState:
    valid: bool
    domain_correct: bool
    a_correct: bool
    ab_correct: bool
    exact: bool


def hierarchy_state(final_sid, gold_sids: Iterable[Sequence], target_domain: str) -> HierarchyState:
    """Derive monotonic final-SID correctness flags without using scalar reward."""
    gold = {tuple(value) for value in gold_sids if value is not None and len(value) == 4}
    valid = final_sid is not None and len(final_sid) == 4
    if not valid:
        return HierarchyState(False, False, False, False, False)
    sid = tuple(final_sid)
    domain_correct = sid[0] == target_domain
    a_correct = domain_correct and any(sid[:2] == item[:2] for item in gold)
    ab_correct = a_correct and any(sid[:3] == item[:3] for item in gold)
    exact = ab_correct and sid in gold
    return HierarchyState(valid, domain_correct, a_correct, ab_correct, exact)


def conditional_hierarchical_credits(
    states: Sequence[HierarchyState],
) -> list[tuple[float, float, float]]:
    """Return per-candidate (A, B, C) credit for exactly one NoThink G8."""
    if len(states) != 8:
        raise ValueError("conditional hierarchical credit requires exactly one G8")
    credits = [[0.0, 0.0, 0.0] for _ in states]
    stages = (
        (lambda state: state.domain_correct, lambda state: state.a_correct),
        (lambda state: state.a_correct, lambda state: state.ab_correct),
        (lambda state: state.ab_correct, lambda state: state.exact),
    )
    for column, ((eligible, correct), increment) in enumerate(zip(stages, STAGE_INCREMENTS)):
        indices = [index for index, state in enumerate(states) if eligible(state)]
        indicators = [float(correct(states[index])) for index in indices]
        if not indicators or len(set(indicators)) == 1:
            continue
        mean = sum(indicators) / len(indicators)
        for index, indicator in zip(indices, indicators):
            credits[index][column] = increment * (indicator - mean) / HIERARCHY_SCALE
    return [tuple(row) for row in credits]


def find_final_sid_token_positions(
    completion_ids: Sequence[int], final_sid, tokenizer,
) -> tuple[int, int, int]:
    """Find A/B/C positions in the last exact contiguous final-SID token block."""
    if final_sid is None or len(final_sid) != 4:
        raise ValueError("a valid final SID is required")
    domain, a, b, c = final_sid
    token_texts = (
        f"<|{domain}_begin|>",
        f"<s_a_{int(a)}>",
        f"<s_b_{int(b)}>",
        f"<s_c_{int(c)}>",
    )
    block = []
    for text in token_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"final SID token must encode singly: {text!r} -> {encoded}")
        block.append(int(encoded[0]))
    ids = [int(value) for value in completion_ids]
    for start in range(len(ids) - 4, -1, -1):
        if ids[start:start + 4] == block:
            return start + 1, start + 2, start + 3
    raise RuntimeError("parsed final SID has no matching contiguous token block")
