"""Pure G8 first-error frontier credit planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


HIERARCHY_SCALE = 8.0
FORMAT_INCREMENT = 0.75
FORMAT_ADV_TOTAL = -FORMAT_INCREMENT / HIERARCHY_SCALE
STAGE_NAMES = ("domain", "a", "b", "c")
STAGE_INCREMENTS = (0.25, 0.5, 1.5, 6.0)
GATED = "gated"
POSITIVE_SUCCESS = "positive_success"
FRONTIER_NEGATIVE = "frontier_negative"


@dataclass(frozen=True)
class FrontierCandidate:
    credits: tuple[float, float, float, float]
    kinds: tuple[str, str, str, str]


@dataclass(frozen=True)
class FrontierPlan:
    candidates: tuple[FrontierCandidate, ...]
    milestone_means: tuple[float, float, float, float]
    positive_active: tuple[bool, bool, bool, bool]
    frontier_active: tuple[bool, bool, bool, bool]
    frontier_negative_counts: tuple[int, int, int, int]
    taxonomy: str


def _milestones(state) -> tuple[bool, bool, bool, bool]:
    return (
        bool(state.domain_correct),
        bool(state.a_correct),
        bool(state.ab_correct),
        bool(state.exact),
    )


def _taxonomy(rows: Sequence[tuple[bool, bool, bool, bool]], valid: Sequence[bool]) -> str:
    if not all(valid):
        return "FORMAT_VIOLATION_PRESENT"
    if all(not row[0] for row in rows):
        return "ALL_WRONG_DOMAIN_FRONTIER"
    if all(row[0] and not row[1] for row in rows):
        return "UNIFORM_A_FAILURE_FRONTIER"
    if all(row[1] and not row[2] for row in rows):
        return "UNIFORM_B_FAILURE_FRONTIER"
    if all(row[2] and not row[3] for row in rows):
        return "UNIFORM_C_FAILURE_FRONTIER"
    if all(row[3] for row in rows):
        return "UNIFORM_EXACT_ZERO_SIGNAL"
    return "MIXED_FRONTIER"


def plan_frontier_credits(states: Sequence, format_valid: Sequence[bool]) -> FrontierPlan:
    """Plan positive milestone credit and one absolute first-error penalty per candidate."""
    if len(states) != 8 or len(format_valid) != 8:
        raise ValueError("frontier credit requires exactly one G8")
    valid = [bool(value) for value in format_valid]
    rows = [
        _milestones(state) if is_valid else (False, False, False, False)
        for state, is_valid in zip(states, valid)
    ]
    means = tuple(sum(float(row[column]) for row in rows) / 8.0 for column in range(4))
    candidates = []
    for row, is_valid in zip(rows, valid):
        credits = [0.0] * 4
        kinds = [GATED] * 4
        if is_valid:
            for column, (success, increment) in enumerate(zip(row, STAGE_INCREMENTS)):
                if success:
                    credits[column] = increment * (1.0 - means[column]) / HIERARCHY_SCALE
                    kinds[column] = POSITIVE_SUCCESS
                    continue
                credits[column] = -increment / HIERARCHY_SCALE
                kinds[column] = FRONTIER_NEGATIVE
                break
        candidates.append(FrontierCandidate(tuple(credits), tuple(kinds)))
    positive_active = tuple(
        any(candidate.kinds[column] == POSITIVE_SUCCESS and candidate.credits[column] != 0.0
            for candidate in candidates)
        for column in range(4)
    )
    frontier_counts = tuple(
        sum(candidate.kinds[column] == FRONTIER_NEGATIVE for candidate in candidates)
        for column in range(4)
    )
    return FrontierPlan(
        tuple(candidates),
        means,
        positive_active,
        tuple(count > 0 for count in frontier_counts),
        frontier_counts,
        _taxonomy(rows, valid),
    )


def distribute_format_penalty(length: int) -> tuple[float, ...]:
    if length <= 0:
        raise ValueError("format penalty requires at least one generated token")
    return (FORMAT_ADV_TOTAL / length,) * length
