"""Pure CPU A/B/C first-error Frontier credit contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


HIERARCHY_SCALE = 8.0
DELTAS = (0.5, 1.5, 6.0)
LEVELS = ("A", "B", "C")
GATED = "gated"
POSITIVE = "positive"
NEGATIVE = "frontier_negative"
FORMAT_INVALID_TOTAL = -0.75 / HIERARCHY_SCALE


@dataclass(frozen=True)
class CandidateCredit:
    credits: tuple[float, float, float]
    kinds: tuple[str, str, str]
    format_invalid: bool
    format_credit_total: float


@dataclass(frozen=True)
class FrontierGroupPlan:
    candidates: tuple[CandidateCredit, ...]
    milestone_means: tuple[float, float, float]
    domain_credit: bool = False
    hierarchy_scale: float = HIERARCHY_SCALE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _milestones(candidate: dict[str, Any]) -> tuple[bool, bool, bool]:
    values = (bool(candidate["A_hit"]), bool(candidate["AB_hit"]), bool(candidate["exact"]))
    if values[2] and not values[1] or values[1] and not values[0]:
        raise ValueError("candidate hierarchy invariant failed")
    return values


def plan_frontier_credit(candidates: Sequence[dict[str, Any]]) -> FrontierGroupPlan:
    if len(candidates) != 8:
        raise ValueError("Frontier credit requires exactly one G8 group")
    rows = [_milestones(candidate) for candidate in candidates]
    means = tuple(sum(row[level] for row in rows) / 8.0 for level in range(3))
    planned = []
    for candidate, row in zip(candidates, rows):
        if not candidate["format_valid"]:
            planned.append(CandidateCredit((0.0, 0.0, 0.0), (GATED, GATED, GATED), True, FORMAT_INVALID_TOTAL))
            continue
        credits = [0.0, 0.0, 0.0]
        kinds = [GATED, GATED, GATED]
        for level, (reached, delta) in enumerate(zip(row, DELTAS)):
            if reached:
                credits[level] = delta * (1.0 - means[level]) / HIERARCHY_SCALE
                kinds[level] = POSITIVE
            else:
                credits[level] = -delta / HIERARCHY_SCALE
                kinds[level] = NEGATIVE
                break
        if sum(kind == NEGATIVE for kind in kinds) > 1:
            raise ValueError("more than one first-error negative")
        planned.append(CandidateCredit(tuple(credits), tuple(kinds), False, 0.0))
    return FrontierGroupPlan(tuple(planned), means)


def group_credit_totals(plan: FrontierGroupPlan) -> tuple[float, float]:
    values = [credit for candidate in plan.candidates for credit in candidate.credits]
    return sum(values), sum(abs(value) for value in values)
