"""Positive-only hierarchical credit and rescue planning for Think G8."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Sequence


CREDIT_DIR = Path(__file__).resolve().parents[1] / "credit"
if str(CREDIT_DIR) not in sys.path:
    sys.path.insert(0, str(CREDIT_DIR))

from hpr_plan_v1 import HPRSite, candidate_abc, plan_hpr, split_abc, validate_plan  # noqa: E402
from think_rescue_v1 import A_RESCUE_REDUCTION  # noqa: E402


G = 8
HIERARCHY_SCALE = 8.0
DELTA_A = 0.5
DELTA_B = 1.5
DELTA_C = 6.0
POSITIVE = "positive"
ZERO = "zero_semantic_credit"
GATED = "gated"


@dataclass(frozen=True)
class ThinkCandidateCredit:
    credits: tuple[float, float, float]
    kinds: tuple[str, str, str]
    semantic_active_mask: tuple[bool, bool, bool]


@dataclass(frozen=True)
class ARescuePlan:
    triggered: bool
    reason: str
    target_a_tokens: tuple[str, ...]
    reduction: str = A_RESCUE_REDUCTION


@dataclass(frozen=True)
class BCHPRPlan:
    trigger: str
    sites: tuple[HPRSite, ...]
    source: str = "frozen hpr_plan_v1.plan_hpr conditional B/C targets"


@dataclass(frozen=True)
class ThinkGroupPlan:
    candidates: tuple[ThinkCandidateCredit, ...]
    token_credits: tuple[tuple[float, float, float], ...]
    semantic_active_mask: tuple[tuple[bool, bool, bool], ...]
    a_statistics: dict[str, Any]
    a_rescue: ARescuePlan
    bc_hpr: BCHPRPlan
    monitoring: dict[str, float | int | bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_and_parse(
    candidates: Sequence[dict[str, Any]], all_gold_abc: Sequence[str]
) -> tuple[list[tuple[str, str, str] | None], tuple[tuple[str, str, str], ...]]:
    if len(candidates) != G:
        raise ValueError("Think credit requires exactly one G8 group")
    gold = tuple(sorted(set(split_abc(value) for value in all_gold_abc)))
    if not gold:
        raise ValueError("Think credit requires nonempty Gold")
    gold_a = {abc[0] for abc in gold}
    gold_ab = {abc[:2] for abc in gold}
    gold_abc = set(gold)
    parsed = []
    for candidate in candidates:
        milestones = (
            bool(candidate["A_hit"]), bool(candidate["AB_hit"]), bool(candidate["exact"])
        )
        if milestones[2] and not milestones[1] or milestones[1] and not milestones[0]:
            raise ValueError("candidate hierarchy invariant failed")
        abc = candidate_abc(candidate) if candidate.get("parsed_abc") else None
        parsed.append(abc)
        if candidate["format_valid"]:
            if abc is None:
                raise ValueError("format-valid candidate lacks parsed ABC")
            expected = (abc[0] in gold_a, abc[:2] in gold_ab, abc in gold_abc)
            if milestones != expected:
                raise ValueError("candidate metrics disagree with parsed ABC and Gold")
        elif any(milestones):
            raise ValueError("format-invalid candidate cannot reach hierarchy milestone")
    return parsed, gold


def _bc_hpr(
    candidates: Sequence[dict[str, Any]], all_gold_abc: Sequence[str]
) -> BCHPRPlan:
    if not any(bool(item["A_hit"]) for item in candidates):
        return BCHPRPlan("NONE", ())
    source = plan_hpr(candidates, all_gold_abc)
    validate_plan(source)
    if source.trigger == "HPR_A":
        raise ValueError("Think route must never produce HPR_A")
    return BCHPRPlan(source.trigger, source.sites)


def plan_think_credit(
    candidates: Sequence[dict[str, Any]], all_gold_abc: Sequence[str]
) -> ThinkGroupPlan:
    parsed, gold = _validate_and_parse(candidates, all_gold_abc)
    gold_a = tuple(sorted({abc[0] for abc in gold}))
    parsed_a_counts = Counter(
        abc[0]
        for candidate, abc in zip(candidates, parsed)
        if candidate["format_valid"] and abc is not None
    )
    correct_a_counts = Counter(
        abc[0]
        for candidate, abc in zip(candidates, parsed)
        if candidate["format_valid"] and candidate["A_hit"] and abc is not None
    )
    mean_ab = sum(bool(item["AB_hit"]) for item in candidates) / G
    mean_exact = sum(bool(item["exact"]) for item in candidates) / G

    planned = []
    for candidate, abc in zip(candidates, parsed):
        if not candidate["format_valid"]:
            planned.append(ThinkCandidateCredit((0.0, 0.0, 0.0), (GATED,) * 3, (False,) * 3))
            continue
        assert abc is not None
        credits = [0.0, 0.0, 0.0]
        kinds = [GATED, GATED, GATED]
        active = [False, False, False]
        if not candidate["A_hit"]:
            kinds[0], active[0] = ZERO, True
        else:
            credits[0] = DELTA_A * (1.0 - parsed_a_counts[abc[0]] / G) / HIERARCHY_SCALE
            kinds[0], active[0] = POSITIVE, True
            if not candidate["AB_hit"]:
                kinds[1], active[1] = ZERO, True
            else:
                credits[1] = DELTA_B * (1.0 - mean_ab) / HIERARCHY_SCALE
                kinds[1], active[1] = POSITIVE, True
                if not candidate["exact"]:
                    kinds[2], active[2] = ZERO, True
                else:
                    credits[2] = DELTA_C * (1.0 - mean_exact) / HIERARCHY_SCALE
                    kinds[2], active[2] = POSITIVE, True
        planned.append(ThinkCandidateCredit(tuple(credits), tuple(kinds), tuple(active)))

    covered = tuple(sorted(correct_a_counts))
    k_a, d_a = len(gold_a), len(covered)
    t_a = min(k_a, 3)
    max_count = max(correct_a_counts.values(), default=0)
    if d_a == 0:
        rescue = ARescuePlan(True, "NO_GOLD_A", gold_a)
    elif k_a >= 2 and d_a < t_a and max_count >= 5:
        rescue = ARescuePlan(True, "A_MODE_COLLAPSE", gold_a)
    else:
        rescue = ARescuePlan(False, "NONE", gold_a)

    valid_parsed = [abc for candidate, abc in zip(candidates, parsed) if candidate["format_valid"] and abc is not None]
    monitoring = {
        "A_hit_rate": sum(bool(item["A_hit"]) for item in candidates) / G,
        "AB_hit_rate": mean_ab,
        "exact_rate": mean_exact,
        "distinct_correct_A": d_a,
        "A_coverage_target": t_a,
        "A_coverage_satisfied": d_a >= t_a,
        "A_mode_collapse": rescue.reason == "A_MODE_COLLAPSE",
        "max_correct_A_frequency": max_count / G,
        "unique_A_per_G8": len({abc[0] for abc in valid_parsed}),
        "unique_AB_per_G8": len({abc[:2] for abc in valid_parsed}),
        "unique_ABC_per_G8": len(set(valid_parsed)),
    }
    stats = {
        "K_A": k_a,
        "T_A": t_a,
        "d_A": d_a,
        "covered_gold_A": covered,
        "correct_gold_A_counts": dict(sorted(correct_a_counts.items())),
        "max_correct_gold_A_count": max_count,
    }
    candidate_tuple = tuple(planned)
    return ThinkGroupPlan(
        candidate_tuple,
        tuple(item.credits for item in candidate_tuple),
        tuple(item.semantic_active_mask for item in candidate_tuple),
        stats,
        rescue,
        _bc_hpr(candidates, all_gold_abc),
        monitoring,
    )
