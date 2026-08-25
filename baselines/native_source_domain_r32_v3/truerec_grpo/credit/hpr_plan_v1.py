"""Pure CPU multi-positive Hierarchical Positive Rescue planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence


ABC_RE = re.compile(r"^(<s_a_\d+>)(<s_b_\d+>)(<s_c_\d+>)$")
TRIGGERS = ("HPR_A", "HPR_B", "HPR_C", "HPR_NONE")


@dataclass(frozen=True)
class HPRSite:
    level: str
    prefix_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    onpolicy_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HPRGroupPlan:
    trigger: str
    sites: tuple[HPRSite, ...]
    site_reduction: str = "mean_within_group"
    extra_model_forward: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_abc(value: str) -> tuple[str, str, str]:
    match = ABC_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid Gold ABC: {value!r}")
    return match.groups()


def candidate_abc(candidate: dict[str, Any]) -> tuple[str, str, str] | None:
    value = candidate.get("parsed_abc")
    return split_abc(value) if value else None


def plan_hpr(candidates: Sequence[dict[str, Any]], all_gold_abc: Sequence[str]) -> HPRGroupPlan:
    if len(candidates) != 8:
        raise ValueError("HPR requires exactly one G8 group")
    gold = sorted(set(split_abc(value) for value in all_gold_abc))
    if not gold:
        raise ValueError("HPR requires nonempty Gold")
    any_a = any(candidate["A_hit"] for candidate in candidates)
    any_ab = any(candidate["AB_hit"] for candidate in candidates)
    any_exact = any(candidate["exact"] for candidate in candidates)
    if any_exact:
        return HPRGroupPlan("HPR_NONE", ())
    if not any_a:
        positions = tuple((index, 0) for index in range(8))
        site = HPRSite("A", (), tuple(sorted({abc[0] for abc in gold})), positions)
        return HPRGroupPlan("HPR_A", (site,))
    if not any_ab:
        reached_a = sorted({candidate_abc(item)[0] for item in candidates if item["A_hit"]})
        sites = []
        for a_token in reached_a:
            positions = tuple((index, 1) for index, item in enumerate(candidates) if item["A_hit"] and candidate_abc(item)[0] == a_token)
            targets = tuple(sorted({abc[1] for abc in gold if abc[0] == a_token}))
            sites.append(HPRSite("B", (a_token,), targets, positions))
        return HPRGroupPlan("HPR_B", tuple(sites))
    reached_ab = sorted({candidate_abc(item)[:2] for item in candidates if item["AB_hit"]})
    sites = []
    for prefix in reached_ab:
        positions = tuple((index, 2) for index, item in enumerate(candidates) if item["AB_hit"] and candidate_abc(item)[:2] == prefix)
        targets = tuple(sorted({abc[2] for abc in gold if abc[:2] == prefix}))
        sites.append(HPRSite("C", prefix, targets, positions))
    return HPRGroupPlan("HPR_C", tuple(sites))


def validate_plan(plan: HPRGroupPlan) -> None:
    if plan.trigger == "HPR_NONE" and plan.sites:
        raise ValueError("HPR_NONE cannot contain sites")
    if plan.trigger != "HPR_NONE" and not plan.sites:
        raise ValueError("triggered HPR requires a site")
    for site in plan.sites:
        if not site.target_tokens or not site.onpolicy_positions:
            raise ValueError("HPR site lacks target or on-policy position")
    keys = [(site.level, site.prefix_tokens) for site in plan.sites]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate HPR prefix site")
