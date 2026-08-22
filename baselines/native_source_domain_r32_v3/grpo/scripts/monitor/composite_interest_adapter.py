"""Read-only adapter for captured Composite Interest monitor events."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable


EXPERIMENT = "GR_REC_Think_CompositeInterest_v1"
FORMULA = "composite_interest_v1"


def is_composite_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == EXPERIMENT


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    required = (
        "beam_raw", "beam_utility", "beam_contribution", "cot_contribution",
        "cot_utility", "composite_reward", "final_sequence_advantage",
    )
    missing = [key for key in required if _finite(row.get(key)) is None]
    if missing:
        raise ValueError("captured Composite candidate missing fields: " + ", ".join(missing))
    total = float(row["beam_contribution"]) + float(row["cot_contribution"])
    if not math.isclose(total, float(row["composite_reward"]), abs_tol=1e-9):
        raise ValueError("captured Composite contribution mismatch")
    row.setdefault("Q", row.get("match_quality"))
    row.setdefault("completion_length", None)
    row.setdefault("pred_interest_units", [])
    row.setdefault("match_details", [])
    row.setdefault("unmatched_pred_indices", [])
    row.setdefault("unmatched_gold_indices", [])
    row.setdefault("unmatched_pred_best_alternatives", [])
    row.setdefault("unmatched_gold_best_alternatives", [])
    return row


def adapt_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one global rollout event into frontend-sized G4 records."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in event.get("candidates", []):
        group_id = str(candidate.get("group_id") or "")
        if group_id:
            by_group[group_id].append(_candidate(candidate))
    output = []
    for captured_group in event.get("groups", []):
        group = dict(captured_group)
        group_id = str(group.get("group_id") or "")
        candidates = sorted(by_group.get(group_id, []), key=lambda row: int(row.get("candidate_id", 0)))
        if len(candidates) != 4:
            raise ValueError(f"captured Composite group {group_id!r} is not aligned G4")
        vectors = {
            "beam_raw_vector": "beam_raw",
            "beam_utility_vector": "beam_utility",
            "cot_utility_vector": "cot_utility",
            "composite_reward_vector": "composite_reward",
            "final_advantage_vector": "final_sequence_advantage",
        }
        for vector, candidate_key in vectors.items():
            values = group.get(vector)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"captured Composite group missing {vector}")
            if any(not math.isclose(float(value), float(candidate[candidate_key]), abs_tol=1e-9)
                   for value, candidate in zip(values, candidates)):
                raise ValueError(f"captured Composite candidate alignment drift: {vector}")
        output.append({
            **group,
            "step": int(event.get("step", 0)),
            "rollout_id": event.get("rollout_id"),
            "route": "think",
            "formula": FORMULA,
            "provenance": {"mode": "captured", "label": "训练时实采"},
            "candidates": candidates,
            "valid": True,
        })
    return output


def adapt_events(
    events: Iterable[dict[str, Any]],
    *,
    from_step: int | None = None,
    to_step: int | None = None,
    rollout_id: int | None = None,
    group_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    groups = []
    for event in events:
        step = int(event.get("step", 0))
        if from_step is not None and step < from_step:
            continue
        if to_step is not None and step > to_step:
            continue
        if rollout_id is not None and event.get("rollout_id") != rollout_id:
            continue
        groups.extend(adapt_event(event))
    if group_id is not None:
        groups = [group for group in groups if group.get("group_id") == group_id]
    return groups[-limit:] if limit is not None else groups


def summary(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate captured scalars only; reward mathematics is never recomputed."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_step[int(event.get("step", 0))].extend(adapt_event(event))

    def mean(values: Iterable[Any]) -> float | None:
        finite = [value for item in values if (value := _finite(item)) is not None]
        return sum(finite) / len(finite) if finite else None

    def population_std(values: Iterable[Any]) -> float | None:
        finite = [value for item in values if (value := _finite(item)) is not None]
        if not finite:
            return None
        center = sum(finite) / len(finite)
        return math.sqrt(sum((value - center) ** 2 for value in finite) / len(finite))

    result = []
    for step, groups in sorted(by_step.items()):
        candidates = [candidate for group in groups for candidate in group["candidates"]]
        count = len(groups)
        rescued = sum(bool(group.get("beam_all_equal")) and not bool(group.get("composite_all_equal")) for group in groups)
        result.append({
            "step": step,
            "group_count": count,
            "candidate_count": len(candidates),
            "beam_raw_mean": mean(candidate.get("beam_raw") for candidate in candidates),
            "beam_utility_mean": mean(candidate.get("beam_utility") for candidate in candidates),
            "cot_utility_mean": mean(candidate.get("cot_utility") for candidate in candidates),
            "composite_reward_mean": mean(candidate.get("composite_reward") for candidate in candidates),
            "composite_reward_std": population_std(
                candidate.get("composite_reward") for candidate in candidates
            ),
            "matched_interest_mean": mean(candidate.get("matched_interest_count") for candidate in candidates),
            "raw_n_mean": mean(candidate.get("raw_n") for candidate in candidates),
            "grounded_n_mean": mean(candidate.get("grounded_n") for candidate in candidates),
            "grounding_coverage_mean": mean(candidate.get("grounding_coverage") for candidate in candidates),
            "beam_zero_std_rate": sum(bool(group.get("beam_all_equal")) for group in groups) / count,
            "composite_zero_std_rate": sum(bool(group.get("composite_all_equal")) for group in groups) / count,
            "rescued_rate": rescued / count,
            "cot_active_rate": sum(bool(group.get("cot_active")) for group in groups) / count,
            "q_active_rate": sum(_finite(candidate.get("Q")) not in (None, 0.0) for candidate in candidates) / len(candidates),
        })
    return result


def captured_payload(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "read_only": True,
        "supported": True,
        "formula": FORMULA,
        "provenance": {"mode": "captured", "label": "训练时实采"},
        "groups": groups,
    }
