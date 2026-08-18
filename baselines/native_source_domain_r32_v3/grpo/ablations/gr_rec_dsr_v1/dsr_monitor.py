"""CPU-only aggregation for optional DSR monitor streams."""
from __future__ import annotations

import math
import os
import statistics
from collections import Counter
from typing import Iterable, Sequence

from .dsr_objectives import (
    build_nothink_rescue_plan,
    choose_think_aux_scores,
    group_aux_advantages,
)


THINK_BRANCHES = (
    "primary_variance",
    "dead_zero",
    "prefix_rescue",
    "cot_only_saturated_or_other",
)
REWARD_LEVELS = (-1.0, -0.25, 0.0, 0.5, 2.0, 8.0)


def decorate_dsr_monitor(writer, runner: str):
    """Add DSR capability metadata without changing the baseline writer."""
    original = writer.write_manifest

    def write_manifest(manifest):
        return original({
            **manifest,
            "runner": runner,
            "experiment": "GR_REC_DSR_Ablation_v1",
            "parent": "BATA baseline",
            "baseline": "GR_REC_v1",
            "dsr": {
                "think_lambda": float(os.environ.get("DSR_THINK_LAMBDA", "0.10")),
                "nothink_scale": float(os.environ.get("DSR_NOTHINK_SCALE", "1.0")),
                "monitor_schema": 2,
                "extra_forward": False,
                "sampling_changed": False,
            },
        })

    writer.write_manifest = write_manifest
    return writer


def _groups(records: Sequence[dict], group_size: int) -> list[list[dict]]:
    if len(records) % group_size:
        raise ValueError(f"record count {len(records)} is not divisible by G={group_size}")
    return [list(records[start:start + group_size]) for start in range(0, len(records), group_size)]


def _rate(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _zero_std(values: Sequence[float]) -> bool:
    return len(values) < 2 or statistics.pstdev(float(value) for value in values) == 0.0


def _count_distribution(values: Sequence[int]) -> dict[str, int]:
    distribution = {
        str(value): sum(count == value for count in values) for value in range(5)
    }
    distribution["5+"] = sum(count >= 5 for count in values)
    return distribution


def _plan_dict(plan) -> dict:
    return {
        "active": plan.active,
        "concentration": plan.concentration,
        "gold_unique_a": plan.gold_unique_a,
        "lambda_a": plan.lambda_a,
        "predicted_as": list(plan.predicted_as),
        "frequency_weights": list(plan.frequency_weights),
        "positions": list(plan.positions),
        "coefficient": plan.coefficient,
    }


def summarize_think_records(records: Sequence[dict]) -> tuple[dict, list[float], list[float]]:
    """Aggregate already-gathered Think records without model or tokenizer work."""
    grouped = _groups(records, 4)
    scores = []
    branches = []
    primary_zero = []
    all_zero = []
    auxiliary_zero = []
    signal_rescued = []
    for group in grouped:
        group_scores, branch = choose_think_aux_scores(group)
        rewards = [float(item["primary_reward"]) for item in group]
        primary_is_zero = _zero_std(rewards)
        auxiliary_is_zero = _zero_std(group_scores)
        scores.extend(group_scores)
        branches.append(branch)
        primary_zero.append(primary_is_zero)
        all_zero.append(all(value == 0.0 for value in rewards))
        auxiliary_zero.append(auxiliary_is_zero)
        signal_rescued.append(primary_is_zero and not auxiliary_is_zero)

    advantages = group_aux_advantages(scores, 4).tolist()
    raw_counts = [int(item["bullet_count"]) for item in records]
    grounded_counts = [int(item["grounded_count"]) for item in records]
    raw_grounded_gaps = [
        raw_count - grounded_count
        for raw_count, grounded_count in zip(raw_counts, grounded_counts)
    ]
    grounding_coverages = [
        grounded_count / raw_count
        for raw_count, grounded_count in zip(raw_counts, grounded_counts)
        if raw_count > 0
    ]
    branch_counts = {name: branches.count(name) for name in THINK_BRANCHES}
    s_cot_values = [float(item["s_cot"]) for item in records]
    payload = {
        "group_count": len(grouped),
        "parser_success_rate": _rate(bool(item["parsed"]["parser_success"]) for item in records),
        "raw_interest_count_mean": statistics.fmean(raw_counts) if raw_counts else 0.0,
        "raw_interest_count_distribution": _count_distribution(raw_counts),
        "grounded_interest_count_mean": statistics.fmean(grounded_counts) if grounded_counts else 0.0,
        "grounded_interest_count_distribution": _count_distribution(grounded_counts),
        "raw_grounded_gap_mean": statistics.fmean(raw_grounded_gaps) if raw_grounded_gaps else 0.0,
        "grounding_coverage_mean": statistics.fmean(grounding_coverages) if grounding_coverages else None,
        "grounding_coverage_defined_rate": len(grounding_coverages) / len(records) if records else 0.0,
        "s_cot_mean": statistics.fmean(s_cot_values) if s_cot_values else 0.0,
        "s_cot_std": statistics.pstdev(s_cot_values) if len(s_cot_values) > 1 else 0.0,
        "d_cot_mean": statistics.fmean(float(item["evidence_diversity"]) for item in records),
        "s_prefix_mean": statistics.fmean(float(item["s_prefix"]) for item in records),
        "s_explore_mean": statistics.fmean(float(item["s_explore"]) for item in records),
        "s_dead_mean": statistics.fmean(float(item["s_dead"]) for item in records),
        "primary_zero_std_rate": _rate(primary_zero),
        "all_zero_rate": _rate(all_zero),
        "aux_zero_std_rate": _rate(auxiliary_zero),
        "think_aux_zero_std_rate": _rate(auxiliary_zero),
        "signal_rescue_rate": _rate(signal_rescued),
        "unique_a_mean": statistics.fmean(int(item["unique_a"]) for item in records),
        "a_entropy_mean": statistics.fmean(float(item["a_entropy_norm"]) for item in records),
        "correct_a_support": statistics.fmean(float(item["s_a"]) for item in records),
        "correct_ab_support": statistics.fmean(float(item["s_ab"]) for item in records),
        "branches": branch_counts,
        "aux_scores": scores,
        "aux_advantages": advantages,
    }
    return payload, scores, advantages


def summarize_nothink_records(
    records: Sequence[dict], rescue_scale: float = 1.0
) -> tuple[dict, list]:
    """Aggregate already-gathered NoThink records and return their loss plans."""
    grouped = _groups(records, 8)
    plans = []
    group_rows = []
    reward_counts = Counter()
    for group in grouped:
        rewards = [float(item["primary_reward"]) for item in group]
        predicted_as = [item.get("predicted_a") for item in group]
        plan = build_nothink_rescue_plan(
            rewards,
            predicted_as,
            [int(item.get("sa_position", -1)) for item in group],
            group[0]["gold_as"],
            rescue_scale=rescue_scale,
        )
        plans.append(plan)
        for reward in rewards:
            reward_counts[reward] += 1
        group_rows.append({
            "rewards": rewards,
            "primary_zero_std": _zero_std(rewards),
            "all_zero": all(value == 0.0 for value in rewards),
            "concentration": plan.concentration,
            "rescue_active": plan.active,
            "gold_unique_a": plan.gold_unique_a,
            "any_gold_a": any(value >= 0.5 for value in rewards),
            "any_gold_ab": any(value >= 2.0 for value in rewards),
            "any_exact": any(value == 8.0 for value in rewards),
        })

    concentrations = [row["concentration"] for row in group_rows]
    all_zero_concentrations = [
        row["concentration"] for row in group_rows if row["all_zero"]
    ]

    def stratum(dense: bool) -> dict:
        selected = [row for row in group_rows if (row["gold_unique_a"] >= 3) == dense]
        rewards = [value for row in selected for value in row["rewards"]]
        return {
            "group_count": len(selected),
            "all_zero_rate": _rate(row["all_zero"] for row in selected),
            "rescue_active_rate": _rate(row["rescue_active"] for row in selected),
            "a_concentration_mean": statistics.fmean(row["concentration"] for row in selected) if selected else 0.0,
            "any_gold_a_hit_rate": _rate(row["any_gold_a"] for row in selected),
            "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        }

    total_candidates = len(records)
    payload = {
        "group_count": len(grouped),
        "primary_zero_std_rate": _rate(row["primary_zero_std"] for row in group_rows),
        "all_zero_group_rate": _rate(row["all_zero"] for row in group_rows),
        "rescue_active_group_rate": _rate(row["rescue_active"] for row in group_rows),
        "all_zero_same_a_rate": _rate(
            row["all_zero"] and row["concentration"] == 1.0 for row in group_rows
        ),
        # Preserve the original all-zero-only fields for existing DSR traces.
        "a_concentration_mean": statistics.fmean(all_zero_concentrations) if all_zero_concentrations else 0.0,
        "a_concentration_p90": _percentile(all_zero_concentrations, 0.90),
        # New unambiguous all-group concentration distribution.
        "group_a_concentration_mean": statistics.fmean(concentrations) if concentrations else 0.0,
        "group_a_concentration_p50": statistics.median(concentrations) if concentrations else 0.0,
        "group_a_concentration_p90": _percentile(concentrations, 0.90),
        "group_a_concentration_max": max(concentrations, default=0.0),
        "any_gold_a_hit_rate": _rate(row["any_gold_a"] for row in group_rows),
        "any_gold_ab_hit_rate": _rate(row["any_gold_ab"] for row in group_rows),
        "any_exact_hit_rate": _rate(row["any_exact"] for row in group_rows),
        "candidate_reward_distribution": {
            f"{value:g}": reward_counts[value] for value in REWARD_LEVELS
        },
        "candidate_reward_rate_distribution": {
            f"{value:g}": reward_counts[value] / total_candidates if total_candidates else 0.0
            for value in REWARD_LEVELS
        },
        "valid_sid_rate": sum(value != -1.0 for row in group_rows for value in row["rewards"]) / total_candidates if total_candidates else 0.0,
        "wrong_domain_rate": sum(value == -0.25 for row in group_rows for value in row["rewards"]) / total_candidates if total_candidates else 0.0,
        "gold_a_strata": {"sparse": stratum(False), "dense": stratum(True)},
        "rescue_active_groups": sum(plan.active for plan in plans),
        "sparse_rescue_count": sum(plan.active and plan.gold_unique_a < 3 for plan in plans),
        "dense_rescue_count": sum(plan.active and plan.gold_unique_a >= 3 for plan in plans),
        "plans": [_plan_dict(plan) for plan in plans],
    }
    return payload, plans
