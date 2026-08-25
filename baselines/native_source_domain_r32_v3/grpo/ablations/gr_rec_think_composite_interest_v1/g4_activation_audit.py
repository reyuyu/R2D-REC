"""CPU-only activation audit on immutable historical Think G4 rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable, Sequence

from .interest_metric import (
    MATCH_QUALITY_FLOOR,
    MATCH_THRESHOLD,
    beam_primary_composite_rewards,
    beam_utility,
    composite_reward,
    population_advantages,
    score_parsed_interests,
)
from .provenance import DEFAULT_GRPO, DEFAULT_SOURCE, load_gold, load_think_groups
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units


DEFAULT_TRACE = Path(
    "/data/GRPO/runs/GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821/"
    "traces/traces.jsonl"
)
PROBE_SEED = 20260818
DOMAINS = ("video", "prod", "ad", "living")


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: Iterable[float]) -> dict:
    items = [float(value) for value in values]
    return {
        "count": len(items),
        "mean": statistics.fmean(items) if items else None,
        "p25": percentile(items, 0.25),
        "p50": percentile(items, 0.50),
        "p75": percentile(items, 0.75),
        "min": min(items) if items else None,
        "max": max(items) if items else None,
    }


def population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average
        start = end
    return result


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    return cosine(
        [value - left_mean for value in left],
        [value - right_mean for value in right],
    )


def correlation(rows: Sequence[dict], left: str, right: str) -> dict:
    valid = [row for row in rows if row[left] is not None and row[right] is not None]
    left_values = [row[left] for row in valid]
    right_values = [row[right] for row in valid]
    return {
        "count": len(valid),
        "pearson": pearson(left_values, right_values),
        "spearman": pearson(ranks(left_values), ranks(right_values)),
    }


def audit_reward_vectors(raw_beam: Sequence[float], cot_utility: Sequence[float]) -> dict:
    if len(raw_beam) != 4 or len(cot_utility) != 4:
        raise ValueError("activation audit requires an intact G4")
    raw = [float(value) for value in raw_beam]
    cot = [float(value) for value in cot_utility]
    composite, tiebreak_scale = beam_primary_composite_rewards(raw, cot)
    return {
        "raw_beam_reward_vector": raw,
        "beam_utility_vector": [beam_utility(value) for value in raw],
        "A_beam_raw": population_advantages(raw),
        "U_cot_vector": cot,
        "composite_reward_vector": composite,
        "interest_tiebreak_scale": tiebreak_scale,
        "A_composite": population_advantages(composite),
        "beam_raw_population_std": population_std(raw),
        "composite_population_std": population_std(composite),
    }


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def raw_n_bucket(value: int) -> str:
    return str(value) if value < 5 else "5+"


def beam_bucket(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 2.0:
        return "0.5_near"
    if value < 8.0:
        return "2_near"
    return "8+"


def mean_bucket(rows: Sequence[dict]) -> dict:
    fields = {
        "mean_u_cot": "u_cot",
        "mean_matched_count": "matched",
        "mean_beam_raw_reward": "beam_raw",
        "mean_composite_reward": "composite",
    }
    result = {"count": len(rows)}
    for output, source in fields.items():
        result[output] = statistics.fmean(row[source] for row in rows) if rows else None
    return result


def domain_counter(group_ids: Iterable[str], groups: dict[str, dict]) -> dict[str, int]:
    counts = Counter(groups[group_id].get("target_domain", "unknown") for group_id in group_ids)
    return {domain: counts.get(domain, 0) for domain in DOMAINS}


def cohort_audit(groups: dict[str, dict], gold: dict[str, str], parsed_gold: dict) -> dict:
    all_ids, joined_ids, eligible_ids = set(groups), set(gold), set(parsed_gold)
    shuffled = sorted(groups)
    random.Random(PROBE_SEED).shuffle(shuffled)
    probes_by_domain = {}
    for group_id in shuffled:
        probes_by_domain.setdefault(groups[group_id]["target_domain"], group_id)
    probes = [probes_by_domain[domain] for domain in ("video", "living", "prod", "ad")]
    projected = sorted(eligible_ids - set(probes))
    random.Random(PROBE_SEED).shuffle(projected)
    dropped = len(projected) % 4
    training_ids = projected[:-dropped] if dropped else projected
    original = domain_counter(all_ids, groups)
    eligible = domain_counter(eligible_ids, groups)
    eligible_rates = {
        domain: eligible[domain] / original[domain] if original[domain] else None
        for domain in DOMAINS
    }
    overall_rate = len(eligible_ids) / len(all_ids)
    skew_domains = [
        domain for domain, value in eligible_rates.items()
        if value is not None and abs(value - overall_rate) > 0.05
    ]
    return {
        "original_think_groups_by_domain": original,
        "eligible_groups_by_domain": eligible,
        "excluded_missing_gold_by_domain": domain_counter(all_ids - joined_ids, groups),
        "excluded_parser_invalid_by_domain": domain_counter(joined_ids - eligible_ids, groups),
        "projected_training_groups_by_domain": domain_counter(training_ids, groups),
        "eligible_rate_by_domain": eligible_rates,
        "overall_eligible_rate": overall_rate,
        "domain_cohort_skew_risk": bool(skew_domains),
        "domain_cohort_skew_risk_rule": "absolute eligible-rate deviation from overall > 5 percentage points",
        "skew_domains": skew_domains,
        "fixed_probe_group_ids": probes,
        "fixed_probes_all_eligible": all(group_id in eligible_ids for group_id in probes),
        "projected_sampler_drop_group_ids": projected[-dropped:] if dropped else [],
        "projected_training_group_count": len(training_ids),
        "projection_note": "No runner exists; training domains use seed-20260818 post-probe shuffle and G4 tail truncation.",
    }


def example(group: dict) -> dict:
    keys = (
        "group_id", "rollout_id", "step", "target_domain",
        "raw_beam_reward_vector", "U_cot_vector", "matched_interest_count_vector",
        "raw_n_vector", "grounded_n_vector", "composite_reward_vector",
        "beam_top_candidate_id", "composite_top_candidate_id",
    )
    return {key: group[key] for key in keys}


def activation_audit(
    trace_path: Path = DEFAULT_TRACE,
    grpo_path: Path = DEFAULT_GRPO,
    source_path: Path = DEFAULT_SOURCE,
) -> dict:
    groups = load_think_groups(grpo_path)
    gold, _, ambiguous = load_gold(source_path, set(groups))
    if ambiguous:
        raise RuntimeError("ambiguous Gold join")
    parsed_gold = {}
    for group_id, cot in gold.items():
        parsed = extract_interest_units(cot, groups[group_id]["prompt"])
        if parsed.parser_success and parsed.units:
            parsed_gold[group_id] = parsed

    trace_counts = Counter()
    audited_groups, candidate_rows = [], []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            if trace.get("route") != "think":
                continue
            trace_counts["historical_think_g4"] += 1
            group_id = trace.get("group_id")
            if group_id not in parsed_gold:
                trace_counts["excluded_ineligible_gold"] += 1
                continue
            candidates = trace.get("candidates") or []
            if len(candidates) != 4:
                trace_counts["excluded_non_g4"] += 1
                continue
            parsed_candidates = [
                extract_interest_units(item.get("completion") or "", groups[group_id]["prompt"])
                for item in candidates
            ]
            if any(not parsed.parser_success or not parsed.units for parsed in parsed_candidates):
                trace_counts["excluded_candidate_parser_invalid_g4"] += 1
                continue

            scores = [score_parsed_interests(parsed, parsed_gold[group_id]) for parsed in parsed_candidates]
            raw_beam = [float(item.get("reward") or 0.0) for item in candidates]
            rewards = audit_reward_vectors(raw_beam, [score.cot_utility for score in scores])
            raw_n = [len(parsed.units) for parsed in parsed_candidates]
            grounded_n = [
                sum(bool(unit.grounded_evidence_sids) for unit in parsed.units)
                for parsed in parsed_candidates
            ]
            grounding = [grounded / raw if raw else None for raw, grounded in zip(raw_n, grounded_n)]
            matched = [score.matched_interest_count for score in scores]
            tiers = [score.coverage_tier for score in scores]
            qualities = [score.match_quality for score in scores]
            similarities = [score.mean_match_similarity for score in scores]
            lengths = [int(item.get("completion_length") or 0) for item in candidates]
            beam_top = argmax(raw_beam)
            composite_top = argmax(rewards["composite_reward_vector"])
            record = {
                "group_id": group_id,
                "rollout_id": trace.get("rollout_id"),
                "step": trace.get("step"),
                "target_domain": groups[group_id]["target_domain"],
                **rewards,
                "matched_interest_count_vector": matched,
                "coverage_tier_vector": tiers,
                "Q_vector": qualities,
                "mean_match_similarity_vector": similarities,
                "raw_n_vector": raw_n,
                "grounded_n_vector": grounded_n,
                "grounding_coverage_vector": grounding,
                "completion_length_vector": lengths,
                "unique_u_cot_value_count": len(set(rewards["U_cot_vector"])),
                "beam_raw_zero_std": rewards["beam_raw_population_std"] == 0.0,
                "u_cot_zero_std": population_std(rewards["U_cot_vector"]) == 0.0,
                "composite_zero_std": rewards["composite_population_std"] == 0.0,
                "rescued_zero_std": rewards["beam_raw_population_std"] == 0.0 and rewards["composite_population_std"] > 0.0,
                "beam_top_candidate_id": candidates[beam_top].get("candidate_id", beam_top),
                "composite_top_candidate_id": candidates[composite_top].get("candidate_id", composite_top),
                "top_winner_changed": beam_top != composite_top,
                "advantage_sign_changed_candidate_count": sum(
                    sign(old) != sign(new)
                    for old, new in zip(rewards["A_beam_raw"], rewards["A_composite"])
                ),
                "beam_composite_advantage_cosine": cosine(rewards["A_beam_raw"], rewards["A_composite"]),
            }
            audited_groups.append(record)
            for index, item in enumerate(candidates):
                candidate_rows.append({
                    "group_id": group_id,
                    "candidate_id": item.get("candidate_id", index),
                    "beam_raw": raw_beam[index],
                    "u_cot": rewards["U_cot_vector"][index],
                    "matched": matched[index],
                    "quality": qualities[index],
                    "mean_similarity": similarities[index],
                    "raw_n": raw_n[index],
                    "grounded_n": grounded_n[index],
                    "grounding_coverage": grounding[index],
                    "completion_length": lengths[index],
                    "composite": rewards["composite_reward_vector"][index],
                })

    total = len(audited_groups)
    beam_zero = [group for group in audited_groups if group["beam_raw_zero_std"]]
    beam_active = [group for group in audited_groups if not group["beam_raw_zero_std"]]
    cot_all_zero = [group for group in audited_groups if all(value == 0.0 for value in group["U_cot_vector"])]
    cot_active = [group for group in audited_groups if not group["u_cot_zero_std"]]
    composite_zero = [group for group in audited_groups if group["composite_zero_std"]]
    rescued = [group for group in audited_groups if group["rescued_zero_std"]]
    changed = [group for group in beam_active if group["top_winner_changed"]]
    changed_signs = sum(group["advantage_sign_changed_candidate_count"] for group in beam_active)
    cosines = [
        group["beam_composite_advantage_cosine"] for group in beam_active
        if group["beam_composite_advantage_cosine"] is not None
    ]

    unique = Counter(group["unique_u_cot_value_count"] for group in audited_groups)
    k_counts = Counter("4+" if row["matched"] >= 4 else str(row["matched"]) for row in candidate_rows)
    coverage_group_counts = {
        "at_least_one_k_ge_1": sum(max(group["matched_interest_count_vector"]) >= 1 for group in audited_groups),
        "at_least_one_k_ge_2": sum(max(group["matched_interest_count_vector"]) >= 2 for group in audited_groups),
        "at_least_one_k_ge_3": sum(max(group["matched_interest_count_vector"]) >= 3 for group in audited_groups),
        "full_coverage_candidate_exists": sum(any(tier == 1.0 for tier in group["coverage_tier_vector"]) for group in audited_groups),
    }
    quality_candidates = sum(row["quality"] > 0.0 for row in candidate_rows)
    quality_groups = sum(any(value > 0.0 for value in group["Q_vector"]) for group in audited_groups)

    correlations = {
        left + "_vs_" + right: correlation(candidate_rows, left, right)
        for left, right in (
            ("u_cot", "raw_n"),
            ("u_cot", "grounded_n"),
            ("u_cot", "grounding_coverage"),
            ("matched", "raw_n"),
            ("matched", "grounded_n"),
            ("u_cot", "completion_length"),
        )
    }

    raw_buckets = {}
    for label in ("0", "1", "2", "3", "4", "5+"):
        rows = [row for row in candidate_rows if raw_n_bucket(row["raw_n"]) == label]
        raw_buckets[label] = mean_bucket(rows)
        raw_buckets[label]["beam_buckets"] = {
            beam_label: mean_bucket([row for row in rows if beam_bucket(row["beam_raw"]) == beam_label])
            for beam_label in ("0", "0.5_near", "2_near", "8+")
        }
    matched_beam = {}
    for beam_label in ("0", "0.5_near", "2_near", "8+"):
        one = [row for row in candidate_rows if row["raw_n"] == 1 and beam_bucket(row["beam_raw"]) == beam_label]
        multi = [row for row in candidate_rows if row["raw_n"] in (3, 4) and beam_bucket(row["beam_raw"]) == beam_label]
        one_summary, multi_summary = mean_bucket(one), mean_bucket(multi)
        matched_beam[beam_label] = {
            "raw_n_1": one_summary,
            "raw_n_3_or_4": multi_summary,
            "multi_minus_one_mean_composite": (
                multi_summary["mean_composite_reward"] - one_summary["mean_composite_reward"]
                if one and multi else None
            ),
        }

    collapse_examples = {
        "A_beam_equal_matched_counts_differ": [
            example(group) for group in beam_zero if len(set(group["matched_interest_count_vector"])) > 1
        ][:5],
        "B_beam_equal_raw_n_diff_ge_2": [
            example(group) for group in beam_zero if max(group["raw_n_vector"]) - min(group["raw_n_vector"]) >= 2
        ][:5],
        "C_high_beam_low_u_cot": [
            example(group) for group in audited_groups
            if any(beam >= 8.0 and cot <= 0.16 for beam, cot in zip(group["raw_beam_reward_vector"], group["U_cot_vector"]))
        ][:5],
        "D_low_beam_high_u_cot": [
            example(group) for group in audited_groups
            if any(beam <= 0.5 and cot >= 0.56 for beam, cot in zip(group["raw_beam_reward_vector"], group["U_cot_vector"]))
        ][:5],
        "E_composite_winner_changed": [example(group) for group in changed][:5],
    }
    cohort = cohort_audit(groups, gold, parsed_gold)
    return {
        "experiment": "GR_REC_Think_CompositeInterest_v1",
        "audit": "REAL G4 COMPOSITE REWARD ACTIVATION AUDIT",
        "source": {
            "trace": str(trace_path),
            "immutable_grouping": "Each trace Think G4 retained intact; no candidate regrouping.",
        },
        "frozen_formula": {
            "match_threshold": MATCH_THRESHOLD,
            "match_quality_floor": MATCH_QUALITY_FLOOR,
            "u_cot": "0.8 * coverage_tier + 0.2 * Q",
            "r_total": "0.6 * U_beam + 0.4 * U_cot",
            "advantage": "(reward - G4 mean) / (population_std + 1e-4)",
        },
        "trace_filter": {
            **dict(trace_counts),
            "audited_complete_parser_valid_g4": total,
            "audited_candidates": len(candidate_rows),
        },
        "activation": {
            "beam_raw_zero_std": {"count": len(beam_zero), "rate": rate(len(beam_zero), total)},
            "u_cot_all_zero": {"count": len(cot_all_zero), "rate": rate(len(cot_all_zero), total)},
            "u_cot_variance_active": {"count": len(cot_active), "rate": rate(len(cot_active), total)},
            "composite_zero_std": {"count": len(composite_zero), "rate": rate(len(composite_zero), total)},
            "rescued_zero_std": {
                "count": len(rescued),
                "rate_over_all_g4": rate(len(rescued), total),
                "rate_over_beam_zero_std_g4": rate(len(rescued), len(beam_zero)),
            },
        },
        "beam_active_group_effect": {
            "beam_variance_active_g4": len(beam_active),
            "top_winner_changed": {"count": len(changed), "rate": rate(len(changed), len(beam_active))},
            "winner_rule": "Stable lowest-candidate-id argmax for both vectors.",
            "advantage_sign_changed_candidates": {
                "count": changed_signs,
                "rate": rate(changed_signs, len(beam_active) * 4),
            },
            "advantage_cosine": distribution(cosines),
        },
        "cot_group_variation": {
            "unique_u_cot_value_count_distribution": {str(key): unique.get(key, 0) for key in range(1, 5)},
            "all_four_equal": {"count": unique[1], "rate": rate(unique[1], total)},
            "at_least_two_values": {
                "count": sum(value for key, value in unique.items() if key >= 2),
                "rate": rate(sum(value for key, value in unique.items() if key >= 2), total),
            },
            "at_least_three_values": {
                "count": sum(value for key, value in unique.items() if key >= 3),
                "rate": rate(sum(value for key, value in unique.items() if key >= 3), total),
            },
            "four_distinct": {"count": unique[4], "rate": rate(unique[4], total)},
        },
        "coverage_activation": {
            "candidate_k_distribution": {
                label: {"count": k_counts[label], "rate": rate(k_counts[label], len(candidate_rows))}
                for label in ("0", "1", "2", "3", "4+")
            },
            "group": {
                key: {"count": value, "rate": rate(value, total)}
                for key, value in coverage_group_counts.items()
            },
        },
        "quality_activation": {
            "q_positive_candidates": {"count": quality_candidates, "rate": rate(quality_candidates, len(candidate_rows))},
            "q_positive_groups": {"count": quality_groups, "rate": rate(quality_groups, total)},
            "mean_match_similarity_all_candidates": distribution(row["mean_similarity"] for row in candidate_rows),
            "mean_match_similarity_matched_candidates": distribution(
                row["mean_similarity"] for row in candidate_rows if row["matched"] > 0
            ),
            "quality_branch_historically_dormant": quality_candidates == 0,
        },
        "structure_correlations": correlations,
        "raw_n_shortcut_audit": {
            "raw_n_buckets": raw_buckets,
            "beam_matched_raw_n_1_vs_3_or_4": matched_beam,
        },
        "domain_cohort": cohort,
        "collapse_examples": collapse_examples,
        "groups": audited_groups,
        "flags": {
            "quality_branch_historically_dormant": quality_candidates == 0,
            "domain_cohort_skew_risk": cohort["domain_cohort_skew_risk"],
        },
        "runtime_status": {
            "gpu_used": False,
            "generation": False,
            "training_started": False,
            "optimizer_step": False,
            "formal_runner_created": False,
            "frontend_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--grpo", type=Path, default=DEFAULT_GRPO)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = activation_audit(args.trace, args.grpo, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "trace_filter": report["trace_filter"],
        "activation": report["activation"],
        "beam_active_group_effect": report["beam_active_group_effect"],
        "quality_activation": report["quality_activation"],
        "flags": report["flags"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
