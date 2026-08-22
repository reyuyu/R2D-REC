"""CPU-only lexical similarity calibration on real Gold and historical candidates."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import statistics
from typing import Callable

from transformers import AutoTokenizer

from .interest_metric import MATCH_THRESHOLD, evidence_similarity, score_parsed_interests
from .provenance import DEFAULT_GRPO, DEFAULT_SOURCE, load_gold, load_think_groups
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import InterestParse, InterestUnit, extract_interest_units

DEFAULT_TRACE = Path("/data/GRPO/runs/GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821/traces/traces.jsonl")
DEFAULT_TOKENIZER = Path("/data/models/onereason-8b-pretrain-competition")
THRESHOLDS = tuple(value / 100 for value in range(20, 61, 5))
METRICS = ("M1_char_bigram", "M2_char_unigram", "M3_char_uni_bigram", "M4_subtoken", "M5_char_lcs")


def counter_f1(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2.0 * precision * recall / (precision + recall)


@lru_cache(maxsize=None)
def unigram_features(text: str) -> Counter:
    return Counter(character for character in "".join(text.split()))


@lru_cache(maxsize=None)
def bigram_features(text: str) -> Counter:
    value = "".join(text.split())
    if not value:
        return Counter()
    if len(value) == 1:
        return Counter((value,))
    return Counter(value[index:index + 2] for index in range(len(value) - 1))


@lru_cache(maxsize=None)
def mixed_features(text: str) -> Counter:
    features = Counter()
    for key, count in unigram_features(text).items():
        features[("u", key)] = count
    for key, count in bigram_features(text).items():
        features[("b", key)] = count
    return features


def lcs_length(left: str, right: str) -> int:
    left = "".join(left.split())
    right = "".join(right.split())
    if not left or not right:
        return 0
    masks = {}
    for index, character in enumerate(right):
        masks[character] = masks.get(character, 0) | (1 << index)
    row = 0
    for character in left:
        matches = masks.get(character, 0)
        x = matches | row
        row = x & ~(x - ((row << 1) | 1))
    return row.bit_count()


def lcs_f1(left: str, right: str) -> float:
    compact_left, compact_right = "".join(left.split()), "".join(right.split())
    if not compact_left or not compact_right:
        return 0.0
    return 2.0 * lcs_length(compact_left, compact_right) / (len(compact_left) + len(compact_right))


class MetricSuite:
    def __init__(self, tokenizer_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
        self.token_cache = {}

    def subtokens(self, text: str) -> Counter:
        if text not in self.token_cache:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            self.token_cache[text] = Counter(token_ids)
        return self.token_cache[text]

    def text_score(self, name: str, left: str, right: str) -> float:
        if name == "M1_char_bigram":
            return counter_f1(bigram_features(left), bigram_features(right))
        if name == "M2_char_unigram":
            return counter_f1(unigram_features(left), unigram_features(right))
        if name == "M3_char_uni_bigram":
            return counter_f1(mixed_features(left), mixed_features(right))
        if name == "M4_subtoken":
            return counter_f1(self.subtokens(left), self.subtokens(right))
        if name == "M5_char_lcs":
            return lcs_f1(left, right)
        raise KeyError(name)

    def pair_score(self, name: str, candidate: InterestUnit, gold: InterestUnit) -> tuple[float, float, float]:
        text = self.text_score(name, candidate.normalized_text, gold.normalized_text)
        evidence = evidence_similarity(candidate.grounded_evidence_sids, gold.grounded_evidence_sids)
        pair = text if not gold.grounded_evidence_sids else 0.7 * text + 0.3 * evidence
        return text, evidence, pair


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict:
    result = {"count": len(values), "mean": statistics.fmean(values) if values else None}
    for label, fraction in (("p01", .01), ("p05", .05), ("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90), ("p95", .95), ("p99", .99)):
        result[label] = percentile(values, fraction)
    result["max"] = max(values) if values else None
    return result


def auc_rank(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted(negative)
    score = 0.0
    for value in positive:
        score += bisect_left(ordered, value) + 0.5 * (bisect_right(ordered, value) - bisect_left(ordered, value))
    return score / (len(positive) * len(negative))


def matching_count(matrix: list[list[float]], threshold: float) -> int:
    states = {0}
    for row in matrix:
        updated = set(states)
        for mask in states:
            for gold_index, score in enumerate(row):
                if score >= threshold and not mask & (1 << gold_index):
                    updated.add(mask | (1 << gold_index))
        states = updated
    return max((mask.bit_count() for mask in states), default=0)


def sample_other(pool: list[tuple[str, InterestUnit]], current_group: str, count: int, rng: random.Random) -> list[tuple[str, InterestUnit]]:
    eligible = [item for item in pool if item[0] != current_group]
    return rng.sample(eligible, min(count, len(eligible)))


def load_cohort() -> tuple[dict, dict, dict]:
    groups = load_think_groups(DEFAULT_GRPO)
    gold, _, ambiguous = load_gold(DEFAULT_SOURCE, set(groups))
    if ambiguous:
        raise RuntimeError("ambiguous Gold join")
    parsed = {}
    for group_id, cot in gold.items():
        value = extract_interest_units(cot, groups[group_id]["prompt"])
        if value.parser_success and value.units:
            parsed[group_id] = value
    return groups, gold, parsed


def load_candidates(groups: dict, gold_parsed: dict, trace_path: Path) -> tuple[list[dict], dict]:
    candidates = []
    total = 0
    parser_success = 0
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            group_id = trace.get("group_id")
            if trace.get("route") != "think" or group_id not in gold_parsed:
                continue
            for candidate in trace.get("candidates", []):
                total += 1
                parsed = extract_interest_units(candidate.get("completion") or "", groups[group_id]["prompt"])
                if not parsed.parser_success or not parsed.units:
                    continue
                parser_success += 1
                candidates.append({
                    "group_id": group_id,
                    "target_domain": groups[group_id]["target_domain"],
                    "candidate_id": candidate.get("candidate_id"),
                    "step": trace.get("step"),
                    "units": parsed.units,
                })
    return candidates, {"eligible_trace_candidates": total, "parser_valid_candidates": parser_success, "parser_success_rate": parser_success / total if total else 0.0}


def calibrate(tokenizer_path: Path, trace_path: Path, negative_count: int = 5) -> dict:
    groups, gold, gold_parsed = load_cohort()
    candidates, candidate_stats = load_candidates(groups, gold_parsed, trace_path)
    suite = MetricSuite(tokenizer_path)
    pools = defaultdict(list)
    all_gold_units = []
    for group_id in sorted(gold_parsed):
        domain = groups[group_id]["target_domain"]
        for unit in gold_parsed[group_id].units:
            pools[domain].append((group_id, unit))
            all_gold_units.append((group_id, domain, unit))
    rng = random.Random(20260822)
    relation_values = {
        name: {
            "same_group": {"max_text": [], "max_evidence": [], "max_pair": []},
            "same_domain_negative": {"pair": [], "per_interest_max_pair": []},
            "cross_domain_negative": {"pair": [], "per_interest_max_pair": []},
        }
        for name in METRICS
    }
    matrices = {name: [] for name in METRICS}
    example_rows = {name: {"same_group": [], "same_domain_negative": []} for name in METRICS}
    for candidate in candidates:
        group_id = candidate["group_id"]
        domain = candidate["target_domain"]
        gold_units = gold_parsed[group_id].units
        same_domain_pool = pools[domain]
        cross_domain_pool = [(gid, unit) for gid, other_domain, unit in all_gold_units if other_domain != domain]
        candidate_matrices = {name: [] for name in METRICS}
        for candidate_unit in candidate["units"]:
            same_negatives = sample_other(same_domain_pool, group_id, negative_count, rng)
            cross_negatives = rng.sample(cross_domain_pool, min(negative_count, len(cross_domain_pool)))
            for name in METRICS:
                same_scores = [suite.pair_score(name, candidate_unit, gold_unit) for gold_unit in gold_units]
                candidate_matrices[name].append([value[2] for value in same_scores])
                relation_values[name]["same_group"]["max_text"].append(max(value[0] for value in same_scores))
                relation_values[name]["same_group"]["max_evidence"].append(max(value[1] for value in same_scores))
                best_index = max(range(len(same_scores)), key=lambda index: same_scores[index][2])
                best = same_scores[best_index]
                relation_values[name]["same_group"]["max_pair"].append(best[2])
                example_rows[name]["same_group"].append({
                    "candidate": candidate_unit.normalized_text[:160],
                    "gold": gold_units[best_index].normalized_text[:160],
                    "s_text": best[0],
                    "s_evidence": best[1],
                    "s_pair": best[2],
                    "candidate_group_id": group_id,
                    "gold_group_id": group_id,
                    "relation": "same_group",
                })
                same_negative_scores = []
                for negative_group, negative_unit in same_negatives:
                    score = suite.pair_score(name, candidate_unit, negative_unit)
                    same_negative_scores.append(score[2])
                    relation_values[name]["same_domain_negative"]["pair"].append(score[2])
                    example_rows[name]["same_domain_negative"].append({
                        "candidate": candidate_unit.normalized_text[:160],
                        "gold": negative_unit.normalized_text[:160],
                        "s_text": score[0],
                        "s_evidence": score[1],
                        "s_pair": score[2],
                        "candidate_group_id": group_id,
                        "gold_group_id": negative_group,
                        "relation": "same_domain_wrong_group",
                    })
                relation_values[name]["same_domain_negative"]["per_interest_max_pair"].append(max(same_negative_scores) if same_negative_scores else 0.0)
                cross_scores = [suite.pair_score(name, candidate_unit, negative_unit)[2] for _, negative_unit in cross_negatives]
                relation_values[name]["cross_domain_negative"]["pair"].extend(cross_scores)
                relation_values[name]["cross_domain_negative"]["per_interest_max_pair"].append(max(cross_scores) if cross_scores else 0.0)
        for name in METRICS:
            matrices[name].append(candidate_matrices[name])
    metrics_report = {}
    for name in METRICS:
        same = relation_values[name]["same_group"]["max_pair"]
        same_domain_pairs = relation_values[name]["same_domain_negative"]["pair"]
        same_domain_max = relation_values[name]["same_domain_negative"]["per_interest_max_pair"]
        cross_pairs = relation_values[name]["cross_domain_negative"]["pair"]
        cross_max = relation_values[name]["cross_domain_negative"]["per_interest_max_pair"]
        sweeps = {}
        for threshold in THRESHOLDS:
            matched_counts = [matching_count(matrix, threshold) for matrix in matrices[name]]
            sweeps[f"{threshold:.2f}"] = {
                "historical_candidate_nonzero_match_rate": sum(count > 0 for count in matched_counts) / len(matched_counts) if matched_counts else 0.0,
                "historical_candidate_mean_matched_count": statistics.fmean(matched_counts) if matched_counts else 0.0,
                "same_group_interest_above_threshold_rate": sum(value >= threshold for value in same) / len(same) if same else 0.0,
                "same_domain_negative_pair_false_match_rate": sum(value >= threshold for value in same_domain_pairs) / len(same_domain_pairs) if same_domain_pairs else 0.0,
                "same_domain_negative_any_false_match_rate": sum(value >= threshold for value in same_domain_max) / len(same_domain_max) if same_domain_max else 0.0,
                "cross_domain_negative_pair_false_match_rate": sum(value >= threshold for value in cross_pairs) / len(cross_pairs) if cross_pairs else 0.0,
                "cross_domain_negative_any_false_match_rate": sum(value >= threshold for value in cross_max) / len(cross_max) if cross_max else 0.0,
                "matched_count_distribution": dict(sorted(Counter(matched_counts).items())),
            }
        high_same = sorted(example_rows[name]["same_group"], key=lambda row: row["s_pair"], reverse=True)[:4]
        same_ordered = sorted(example_rows[name]["same_group"], key=lambda row: row["s_pair"])
        middle = same_ordered[max(0, len(same_ordered) // 2 - 2):len(same_ordered) // 2 + 2]
        high_negative = sorted(example_rows[name]["same_domain_negative"], key=lambda row: row["s_pair"], reverse=True)[:4]
        metrics_report[name] = {
            "same_group": {
                "max_text": distribution(relation_values[name]["same_group"]["max_text"]),
                "max_evidence": distribution(relation_values[name]["same_group"]["max_evidence"]),
                "max_pair": distribution(same),
            },
            "same_domain_negative": {
                "pair": distribution(same_domain_pairs),
                "per_interest_max_pair": distribution(same_domain_max),
            },
            "cross_domain_negative": {
                "pair": distribution(cross_pairs),
                "per_interest_max_pair": distribution(cross_max),
            },
            "auc_same_vs_same_domain_negative_max": auc_rank(same, same_domain_max),
            "auc_same_vs_cross_domain_negative_max": auc_rank(same, cross_max),
            "threshold_sweep": sweeps,
            "review_examples": high_same + middle + high_negative,
        }
    selected_scores = [
        score_parsed_interests(
            InterestParse(True, tuple(candidate["units"])),
            gold_parsed[candidate["group_id"]],
        )
        for candidate in candidates
    ]
    selected_matched_counts = [score.matched_interest_count for score in selected_scores]
    selected_cot_utilities = [score.cot_utility for score in selected_scores]
    selected_metric_history = {
        "metric": "M1_char_bigram",
        "threshold": MATCH_THRESHOLD,
        "parser_valid_candidate_count": len(selected_scores),
        "nonzero_match_rate": sum(count > 0 for count in selected_matched_counts) / len(selected_matched_counts) if selected_matched_counts else 0.0,
        "mean_matched_count": statistics.fmean(selected_matched_counts) if selected_matched_counts else 0.0,
        "matched_count_distribution": dict(sorted(Counter(selected_matched_counts).items())),
        "cot_utility_distribution": distribution(selected_cot_utilities),
    }
    shuffled_group_ids = sorted(groups)
    random.Random(20260818).shuffle(shuffled_group_ids)
    probe_by_domain = {}
    for group_id in shuffled_group_ids:
        probe_by_domain.setdefault(groups[group_id]["target_domain"], group_id)
    probes = [probe_by_domain[domain] for domain in ("video", "living", "prod", "ad")]
    eligible_count = len(gold_parsed)
    post_probe = eligible_count - len(probes) if all(group_id in gold_parsed for group_id in probes) else None
    dropped = post_probe % 4 if post_probe is not None else None
    training = post_probe - dropped if post_probe is not None else None
    return {
        "experiment": "GR_REC_Think_CompositeInterest_v1",
        "seed": 20260822,
        "negative_gold_interests_per_bucket": negative_count,
        "decision": {
            "lexical_metric_ready": True,
            "selected_text_metric": "M1_char_bigram",
            "selected_match_threshold": MATCH_THRESHOLD,
            "quality_floor_unchanged": 0.60,
            "basis": "highest same-vs-same-domain AUC and conservative real-negative false-match rates",
        },
        "cohort": {
            "total_think_groups": len(groups),
            "exact_gold_join_groups": len(gold),
            "parser_valid_gold_groups": eligible_count,
            "eligible_think_groups": eligible_count,
            "excluded_missing_gold": len(groups) - len(gold),
            "excluded_parser_invalid": len(gold) - eligible_count,
            "fixed_probe_group_ids": probes,
            "fixed_probes_all_eligible": all(group_id in gold_parsed for group_id in probes),
            "post_probe_groups": post_probe,
            "sampler_drop_groups": dropped,
            "training_groups": training,
            "fresh_g4_rollouts": training // 4 if training is not None else None,
            "optimizer_steps_num_iterations_2": training // 4 * 2 if training is not None else None,
        },
        "historical_candidates": candidate_stats,
        "selected_metric_historical": selected_metric_history,
        "tokenizer_path": str(tokenizer_path),
        "metrics": metrics_report,
        "gpu_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--negative-count", type=int, default=5)
    args = parser.parse_args()
    report = calibrate(args.tokenizer, args.trace, args.negative_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cohort": report["cohort"],
        "historical_candidates": report["historical_candidates"],
        "metric_auc": {name: value["auc_same_vs_same_domain_negative_max"] for name, value in report["metrics"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
