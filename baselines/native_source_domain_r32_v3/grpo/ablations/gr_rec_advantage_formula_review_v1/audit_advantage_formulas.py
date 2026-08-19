"""CPU-only historical audit of Current, ExactFloor and Centered Exact-Clamp."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
EXACT_FLOOR_DIR = GRPO_ROOT / "ablations" / "gr_rec_exact_floor_v1"
sys.path.insert(0, str(EXACT_FLOOR_DIR))

from audit_historical_rewards import (  # noqa: E402
    DEFAULT_GR_REC,
    DEFAULT_SIMPLE,
    current_advantages,
    gold_bucket,
    load_gr_rec_probes,
    load_simple_forensic,
    reward_bucket,
    reward_pattern,
)
from exact_floor import exact_floor_advantages  # noqa: E402

try:
    from .formulas import centered_exact_clamp_advantages
except ImportError:  # Direct script entry point.
    from formulas import centered_exact_clamp_advantages


RULES = ("current", "exact_floor", "exact_clamp")
LEVELS = ("0", "0.5", "2", "8", ">8", "other")


def formula_outputs(rewards: list[float]) -> dict[str, list[float]]:
    tensor = torch.tensor(rewards, dtype=torch.float64)
    group_size = len(rewards)
    return {
        "current": current_advantages(tensor, group_size).tolist(),
        "exact_floor": exact_floor_advantages(tensor, group_size).tolist(),
        "exact_clamp": centered_exact_clamp_advantages(tensor, group_size).tolist(),
    }


def build_rows(groups: list[dict]):
    group_rows = []
    candidates = []
    for group_index, group in enumerate(groups):
        rewards = group["rewards"]
        outputs = formula_outputs(rewards)
        pattern = reward_pattern(rewards)
        row = {
            **group,
            "group_index": group_index,
            "gold_bucket": gold_bucket(group["gold_count"]),
            "reward_pattern": pattern,
            "advantages": outputs,
            "abs_sums": {rule: sum(abs(value) for value in outputs[rule]) for rule in RULES},
        }
        group_rows.append(row)
        for candidate_index, reward in enumerate(rewards):
            candidates.append({
                "group_index": group_index,
                "source": group["source"],
                "route": group["route"],
                "domain": group["domain"],
                "gold_count": group["gold_count"],
                "gold_bucket": row["gold_bucket"],
                "reward_pattern": pattern,
                "reward": reward,
                "reward_level": reward_bucket(reward),
                "advantages": {rule: outputs[rule][candidate_index] for rule in RULES},
            })
    return group_rows, candidates


def _candidate_rule_stats(rows: list[dict], rule: str, total_mass: float) -> dict:
    values = [row["advantages"][rule] for row in rows]
    count = len(values)
    mass = sum(abs(value) for value in values)
    return {
        "mean_advantage": sum(values) / count if count else None,
        "mean_abs_advantage": mass / count if count else None,
        "total_abs_advantage": mass,
        "share_of_total_abs_advantage": mass / total_mass if total_mass else None,
        "positive_rate": sum(value > 0 for value in values) / count if count else None,
        "negative_rate": sum(value < 0 for value in values) / count if count else None,
        "zero_rate": sum(value == 0 for value in values) / count if count else None,
    }


def candidate_breakdown(candidates: list[dict], key) -> dict:
    buckets = defaultdict(list)
    for row in candidates:
        buckets[str(key(row))].append(row)
    totals = {
        rule: sum(abs(row["advantages"][rule]) for row in candidates) for rule in RULES
    }
    output = {}
    for label, rows in sorted(buckets.items()):
        output[label] = {"candidate_count": len(rows)}
        for rule in RULES:
            output[label][rule] = _candidate_rule_stats(rows, rule, totals[rule])
    return output


def bias_breakdown(groups: list[dict], candidates: list[dict], key) -> dict:
    group_counts = defaultdict(set)
    buckets = defaultdict(list)
    for row in candidates:
        label = str(key(row))
        buckets[label].append(row)
        group_counts[label].add(row["group_index"])
    totals = {
        rule: sum(abs(row["advantages"][rule]) for row in candidates) for rule in RULES
    }
    output = {}
    for label, rows in sorted(buckets.items()):
        output[label] = {
            "group_count": len(group_counts[label]),
            "candidate_count": len(rows),
        }
        for rule in RULES:
            output[label][rule] = _candidate_rule_stats(rows, rule, totals[rule])
    return output


def equal_high_summary(groups: list[dict]) -> dict:
    predicates = {
        "all_reward_gt_8": lambda values: all(value > 8 for value in values),
        "all_reward_ge_8": lambda values: all(value >= 8 for value in values),
        "group_mean_gt_8": lambda values: sum(values) / len(values) > 8,
        "equal_high": lambda values: min(values) >= 8 and max(values) == min(values),
    }
    output = {}
    for name, predicate in predicates.items():
        selected = [group for group in groups if predicate(group["rewards"])]
        output[name] = {"group_count": len(selected)}
        for rule in RULES:
            masses = [group["abs_sums"][rule] for group in selected]
            output[name][rule] = {
                "mean_group_abs_sum": sum(masses) / len(masses) if masses else None,
                "total_abs_sum": sum(masses),
            }
    return output


def pattern_summary(groups: list[dict]) -> dict:
    buckets = defaultdict(list)
    for group in groups:
        buckets[group["reward_pattern"]].append(group)
    output = {}
    for pattern, rows in sorted(buckets.items()):
        output[pattern] = {"group_count": len(rows)}
        for rule in RULES:
            masses = [row["abs_sums"][rule] for row in rows]
            output[pattern][rule] = {
                "mean_group_abs_sum": sum(masses) / len(masses),
                "total_abs_sum": sum(masses),
            }
    return output


def _mean_positive(candidates: list[dict], rule: str, predicate) -> float | None:
    values = [
        row["advantages"][rule]
        for row in candidates
        if predicate(row) and row["advantages"][rule] > 0
    ]
    return sum(values) / len(values) if values else None


def key_ratios(groups: list[dict], candidates: list[dict], patterns: dict) -> dict:
    total = {
        rule: sum(abs(row["advantages"][rule]) for row in candidates) for rule in RULES
    }
    output = {}
    for rule in RULES:
        positive_high = _mean_positive(candidates, rule, lambda row: row["reward"] > 8)
        positive_a = _mean_positive(candidates, rule, lambda row: row["reward_level"] == "0.5")
        positive_by_level = {
            level: _mean_positive(
                candidates, rule, lambda row, expected=level: row["reward_level"] == expected
            )
            for level in ("0.5", "2", "8", ">8")
        }
        multi = patterns.get("MULTI_EXACT", {}).get(rule, {}).get("mean_group_abs_sum")
        a_only = patterns.get("A_ONLY", {}).get(rule, {}).get("mean_group_abs_sum")
        high_mass = sum(
            abs(row["advantages"][rule]) for row in candidates if row["reward"] >= 8
        )
        dense_mass = sum(
            abs(row["advantages"][rule]) for row in candidates if row["gold_count"] >= 5
        )
        video_mass = sum(
            abs(row["advantages"][rule]) for row in candidates if row["domain"] == "video"
        )
        output[rule] = {
            "mean_positive_advantage_by_reward_level": positive_by_level,
            "mean_positive_r_gt_8_over_r_0_5": (
                positive_high / positive_a if positive_high is not None and positive_a else None
            ),
            "multi_exact_mean_mass_over_a_only": multi / a_only if multi is not None and a_only else None,
            "r_ge_8_mass_share": high_mass / total[rule] if total[rule] else None,
            "gold_count_ge_5_mass_share": dense_mass / total[rule] if total[rule] else None,
            "video_mass_share": video_mass / total[rule] if total[rule] else None,
            "total_abs_advantage": total[rule],
        }
    return output


def archetype_results() -> list[dict]:
    patterns = [
        [0, 0, 0, 0],
        [0, 0, 0, 0.5],
        [0, 0, 0, 2],
        [0, 0, 0, 8],
        [8, 8, 8, 9],
        [8, 8, 8, 12],
        [8, 2, 3, 9],
        [8, 8, 8, 8],
        [12, 12, 12, 12],
        [12, 12, 12, 14],
    ]
    return [{"rewards": values, **formula_outputs(values)} for values in patterns]


def summarize(groups: list[dict]) -> tuple[dict, list[dict]]:
    group_rows, candidates = build_rows(groups)
    patterns = pattern_summary(group_rows)
    sources = sorted({group["source"] for group in groups})
    by_source = {}
    for source in sources:
        source_groups = [group for group in group_rows if group["source"] == source]
        source_candidates = [row for row in candidates if row["source"] == source]
        source_patterns = pattern_summary(source_groups)
        by_source[source] = {
            "group_count": len(source_groups),
            "candidate_count": len(source_candidates),
            "reward_levels": candidate_breakdown(source_candidates, lambda row: row["reward_level"]),
            "bias": {
                "route": bias_breakdown(source_groups, source_candidates, lambda row: row["route"]),
                "domain": bias_breakdown(source_groups, source_candidates, lambda row: row["domain"]),
                "gold_count": bias_breakdown(source_groups, source_candidates, lambda row: row["gold_bucket"]),
            },
            "equal_high": equal_high_summary(source_groups),
            "patterns": source_patterns,
            "key_ratios": key_ratios(source_groups, source_candidates, source_patterns),
        }
    result = {
        "metadata": {
            "group_count": len(group_rows),
            "candidate_count": len(candidates),
            "advantage_mass_proxy": "sum(abs(advantage)); proxy only, not gradient norm",
            "gradient_caveats": ["token length", "policy logprob gradient", "PPO ratio", "clip"],
            "sources": {
                "GR_REC_v1_probe": "fixed Probe rewards, not full training coverage",
                "DSR_Simple_training": "full simple_forensic primary rewards; auxiliary rewards excluded",
            },
        },
        "archetypes": archetype_results(),
        "overall": {
            "reward_levels": candidate_breakdown(candidates, lambda row: row["reward_level"]),
            "bias": {
                "route": bias_breakdown(group_rows, candidates, lambda row: row["route"]),
                "domain": bias_breakdown(group_rows, candidates, lambda row: row["domain"]),
                "gold_count": bias_breakdown(group_rows, candidates, lambda row: row["gold_bucket"]),
            },
            "equal_high": equal_high_summary(group_rows),
            "patterns": patterns,
            "key_ratios": key_ratios(group_rows, candidates, patterns),
        },
        "by_source": by_source,
    }
    return result, group_rows


def write_groups_csv(path: Path, groups: list[dict]) -> None:
    fields = [
        "source", "route", "domain", "gold_count", "reward_pattern", "rewards",
        "current_adv", "exact_floor_adv", "exact_clamp_adv",
        "current_abs_sum", "floor_abs_sum", "clamp_abs_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            writer.writerow({
                "source": group["source"],
                "route": group["route"],
                "domain": group["domain"],
                "gold_count": group["gold_count"],
                "reward_pattern": group["reward_pattern"],
                "rewards": json.dumps(group["rewards"], separators=(",", ":")),
                "current_adv": json.dumps(group["advantages"]["current"], separators=(",", ":")),
                "exact_floor_adv": json.dumps(group["advantages"]["exact_floor"], separators=(",", ":")),
                "exact_clamp_adv": json.dumps(group["advantages"]["exact_clamp"], separators=(",", ":")),
                "current_abs_sum": group["abs_sums"]["current"],
                "floor_abs_sum": group["abs_sums"]["exact_floor"],
                "clamp_abs_sum": group["abs_sums"]["exact_clamp"],
            })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-rec-probes", type=Path, default=DEFAULT_GR_REC)
    parser.add_argument("--simple-forensic", type=Path, default=DEFAULT_SIMPLE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    groups = list(load_gr_rec_probes(args.gr_rec_probes))
    groups.extend(load_simple_forensic(args.simple_forensic))
    result, group_rows = summarize(groups)
    for source in ("overall", *result["by_source"]):
        section = result["overall"] if source == "overall" else result["by_source"][source]
        for level in ("8", ">8"):
            stats = section["reward_levels"].get(level, {}).get("exact_clamp")
            if stats and stats["negative_rate"] != 0:
                raise AssertionError(f"Centered Exact-Clamp negative high reward: {source}/{level}")
    equal_high = result["overall"]["equal_high"]["equal_high"]["exact_clamp"]
    if equal_high["total_abs_sum"] != 0:
        raise AssertionError("Centered Exact-Clamp equal-high mass must be zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "advantage_formula_comparison.json"
    csv_path = args.output_dir / "advantage_formula_groups.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_groups_csv(csv_path, group_rows)
    print(json.dumps({
        "groups": len(group_rows),
        "candidates": result["metadata"]["candidate_count"],
        "json": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
