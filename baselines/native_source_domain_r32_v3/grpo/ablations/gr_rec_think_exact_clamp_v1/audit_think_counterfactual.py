"""CPU-only Current-vs-ThinkExactClamp audit over saved Think rewards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

try:
    from .think_exact_clamp import think_exact_clamp_advantages
except ImportError:  # Direct script entry point.
    from think_exact_clamp import think_exact_clamp_advantages


DEFAULT_GR_REC = Path(
    "/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/probes.jsonl"
)
DEFAULT_SIMPLE = Path(
    "/data/GRPO/runs/GR-REC-DSR-SIMPLE-V1-FULL-E1-GATE200-20260819/simple_forensic.jsonl"
)
RULES = ("current", "think_exact_clamp")


def current_advantages(rewards: torch.Tensor) -> torch.Tensor:
    grouped = rewards.reshape(-1, 4)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, correction=0, keepdim=True)
    return ((grouped - mean) / (std + 1e-4)).reshape_as(rewards)


def gold_bucket(count: int) -> str:
    return "1-2" if count <= 2 else "3-4" if count <= 4 else "5+"


def reward_level(value: float) -> str:
    for target, label in ((0.0, "0"), (0.5, "0.5"), (2.0, "2"), (8.0, "8")):
        if abs(value - target) < 1e-8:
            return label
    return ">8" if value > 8 else "other"


def reward_pattern(values: list[float]) -> str:
    exact = sum(abs(value - 8.0) < 1e-8 for value in values)
    above = sum(value > 8.0 for value in values)
    if all(abs(value) < 1e-8 for value in values):
        return "ALL_ZERO"
    if all(abs(value - 8.0) < 1e-8 for value in values):
        return "EXACT_SATURATED"
    if exact >= 2:
        return "MULTI_EXACT"
    if above:
        return "MIXED_HIGH"
    if exact == 1:
        return "SINGLE_EXACT"
    if any(abs(value - 2.0) < 1e-8 for value in values):
        return "AB_SIGNAL"
    if any(abs(value - 0.5) < 1e-8 for value in values):
        return "A_ONLY"
    return "OTHER"


def load_gr_rec_think(path: Path):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        candidates = row.get("think", {}).get("candidates") or []
        rewards = [candidate.get("reward") for candidate in candidates]
        if len(rewards) == 4 and all(value is not None for value in rewards):
            yield {
                "source": "GR_REC_v1_probe",
                "route": "think",
                "domain": row.get("target_domain", "unknown"),
                "gold_count": len(row.get("gold_sids") or []),
                "rewards": [float(value) for value in rewards],
            }


def load_simple_think(path: Path):
    pending = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("type") == "think_candidate":
            key = (row.get("step"), row.get("rollout_id"), row.get("group_id"))
            pending[key].append(row)
    for rows in pending.values():
        rows.sort(key=lambda row: row["candidate_id"])
        if len(rows) == 4:
            yield {
                "source": "DSR_Simple_training",
                "route": "think",
                "domain": rows[0].get("domain", "unknown"),
                "gold_count": int(rows[0].get("gold_count") or 0),
                "rewards": [float(row["primary_reward"]) for row in rows],
            }


def build_rows(groups: list[dict]):
    group_rows = []
    candidates = []
    for group_index, group in enumerate(groups):
        tensor = torch.tensor(group["rewards"], dtype=torch.float64)
        advantages = {
            "current": current_advantages(tensor).tolist(),
            "think_exact_clamp": think_exact_clamp_advantages(tensor).tolist(),
        }
        pattern = reward_pattern(group["rewards"])
        group_row = {
            **group,
            "group_index": group_index,
            "gold_bucket": gold_bucket(group["gold_count"]),
            "reward_pattern": pattern,
            "advantages": advantages,
            "abs_sums": {rule: sum(abs(value) for value in advantages[rule]) for rule in RULES},
        }
        group_rows.append(group_row)
        for index, reward in enumerate(group["rewards"]):
            candidates.append({
                "group_index": group_index,
                "source": group["source"],
                "domain": group["domain"],
                "gold_count": group["gold_count"],
                "gold_bucket": group_row["gold_bucket"],
                "pattern": pattern,
                "reward": reward,
                "level": reward_level(reward),
                "advantages": {rule: advantages[rule][index] for rule in RULES},
            })
    return group_rows, candidates


def candidate_breakdown(candidates: list[dict], key) -> dict:
    buckets = defaultdict(list)
    for row in candidates:
        buckets[str(key(row))].append(row)
    totals = {rule: sum(abs(row["advantages"][rule]) for row in candidates) for rule in RULES}
    output = {}
    for label, rows in sorted(buckets.items()):
        output[label] = {"candidate_count": len(rows)}
        for rule in RULES:
            values = [row["advantages"][rule] for row in rows]
            positive = [value for value in values if value > 0]
            mass = sum(abs(value) for value in values)
            output[label][rule] = {
                "mean_positive_advantage": sum(positive) / len(positive) if positive else None,
                "mean_abs_advantage": mass / len(values),
                "total_abs_advantage": mass,
                "share_of_total_abs_advantage": mass / totals[rule] if totals[rule] else None,
                "positive_rate": sum(value > 0 for value in values) / len(values),
                "negative_rate": sum(value < 0 for value in values) / len(values),
                "zero_rate": sum(value == 0 for value in values) / len(values),
            }
    return output


def pattern_breakdown(groups: list[dict]) -> dict:
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


def summarize(groups: list[dict]):
    group_rows, candidates = build_rows(groups)
    patterns = pattern_breakdown(group_rows)
    total_mass = {
        rule: sum(abs(row["advantages"][rule]) for row in candidates) for rule in RULES
    }
    bias = {
        "domain": candidate_breakdown(candidates, lambda row: row["domain"]),
        "gold_count": candidate_breakdown(candidates, lambda row: row["gold_bucket"]),
    }
    ratios = {}
    for rule in RULES:
        multi = patterns["MULTI_EXACT"][rule]["mean_group_abs_sum"]
        a_only = patterns["A_ONLY"][rule]["mean_group_abs_sum"]
        ratios[rule] = {
            "multi_exact_mean_mass_over_a_only": multi / a_only,
            "video_mass_share": bias["domain"]["video"][rule]["share_of_total_abs_advantage"],
            "gold_count_5_plus_mass_share": (
                bias["gold_count"]["5+"][rule]["share_of_total_abs_advantage"]
            ),
            "total_abs_advantage": total_mass[rule],
        }
    result = {
        "metadata": {
            "route": "think_only",
            "group_count": len(group_rows),
            "candidate_count": len(candidates),
            "advantage_mass_proxy": "sum(abs(advantage)); proxy only, not gradient norm",
            "sources": {
                "GR_REC_v1_probe": "fixed Probe rewards, not full training coverage",
                "DSR_Simple_training": "full simple_forensic Think primary rewards",
            },
        },
        "reward_levels": candidate_breakdown(candidates, lambda row: row["level"]),
        "patterns": patterns,
        "bias": bias,
        "ratios": ratios,
    }
    return result, group_rows


def write_csv(path: Path, groups: list[dict]):
    fields = [
        "source", "route", "domain", "gold_count", "reward_pattern", "rewards",
        "current_adv", "think_exact_clamp_adv", "current_abs_sum", "clamp_abs_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            writer.writerow({
                "source": group["source"], "route": "think", "domain": group["domain"],
                "gold_count": group["gold_count"], "reward_pattern": group["reward_pattern"],
                "rewards": json.dumps(group["rewards"], separators=(",", ":")),
                "current_adv": json.dumps(group["advantages"]["current"], separators=(",", ":")),
                "think_exact_clamp_adv": json.dumps(
                    group["advantages"]["think_exact_clamp"], separators=(",", ":")
                ),
                "current_abs_sum": group["abs_sums"]["current"],
                "clamp_abs_sum": group["abs_sums"]["think_exact_clamp"],
            })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-rec-probes", type=Path, default=DEFAULT_GR_REC)
    parser.add_argument("--simple-forensic", type=Path, default=DEFAULT_SIMPLE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    groups = list(load_gr_rec_think(args.gr_rec_probes))
    groups.extend(load_simple_think(args.simple_forensic))
    result, rows = summarize(groups)
    for level in ("8", ">8"):
        if result["reward_levels"][level]["think_exact_clamp"]["negative_rate"] != 0:
            raise AssertionError(f"high-quality negative advantage at reward level {level}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "think_counterfactual.json"
    csv_path = args.output_dir / "think_counterfactual_groups.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    print(json.dumps({"groups": len(rows), "json": str(json_path), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
