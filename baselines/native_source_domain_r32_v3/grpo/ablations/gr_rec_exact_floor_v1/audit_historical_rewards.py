"""CPU-only counterfactual audit over saved GR_REC reward groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

try:
    from .exact_floor import exact_floor_advantages
except ImportError:  # Direct script entry point.
    from exact_floor import exact_floor_advantages


DEFAULT_GR_REC = Path(
    "/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/probes.jsonl"
)
DEFAULT_SIMPLE = Path(
    "/data/GRPO/runs/GR-REC-DSR-SIMPLE-V1-FULL-E1-GATE200-20260819/simple_forensic.jsonl"
)


def gold_bucket(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 4:
        return "3-4"
    return "5+"


def reward_bucket(reward: float) -> str:
    for value, label in ((0.0, "0"), (0.5, "0.5"), (2.0, "2"), (8.0, "8")):
        if abs(reward - value) < 1e-8:
            return label
    return ">8" if reward > 8 else "other"


def reward_pattern(rewards: list[float]) -> str:
    exact = sum(abs(value - 8.0) < 1e-8 for value in rewards)
    above = sum(value > 8.0 for value in rewards)
    if all(abs(value) < 1e-8 for value in rewards):
        return "ALL_ZERO"
    if all(abs(value - 8.0) < 1e-8 for value in rewards):
        return "EXACT_SATURATED"
    if exact >= 2:
        return "MULTI_EXACT"
    if above:
        return "MIXED_HIGH"
    if exact == 1:
        return "SINGLE_EXACT"
    if any(abs(value - 2.0) < 1e-8 for value in rewards):
        return "AB_SIGNAL"
    if any(abs(value - 0.5) < 1e-8 for value in rewards):
        return "A_ONLY"
    return "OTHER"


def current_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, correction=0, keepdim=True)
    return ((grouped - mean) / (std + 1e-4)).reshape_as(rewards)


def load_gr_rec_probes(path: Path):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for route in ("think", "nothink"):
            candidates = row.get(route, {}).get("candidates") or []
            rewards = [candidate.get("reward") for candidate in candidates]
            rewards = [float(value) for value in rewards if value is not None]
            expected = 4 if route == "think" else 8
            if len(rewards) == expected:
                yield {
                    "source": "GR_REC_v1_probe",
                    "route": route,
                    "domain": row.get("target_domain", "unknown"),
                    "gold_count": len(row.get("gold_sids") or []),
                    "rewards": rewards,
                }


def load_simple_forensic(path: Path):
    pending: dict[tuple, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("type") == "nothink_group":
            yield {
                "source": "DSR_Simple_training",
                "route": "nothink",
                "domain": row.get("domain", "unknown"),
                "gold_count": int(row.get("gold_count") or 0),
                "rewards": [float(value) for value in row["candidate_rewards"]],
            }
        elif row.get("type") == "think_candidate":
            key = (row.get("step"), row.get("rollout_id"), row.get("group_id"))
            pending[key].append(row)
    for rows in pending.values():
        rows.sort(key=lambda row: row["candidate_id"])
        if len(rows) != 4:
            continue
        yield {
            "source": "DSR_Simple_training",
            "route": "think",
            "domain": rows[0].get("domain", "unknown"),
            "gold_count": int(rows[0].get("gold_count") or 0),
            "rewards": [float(row["primary_reward"]) for row in rows],
        }


def _rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def summarize(groups: list[dict]) -> dict:
    levels = defaultdict(lambda: defaultdict(lambda: {"n": 0, "abs": 0.0, "pos": 0, "neg": 0}))
    patterns = Counter()
    strata = defaultdict(lambda: {"groups": 0, "mean_gt_8": 0, "all_gt_8": 0})
    prevalence = defaultdict(lambda: {"groups": 0, "mean_gt_8": 0, "all_gt_8": 0})

    for group in groups:
        values = group["rewards"]
        tensor = torch.tensor(values, dtype=torch.float64)
        current = current_advantages(tensor, len(values))
        exact = exact_floor_advantages(tensor, len(values))
        patterns[(group["source"], group["route"], reward_pattern(values))] += 1
        mean_gt = sum(values) / len(values) > 8.0
        all_gt = all(value > 8.0 for value in values)
        stratum_keys = [
            f"source={group['source']}",
            f"source={group['source']}|route={group['route']}",
            f"source={group['source']}|domain={group['domain']}",
            f"source={group['source']}|gold={gold_bucket(group['gold_count'])}",
        ]
        for key in stratum_keys:
            bucket = strata[key]
            bucket["groups"] += 1
            bucket["mean_gt_8"] += int(mean_gt)
            bucket["all_gt_8"] += int(all_gt)
        source_prevalence = prevalence[group["source"]]
        source_prevalence["groups"] += 1
        source_prevalence["mean_gt_8"] += int(mean_gt)
        source_prevalence["all_gt_8"] += int(all_gt)

        for reward, current_value, exact_value in zip(values, current.tolist(), exact.tolist()):
            label = reward_bucket(reward)
            for rule, advantage in (("current", current_value), ("exact_floor", exact_value)):
                cell = levels[group["source"]][(label, rule)]
                cell["n"] += 1
                cell["abs"] += abs(advantage)
                cell["pos"] += int(advantage > 0)
                cell["neg"] += int(advantage < 0)

    level_output = {}
    for source, values in levels.items():
        level_output[source] = {}
        for (level, rule), cell in sorted(values.items()):
            level_output[source].setdefault(level, {})[rule] = {
                "n": cell["n"],
                "mean_abs_advantage": cell["abs"] / cell["n"],
                "positive_rate": cell["pos"] / cell["n"],
                "negative_rate": cell["neg"] / cell["n"],
            }

    for bucket in list(strata.values()) + list(prevalence.values()):
        bucket["mean_gt_8_rate"] = _rate(bucket["mean_gt_8"], bucket["groups"])
        bucket["all_gt_8_rate"] = _rate(bucket["all_gt_8"], bucket["groups"])
    return {
        "scope": {
            "GR_REC_v1_probe": "fixed-probe candidate rewards; not full training coverage",
            "DSR_Simple_training": "full saved simple_forensic primary rewards",
        },
        "groups": len(groups),
        "reward_levels": level_output,
        "patterns": {
            f"{source}|{route}|{pattern}": count
            for (source, route, pattern), count in sorted(patterns.items())
        },
        "prevalence": dict(prevalence),
        "strata": dict(strata),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-rec-probes", type=Path, default=DEFAULT_GR_REC)
    parser.add_argument("--simple-forensic", type=Path, default=DEFAULT_SIMPLE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("historical_audit.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    groups = list(load_gr_rec_probes(args.gr_rec_probes))
    groups.extend(load_simple_forensic(args.simple_forensic))
    result = summarize(groups)
    for source, levels in result["reward_levels"].items():
        exact_eight = levels.get("8", {}).get("exact_floor")
        if exact_eight and exact_eight["negative_rate"] != 0:
            raise AssertionError(f"ExactFloor P(A<0|R=8) is nonzero for {source}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": result["groups"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
