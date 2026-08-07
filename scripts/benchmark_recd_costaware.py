#!/usr/bin/env python3
"""Compare recC baseline and recD cost-aware smoke monitor JSONL files."""
import argparse
import json
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(label, values):
    ordered = sorted(values)
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": p90,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("cost_aware", type=Path)
    args = parser.parse_args()
    baseline, cost = rows(args.baseline), rows(args.cost_aware)
    for label, data in (("baseline", baseline), ("cost_aware", cost)):
        seconds = [row["macro_seconds"] for row in data]
        tokens = [row["pack_tokens_mean"] * 8 for row in data]
        segments = [row["pack_segments_mean"] * 8 for row in data]
        print(label, "macro_seconds", summarize(label, seconds))
        print(label, "global_tokens_per_second", statistics.mean(t / s for t, s in zip(tokens, seconds)))
        print(label, "global_segments_per_second", statistics.mean(n / s for n, s in zip(segments, seconds)))
        print(label, "pack_utilization_mean", statistics.mean(row["pack_utilization_mean"] for row in data))
    base_mean = statistics.mean(row["macro_seconds"] for row in baseline)
    cost_mean = statistics.mean(row["macro_seconds"] for row in cost)
    print("runtime_delta_pct", (cost_mean / base_mean - 1.0) * 100.0)
    print("first_common_allocations_equal", all(
        left.get("current_macro_allocation") == right.get("current_macro_allocation")
        for left, right in zip(baseline, cost[:len(baseline)])
    ))
    gaps = [row["rank_cost_gap_ratio"] for row in cost]
    token_gaps = [row["rank_token_gap_ratio"] for row in cost]
    print("cost_gap_ratio", summarize("cost_gap_ratio", gaps))
    print("token_gap_ratio", summarize("token_gap_ratio", token_gaps))
    print("partition_changed", sum(int(row.get("partition_changed", 0)) for row in cost), "/", len(cost))


if __name__ == "__main__":
    main()
