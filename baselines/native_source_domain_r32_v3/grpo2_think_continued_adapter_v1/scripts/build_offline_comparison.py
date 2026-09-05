#!/usr/bin/env python3
"""Aggregate the six fixed, inference-only GRPO-2 checkpoint probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continued_retention import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, help="LABEL=probes.jsonl")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for raw in args.entry:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid --entry: {raw!r}")
        values = summarize(path)
        if len(values) != 1:
            raise RuntimeError(f"expected one probe step for {label}, got {sorted(values)}")
        records.append({"model": label, "metrics": next(iter(values.values()))})

    payload = {
        "status": "PASS",
        "paired_fixed_probe": True,
        "automatic_best_selection": False,
        "models": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# GRPO-2 offline checkpoint comparison",
        "",
        "No checkpoint is automatically selected as best.",
        "",
        "| Model | Think mean reward | Think Success@K | Think Success@32 | NoThink mean reward | NoThink Success@K | NoThink positive rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        think = record["metrics"]["think"]
        no = record["metrics"]["nothink"]
        lines.append(
            f"| {record['model']} | {think['mean_reward']:.6f} | "
            f"{think['success_at_k']['value']:.6f} | {think['success_at_32']['value']:.6f} | "
            f"{no['mean_reward']:.6f} | {no['success_at_k']['value']:.6f} | "
            f"{no['positive_candidate_rate']['value']:.6f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "models": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
