#!/usr/bin/env python3
"""Render task-loss and packing charts from a multitask monitor JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = ("material", "user", "recommendation", "world")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("No metric records found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["macro_step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for task in TASKS:
        axes[0].plot(steps, [row.get(f"{task}_loss", float("nan")) for row in rows], label=task)
        axes[1].plot(
            steps,
            [row.get(f"{task}_avg_packed_tokens", float("nan")) for row in rows],
            label=task,
        )
    axes[0].set_ylabel("mean microbatch loss")
    axes[0].set_title("Task losses")
    axes[1].set_ylabel("avg packed tokens")
    axes[1].set_xlabel("macro-step")
    axes[1].set_title("Packing length")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "multitask_training_curves.png", dpi=160)


if __name__ == "__main__":
    main()
