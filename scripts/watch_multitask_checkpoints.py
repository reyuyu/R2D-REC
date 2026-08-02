#!/usr/bin/env python3
"""Evaluate completed LoRA checkpoints on a fixed 2% development subset.

Run this separately from training, ideally on an otherwise idle GPU.  It never
changes the training process; it merely notices a completed checkpoint, writes
generation records and appends a compact SID metric record for plotting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def completed_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        required = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")
        if path.is_dir() and all((path / name).is_file() for name in required):
            checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_step)


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, monitor_dir: Path) -> dict:
    step = checkpoint_step(checkpoint)
    predictions = monitor_dir / f"predictions_step_{step}.jsonl"
    metrics = monitor_dir / f"eval_step_{step}.json"
    generate_cmd = [
        sys.executable,
        str(args.generator),
        "--model",
        str(args.model),
        "--adapter",
        str(checkpoint),
        "--dev-dir",
        str(args.dev_dir),
        "--output",
        str(predictions),
        "--max-samples-per-task",
        str(args.max_samples_per_task),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    subprocess.run(generate_cmd, check=True)
    subprocess.run(
        [sys.executable, str(args.evaluator), "--predictions", str(predictions), "--output", str(metrics)],
        check=True,
    )
    result = json.loads(metrics.read_text(encoding="utf-8"))
    result["macro_step"] = step
    result["checkpoint"] = str(checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, default=Path("/data/lf_data_splits"))
    parser.add_argument("--generator", type=Path, default=Path(__file__).with_name("generate_multitask_predictions.py"))
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("evaluate_sid_action_predictions.py"))
    parser.add_argument("--max-samples-per-task", type=int, default=128, help="0 means all 2% dev examples.")
    parser.add_argument("--max-prompt-tokens", type=int, default=7168)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Evaluate currently complete checkpoints, then exit.")
    args = parser.parse_args()

    monitor_dir = args.output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    state_path = monitor_dir / "checkpoint_eval_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": []}
    completed = set(state.get("completed", []))
    metrics_path = monitor_dir / "eval_metrics.jsonl"

    while True:
        for checkpoint in completed_checkpoints(args.output_dir):
            step = checkpoint_step(checkpoint)
            if step in completed:
                continue
            print(f"evaluating macro-step {step}: {checkpoint}", flush=True)
            result = evaluate_checkpoint(args, checkpoint, monitor_dir)
            with metrics_path.open("a", encoding="utf-8") as writer:
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed.add(step)
            state_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2) + "\n", encoding="utf-8")
            print(f"completed macro-step {step}", flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
