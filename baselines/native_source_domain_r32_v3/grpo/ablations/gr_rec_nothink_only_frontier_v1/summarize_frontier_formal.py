#!/usr/bin/env python3
"""CPU-only forensic summary for a completed Frontier v1 formal run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

from safetensors import safe_open

STAGES = ("domain", "a", "b", "c")
REWARD_LEVELS = ("-1.0", "-0.25", "0.0", "0.5", "2.0", "8.0")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def numeric(values) -> dict:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
    }


def metric_window(rows: list[dict], start: int, end: int) -> dict:
    selected = [row for row in rows if start <= int(row["step"]) <= end]
    return {
        "steps": [start, end],
        "loss": numeric(row.get("loss") for row in selected),
        "grad_norm": numeric(row.get("grad_norm") for row in selected),
        "approx_kl": numeric(row.get("approx_kl") for row in selected),
        "clip_fraction": numeric(row.get("clip_fraction") for row in selected),
        "reward_mean": numeric(row.get("reward_mean") for row in selected),
        "reward_std": numeric(row.get("reward_std") for row in selected),
        "zero_std_ratio": numeric(row.get("zero_std_ratio") for row in selected),
    }


def adapter_delta(fresh_path: Path, trained_path: Path) -> dict:
    squared = 0.0
    max_abs = 0.0
    tensors = 0
    elements = 0
    with safe_open(str(fresh_path), framework="pt", device="cpu") as fresh, safe_open(
        str(trained_path), framework="pt", device="cpu"
    ) as trained:
        if set(fresh.keys()) != set(trained.keys()):
            raise RuntimeError("fresh and trained adapter keys differ")
        for key in fresh.keys():
            delta = trained.get_tensor(key).float() - fresh.get_tensor(key).float()
            squared += float((delta * delta).sum())
            max_abs = max(max_abs, float(delta.abs().max()))
            tensors += 1
            elements += delta.numel()
    return {
        "tensor_count": tensors,
        "element_count": elements,
        "l2": math.sqrt(squared),
        "max_abs": max_abs,
        "changed": squared > 0,
    }


def probe_summary(rows: list[dict]) -> list[dict]:
    result = []
    for step in sorted({int(row["step"]) for row in rows}):
        selected = [row for row in rows if int(row["step"]) == step]
        item = {"step": step, "group_count": len(selected)}
        for key in ("think", "nothink"):
            values = [row[key] for row in selected]
            item[key] = {
                field: numeric(value.get(field) for value in values)
                for field in (
                    "reward_mean", "reward_std", "closure_rate", "exact_count",
                    "ab_count", "a_count", "invalid_count",
                )
            }
        result.append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-adapter", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    metrics = read_jsonl(args.run_dir / "metrics.jsonl")
    rollouts = read_jsonl(args.run_dir / "rollouts.jsonl")
    probes = read_jsonl(args.run_dir / "probes.jsonl")
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((args.checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    expected_steps = int(manifest["expected_optimizer_steps"])

    frontier_counts = Counter()
    positive_groups = Counter()
    taxonomies = Counter()
    format_reasons = Counter()
    format_violations = 0
    group_count = 0
    c_by_rollout = defaultdict(int)
    # Ranks 0 and 2 own the two distinct global G8 groups; 1 and 3 mirror them.
    for rank in (0, 2):
        rows = read_jsonl(args.run_dir / "ranks" / f"rank{rank}-nothink-frontier.jsonl")
        for row in rows:
            if row.get("type") != "nothink_frontier_plan":
                continue
            group_count += 1
            rollout_id = int(row["rollout_id"])
            count = int(row["format_violation_candidate_count"])
            format_violations += count
            format_reasons.update(row.get("format_violation_reason_counts", {}))
            taxonomies[row["zero_signal_taxonomy"]] += 1
            for stage in STAGES:
                frontier = int(row[f"{stage}_frontier_negative_count"])
                frontier_counts[stage] += frontier
                c_by_rollout[rollout_id] += frontier if stage == "c" else 0
                positive_groups[stage] += int(bool(row[f"{stage}_positive_active"]))

    bridge_active_groups = 0
    bridge_weighted = []
    for rank in (0, 2):
        rows = read_jsonl(args.run_dir / "ranks" / f"rank{rank}-nothink-bridge.jsonl")
        for row in rows:
            if row.get("type") == "nothink_hierarchical_bridge_plan":
                bridge_active_groups += int(bool(row["bridge_active"]))
            elif row.get("type") == "nothink_hierarchical_bridge_loss":
                bridge_weighted.append(float(row.get("bridge_total_weighted", 0.0)))

    reward_levels = Counter()
    for row in rollouts:
        reward_levels.update({key: int(value) for key, value in row["reward_level_dist"].items()})
    grad_by_rollout = defaultdict(list)
    for row in metrics:
        grad_by_rollout[int(row["rollout_id"])].append(float(row["grad_norm"]))
    c_grad = {
        "with_c_frontier": numeric(
            max(values) for rollout_id, values in grad_by_rollout.items() if c_by_rollout[rollout_id] > 0
        ),
        "without_c_frontier": numeric(
            max(values) for rollout_id, values in grad_by_rollout.items() if c_by_rollout[rollout_id] == 0
        ),
        "max_c_frontier_candidates_in_rollout": max(c_by_rollout.values(), default=0),
    }

    trained_weights = args.checkpoint / "adapter_model.safetensors"
    lora_delta = adapter_delta(args.fresh_adapter / "adapter_model.safetensors", trained_weights)
    with safe_open(str(trained_weights), framework="pt", device="cpu") as trained:
        adapter_only = all("lora" in key.lower() for key in trained.keys())

    nonfinite = [
        {"step": row.get("step"), "field": key, "value": value}
        for row in metrics
        for key, value in row.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    error_patterns = {
        "nan_or_inf": r"(?:^|[^A-Za-z])(NaN|Inf)(?:[^A-Za-z]|$)",
        "oom": r"out of memory|CUDA OOM",
        "runtime_exception": r"Traceback|RuntimeError|ChildFailedError",
        "checkpoint_error": r"checkpoint.{0,80}(?:fail|error)|saving.{0,80}(?:fail|error)",
    }
    log_errors = {
        key: bool(re.search(pattern, log_text, flags=re.IGNORECASE | re.MULTILINE))
        for key, pattern in error_patterns.items()
    }
    checkpoints = sorted(
        int(path.name.split("-")[-1])
        for path in args.checkpoint.parent.glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit()
    )
    final_history = state.get("log_history", [{}])[-1]
    run_summaries = [
        json.loads(line)
        for line in log_text.splitlines()
        if line.startswith('{"run_id":') and '"global_step":' in line
    ]
    run_summary = run_summaries[-1] if run_summaries else {}
    payload = {
        "experiment": "GR_REC_NoThinkOnly_Frontier_v1",
        "phase": "4xA800 formal full epoch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "launch_commit": manifest["git_commit"],
        "initialization": manifest["initialization"],
        "completed_steps": int(state["global_step"]),
        "expected_steps": expected_steps,
        "rollout_count": len(rollouts),
        "g8_group_count": group_count,
        "candidate_count": group_count * 8,
        "checkpoints": checkpoints,
        "train_runtime_sec": run_summary.get("train_runtime_sec", final_history.get("train_runtime")),
        "train_loss": run_summary.get("train_loss", final_history.get("train_loss")),
        "metrics": {
            "full": metric_window(metrics, 1, expected_steps),
            "early": metric_window(metrics, 1, 250),
            "middle": metric_window(metrics, 647, 896),
            "late": metric_window(metrics, expected_steps - 249, expected_steps),
        },
        "reward_level_candidate_counts": {level: reward_levels[level] for level in REWARD_LEVELS},
        "format": {
            "violation_candidate_count": format_violations,
            "violation_candidate_rate": format_violations / (group_count * 8),
            "reason_counts": dict(format_reasons),
        },
        "frontier_candidate_counts": dict(frontier_counts),
        "frontier_candidate_rates": {
            stage: frontier_counts[stage] / (group_count * 8) for stage in STAGES
        },
        "positive_stage_active_group_counts": dict(positive_groups),
        "zero_signal_taxonomy_counts": dict(taxonomies),
        "c_frontier_vs_grad": c_grad,
        "bridge": {
            "active_group_count": bridge_active_groups,
            "active_group_rate": bridge_active_groups / group_count,
            "weighted_loss": numeric(bridge_weighted),
            "lambda": 0.02,
        },
        "fixed_probe": probe_summary(probes),
        "parameter_evidence": {
            "lora_delta": lora_delta,
            "base_delta": 0.0,
            "checkpoint_keys_are_adapter_only": adapter_only,
            "base_delta_evidence": "base frozen; optimizer LoRA-only; checkpoint adapter-only",
        },
        "safety": {
            "nonfinite_metrics": nonfinite,
            "log_errors": log_errors,
            "pass": (
                int(state["global_step"]) == expected_steps
                and len(metrics) == expected_steps
                and len(rollouts) * 2 == expected_steps
                and group_count == expected_steps
                and lora_delta["changed"]
                and adapter_only
                and not nonfinite
                and not any(log_errors.values())
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["safety"]["pass"]:
        raise RuntimeError("FRONTIER_FORMAL_FORENSIC_FAILED")


if __name__ == "__main__":
    main()
