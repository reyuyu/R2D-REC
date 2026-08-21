#!/usr/bin/env python3
"""CPU-only forensic summary for the bounded Frontier Smoke24 run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open

STAGES = ("domain", "a", "b", "c")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def numeric(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def adapter_delta(fresh_path, trained_path):
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
            left = fresh.get_tensor(key).float()
            right = trained.get_tensor(key).float()
            delta = right - left
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


def correlation(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


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
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((args.checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")

    frontier_by_rollout = defaultdict(lambda: {stage: 0 for stage in STAGES})
    positive_group_by_stage = Counter()
    format_violations = 0
    group_count = 0
    c_counts = {}
    # Ranks 0 and 2 are the two distinct global G8 groups; ranks 1 and 3 mirror them.
    for rank in (0, 2):
        path = args.run_dir / "ranks" / f"rank{rank}-nothink-frontier.jsonl"
        plans = [row for row in read_jsonl(path) if row.get("type") == "nothink_frontier_plan"]
        for row in plans:
            rollout_id = int(row["rollout_id"])
            group_count += 1
            format_violations += int(row["format_violation_candidate_count"])
            for stage in STAGES:
                frontier_by_rollout[rollout_id][stage] += int(
                    row[f"{stage}_frontier_negative_count"]
                )
                positive_group_by_stage[stage] += int(bool(row[f"{stage}_positive_active"]))
    for rollout_id, counts in frontier_by_rollout.items():
        c_counts[rollout_id] = counts["c"]

    bridge_active_groups = 0
    bridge_weighted = []
    for rank in (0, 2):
        path = args.run_dir / "ranks" / f"rank{rank}-nothink-bridge.jsonl"
        for row in read_jsonl(path):
            if row.get("type") == "nothink_hierarchical_bridge_plan":
                bridge_active_groups += int(bool(row["bridge_active"]))
            elif row.get("type") == "nothink_hierarchical_bridge_loss":
                bridge_weighted.append(float(row.get("bridge_total_weighted", 0.0)))

    grad_by_rollout = defaultdict(list)
    for row in metrics:
        grad_by_rollout[int(row["rollout_id"])].append(float(row["grad_norm"]))
    rollout_grad_max = {
        rollout_id: max(values) for rollout_id, values in grad_by_rollout.items()
    }
    paired = [
        {
            "rollout_id": rollout_id,
            "c_frontier_candidate_count": c_counts.get(rollout_id, 0),
            "grad_norm_max": rollout_grad_max[rollout_id],
        }
        for rollout_id in sorted(rollout_grad_max)
    ]
    xs = [row["c_frontier_candidate_count"] for row in paired]
    ys = [row["grad_norm_max"] for row in paired]
    c_positive = [row["grad_norm_max"] for row in paired if row["c_frontier_candidate_count"] > 0]
    c_zero = [row["grad_norm_max"] for row in paired if row["c_frontier_candidate_count"] == 0]

    fresh_weights = args.fresh_adapter / "adapter_model.safetensors"
    trained_weights = args.checkpoint / "adapter_model.safetensors"
    lora_delta = adapter_delta(fresh_weights, trained_weights)
    checkpoint_keys_are_adapter_only = True
    with safe_open(str(trained_weights), framework="pt", device="cpu") as trained:
        checkpoint_keys_are_adapter_only = all("lora" in key.lower() for key in trained.keys())

    nonfinite_metrics = [
        {"step": row.get("step"), "field": key, "value": value}
        for row in metrics
        for key, value in row.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    error_patterns = {
        "nan_or_inf": r"(?:^|[^A-Za-z])(NaN|Inf)(?:[^A-Za-z]|$)",
        "oom": r"out of memory|CUDA OOM",
        "runtime_exception": r"Traceback|RuntimeError|ChildFailedError",
    }
    log_errors = {
        key: bool(re.search(pattern, log_text, flags=re.IGNORECASE | re.MULTILINE))
        for key, pattern in error_patterns.items()
    }
    frontier_totals = {
        stage: sum(counts[stage] for counts in frontier_by_rollout.values())
        for stage in STAGES
    }
    payload = {
        "experiment": "GR_REC_NoThinkOnly_Frontier_v1",
        "phase": "4xA800 bounded Smoke24",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id"),
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "initialization": "fresh original BATA adapter",
        "completed_steps": int(state["global_step"]),
        "expected_steps": 24,
        "rollout_count": len(rollouts),
        "g8_group_count": group_count,
        "candidate_count": group_count * 8,
        "metrics": {
            "loss": numeric(row.get("loss") for row in metrics),
            "grad_norm": numeric(row.get("grad_norm") for row in metrics),
            "approx_kl": numeric(row.get("approx_kl") for row in metrics),
            "clip_fraction": numeric(row.get("clip_fraction") for row in metrics),
            "reward_mean": numeric(row.get("reward_mean") for row in metrics),
        },
        "format": {
            "violation_candidate_count": format_violations,
            "violation_candidate_rate": format_violations / (group_count * 8),
        },
        "frontier_candidate_counts": frontier_totals,
        "positive_stage_active_group_counts": dict(positive_group_by_stage),
        "c_frontier_vs_grad": {
            "rollouts": paired,
            "pearson_c_count_vs_rollout_max_grad": correlation(xs, ys),
            "c_positive_rollout_count": len(c_positive),
            "c_zero_rollout_count": len(c_zero),
            "c_positive_grad_norm_max": numeric(c_positive),
            "c_zero_grad_norm_max": numeric(c_zero),
            "largest_grad_rollout": max(paired, key=lambda row: row["grad_norm_max"]),
        },
        "bridge": {
            "active_group_count": bridge_active_groups,
            "weighted_loss": numeric(bridge_weighted),
            "lambda": 0.02,
        },
        "parameter_evidence": {
            "lora_delta": lora_delta,
            "base_delta": 0.0,
            "base_delta_evidence": (
                "base parameters were frozen; optimizer contained only trainable LoRA; "
                "checkpoint contains adapter-only LoRA tensors"
            ),
            "checkpoint_keys_are_adapter_only": checkpoint_keys_are_adapter_only,
        },
        "safety": {
            "nonfinite_metrics": nonfinite_metrics,
            "log_errors": log_errors,
            "nan_or_inf": bool(nonfinite_metrics) or log_errors["nan_or_inf"],
            "oom": log_errors["oom"],
            "runtime_exception": log_errors["runtime_exception"],
            "pass": (
                int(state["global_step"]) == 24
                and lora_delta["changed"]
                and checkpoint_keys_are_adapter_only
                and not nonfinite_metrics
                and not any(log_errors.values())
            ),
        },
        "startup_incident": {
            "archived_path": (
                "/data/GRPO/archives/"
                "GR-REC-NOTHINK-ONLY-FRONTIER-V1-SMOKE24-20260822-"
                "startup-nccl-failed-step0"
            ),
            "training_steps": 0,
            "cause": "missing standard loopback NCCL/GLOO interface environment",
            "excluded_from_smoke_metrics": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["safety"]["pass"]:
        raise RuntimeError("FRONTIER_SMOKE24_FORENSIC_FAILED")


if __name__ == "__main__":
    main()
