#!/usr/bin/env python3
"""Offline repeatability and historical comparison for fresh-base step553 runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch

from compare_stable_pilots import (
    adapter_pair,
    canonical_state_hash,
    effective_inner,
    effective_pair,
    load_adapter,
    load_torch_state,
    metrics_from_inner,
    module_prefixes,
    sha256_file,
)


LABELS = ("STABLE553-A", "STABLE553-B")
REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
)
PROJECTION_RE = re.compile(r"\.(q|k|v|o|gate|up|down)_proj$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
SCALAR_KEYS = ("loss", "grad_norm", "learning_rate", "epoch")


def training_scalar_rows(checkpoint: Path) -> list[dict[str, Any]]:
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return [
        {"step": int(row["step"]), **{key: row.get(key) for key in SCALAR_KEYS}}
        for row in state.get("log_history", [])
        if "step" in row and "loss" in row and int(row["step"]) <= 553
    ]


def checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete checkpoint-553: {missing}")
    trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if trainer.get("global_step") != 553 or trainer.get("max_steps") != 1106:
        raise RuntimeError("checkpoint trainer-state step/horizon mismatch")
    optimizer = load_torch_state(checkpoint / "optimizer.pt")
    scheduler = load_torch_state(checkpoint / "scheduler.pt")
    optimizer_steps: list[float] = []
    for row in optimizer.get("state", {}).values():
        step = row.get("step") if isinstance(row, dict) else None
        if torch.is_tensor(step) and step.numel() == 1:
            optimizer_steps.append(float(step.item()))
        elif isinstance(step, (int, float)):
            optimizer_steps.append(float(step))
    rng = {
        f"rank{rank}": canonical_state_hash(load_torch_state(checkpoint / f"rng_state_{rank}.pth"))
        for rank in range(4)
    }
    scalars = training_scalar_rows(checkpoint)
    return {
        "file_sha256": {name: sha256_file(checkpoint / name) for name in REQUIRED_FILES},
        "global_step": trainer.get("global_step"),
        "epoch": trainer.get("epoch"),
        "max_steps": trainer.get("max_steps"),
        "scalar_history": scalars,
        "latest_scalar": scalars[-1] if scalars else None,
        "optimizer_file_sha256": sha256_file(checkpoint / "optimizer.pt"),
        "optimizer_canonical_sha256": canonical_state_hash(optimizer),
        "optimizer_state_count": len(optimizer.get("state", {})),
        "optimizer_adam_step_min": min(optimizer_steps) if optimizer_steps else None,
        "optimizer_adam_step_max": max(optimizer_steps) if optimizer_steps else None,
        "scheduler_file_sha256": sha256_file(checkpoint / "scheduler.pt"),
        "scheduler_canonical_sha256": canonical_state_hash(scheduler),
        "scheduler_last_epoch": scheduler.get("last_epoch"),
        "rng_canonical_sha256": rng,
    }


def group_prefixes(prefixes: Iterable[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for prefix in prefixes:
        projection = PROJECTION_RE.search(prefix)
        layer = LAYER_RE.search(prefix)
        if projection is None or layer is None:
            raise RuntimeError(f"cannot classify LoRA module: {prefix}")
        result.setdefault(f"projection_{projection.group(1)}", []).append(prefix)
        result.setdefault(f"layer_{int(layer.group(1)):02d}", []).append(prefix)
    return result


def effective_pair_for_prefixes(
    left: dict[str, torch.Tensor], left_scale: float,
    right: dict[str, torch.Tensor], right_scale: float,
    prefixes: Iterable[str],
) -> dict[str, float]:
    selected = list(prefixes)
    dot = effective_inner(left, left_scale, right, right_scale, selected)
    left_norm = effective_inner(left, left_scale, left, left_scale, selected)
    right_norm = effective_inner(right, right_scale, right, right_scale, selected)
    return metrics_from_inner(dot, left_norm, right_norm)


def compare_scalar_histories(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_step = {row["step"]: row for row in left}
    right_by_step = {row["step"]: row for row in right}
    common = sorted(set(left_by_step) & set(right_by_step))
    exact = True
    first_divergence = None
    for step in common:
        if any(left_by_step[step].get(key) != right_by_step[step].get(key) for key in SCALAR_KEYS):
            exact = False
            first_divergence = step
            break
    return {
        "common_logging_step_count": len(common),
        "common_logging_steps": common,
        "exact": exact and len(left) == len(right) == len(common),
        "first_scalar_divergence_step": first_divergence,
    }


def historical_scalar_comparison(stable: list[dict[str, Any]], historical: list[dict[str, Any]]) -> dict[str, Any]:
    stable_by_step = {row["step"]: row for row in stable}
    historical_by_step = {row["step"]: row for row in historical}
    common = sorted(set(stable_by_step) & set(historical_by_step))
    loss_delta: list[float] = []
    grad_delta: list[float] = []
    first_exact = first_clear = None
    for step in common:
        current = stable_by_step[step]
        old = historical_by_step[step]
        deltas = {
            key: abs(float(current[key]) - float(old[key]))
            for key in SCALAR_KEYS
            if current.get(key) is not None and old.get(key) is not None
        }
        if deltas.get("loss") is not None:
            loss_delta.append(deltas["loss"])
        if deltas.get("grad_norm") is not None:
            grad_delta.append(deltas["grad_norm"])
        if first_exact is None and any(value != 0.0 for value in deltas.values()):
            first_exact = step
        clearly_different = (
            deltas.get("loss", 0.0) > 1e-6
            or deltas.get("grad_norm", 0.0) > 1e-6
            or deltas.get("learning_rate", 0.0) > 1e-15
            or deltas.get("epoch", 0.0) > 1e-12
        )
        if first_clear is None and clearly_different:
            first_clear = step
    return {
        "common_logging_step_count": len(common),
        "first_exact_scalar_divergence_step": first_exact,
        "first_clearly_divergent_logging_step": first_clear,
        "mean_abs_loss_delta": sum(loss_delta) / len(loss_delta) if loss_delta else None,
        "max_abs_loss_delta": max(loss_delta) if loss_delta else None,
        "mean_abs_grad_norm_delta": sum(grad_delta) / len(grad_delta) if grad_delta else None,
        "max_abs_grad_norm_delta": max(grad_delta) if grad_delta else None,
    }


def stable553_verdict(pair: dict[str, Any]) -> str:
    """Apply the repeatability gate without consulting historical similarity."""

    exact = (
        pair["adapter_file_sha_exact"]
        and pair["adapter_canonical_sha_exact"]
        and pair["raw_lora"]["exact"]
        and pair["effective_ba"]["relative_l2"] == 0.0
        and pair["optimizer_exact"]
        and pair["scheduler_exact"]
        and pair["rng_exact"]
        and pair["scalar_logs"]["exact"]
    )
    if exact:
        return "STABLE553_EXACT"
    finite = all(
        math.isfinite(pair[scope][metric])
        for scope in ("raw_lora", "effective_ba")
        for metric in ("cosine", "relative_l2")
    )
    return "STABLE553_NUMERIC" if finite else "STABLE553_FAIL"


def compare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = {label: args.root / "runs" / label / "output" / "checkpoint-553" for label in LABELS}
    states = {label: checkpoint_state(path) for label, path in checkpoints.items()}
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    scales: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        adapters[label], scales[label], metadata[label] = load_adapter(checkpoint)
    historical, historical_scale, historical_metadata = load_adapter(args.historical_553)

    raw = adapter_pair(adapters[LABELS[0]], adapters[LABELS[1]])
    effective = effective_pair(
        adapters[LABELS[0]], scales[LABELS[0]], adapters[LABELS[1]], scales[LABELS[1]]
    )
    scalars = compare_scalar_histories(
        states[LABELS[0]]["scalar_history"], states[LABELS[1]]["scalar_history"]
    )
    pair = {
        "adapter_file_sha_exact": metadata[LABELS[0]]["file_sha256"] == metadata[LABELS[1]]["file_sha256"],
        "adapter_canonical_sha_exact": metadata[LABELS[0]]["canonical_tensor_sha256"]
        == metadata[LABELS[1]]["canonical_tensor_sha256"],
        "raw_lora": raw,
        "effective_ba": effective,
        "optimizer_exact": states[LABELS[0]]["optimizer_canonical_sha256"]
        == states[LABELS[1]]["optimizer_canonical_sha256"],
        "scheduler_exact": states[LABELS[0]]["scheduler_canonical_sha256"]
        == states[LABELS[1]]["scheduler_canonical_sha256"],
        "rng_exact": states[LABELS[0]]["rng_canonical_sha256"]
        == states[LABELS[1]]["rng_canonical_sha256"],
        "scalar_logs": scalars,
    }

    groups = group_prefixes(module_prefixes(historical))
    historical_comparison = {
        "raw_lora": adapter_pair(adapters[LABELS[0]], historical),
        "effective_ba": effective_pair(
            adapters[LABELS[0]], scales[LABELS[0]], historical, historical_scale
        ),
        "projections": {
            key.removeprefix("projection_"): effective_pair_for_prefixes(
                adapters[LABELS[0]], scales[LABELS[0]], historical, historical_scale, prefixes
            )
            for key, prefixes in groups.items()
            if key.startswith("projection_")
        },
        "layers": {
            key: effective_pair_for_prefixes(
                adapters[LABELS[0]], scales[LABELS[0]], historical, historical_scale, prefixes
            )
            for key, prefixes in groups.items()
            if key.startswith("layer_")
        },
        "scalar_trajectory": historical_scalar_comparison(
            states[LABELS[0]]["scalar_history"], training_scalar_rows(args.historical_553)
        ),
    }

    verdict = stable553_verdict(pair)
    exact = verdict == "STABLE553_EXACT"
    return {
        "verdict": verdict,
        "first_epoch_repeatability": "CONFIRMED" if exact else "NOT_CONFIRMED",
        "recipe": "BATA-STABLE-V0",
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "historical_checkpoint": {"path": str(args.historical_553), **historical_metadata},
        "adapters": metadata,
        "states": states,
        "pairwise_repeatability": pair,
        "historical553_comparison": historical_comparison,
        "external_evaluation": {"status": "NOT_EXECUTED"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-553", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
