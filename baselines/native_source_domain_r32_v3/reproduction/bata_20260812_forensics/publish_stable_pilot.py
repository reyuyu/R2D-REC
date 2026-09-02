#!/usr/bin/env python3
"""Publish path-free BATA-STABLE-V0 JSON and Markdown evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


LABELS = ("STABLE560-A", "STABLE560-B", "STABLE560-C")


def layer_summary(layers: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics = ("hist_update_cosine", "hist_progress", "hist_residual")
    return {
        metric: {
            "min": min(row[metric] for row in layers.values()),
            "mean": sum(row[metric] for row in layers.values()) / len(layers),
            "max": max(row[metric] for row in layers.values()),
        }
        for metric in metrics
    }


def public_result(private: dict[str, Any], manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if private.get("verdict") not in {"STABLE560_EXACT", "STABLE560_NUMERIC", "STABLE560_FAIL"}:
        raise RuntimeError("stable result has no publishable verdict")
    result = copy.deepcopy(private)
    result.pop("checkpoints", None)
    for checkpoint in result["historical_checkpoints"].values():
        checkpoint.pop("path", None)
    result["contract"] = {
        "historical_checkpoint_sha256": manifest["historical_checkpoint_sha256"],
        "pristine_source_sha256": manifest["pristine_source_sha256"],
        "runtime_source_sha256": manifest["runtime_source_sha256"],
        "llamafactory_contract": manifest["llamafactory_contract"],
        "tokenized_path_recorded_server_side": bool(manifest.get("tokenized_path")),
    }
    runtime: dict[str, Any] = {}
    for label in LABELS:
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "runs" / label / "evidence").glob("runtime_rank*.json"))
        ]
        if len(rows) != 4 or any(row.get("status") != "PASS" for row in rows):
            raise RuntimeError(f"{label} runtime evidence is incomplete")
        first = rows[0]
        nvidia_lines = (first.get("nvidia_smi") or "").splitlines()
        runtime[label] = {
            "status": "PASS",
            "rank_count": len(rows),
            "completed_global_step": first["completed_global_step"],
            "wall_seconds_rank_min": min(row["wall_seconds"] for row in rows),
            "wall_seconds_rank_max": max(row["wall_seconds"] for row in rows),
            "python": first.get("python"),
            "torch": first.get("torch"),
            "cuda": first.get("cuda"),
            "cudnn": first.get("cudnn"),
            "transformers": first.get("transformers"),
            "flash_attn": first.get("flash_attn"),
            "liger_kernel": first.get("liger_kernel"),
            "gpu_models": sorted({line.split(",")[1].strip() for line in nvidia_lines if "," in line}),
            "driver_versions": sorted({line.split(",")[-1].strip() for line in nvidia_lines if "," in line}),
            "environment": first.get("environment"),
            "torch_flags": first.get("torch_flags"),
        }
    result["runtime"] = runtime
    result["layer_summaries"] = {
        label: layer_summary(result["historical_direction"][label]["layers"])
        for label in LABELS
    }
    result["scope"] = {
        "start_step": 553,
        "stop_step": 560,
        "scheduler_horizon": 1106,
        "external_evaluation_executed": False,
        "continued_to_step1106": False,
        "heavy_forensic_instrumentation_enabled": False,
    }
    serialized = json.dumps(result, sort_keys=True)
    forbidden = ("/root/", "/data/", "tokenized_path\": \"/")
    if any(value in serialized for value in forbidden):
        raise RuntimeError("public result contains a server path")
    return result


def markdown(result: dict[str, Any]) -> str:
    scalar = result["states"]["STABLE560-A"]["scalar_step560"]
    historical = result["historical_scalar_step560"]
    direction = result["historical_direction"]["STABLE560-A"]
    runtime = result["runtime"]
    lines = [
        "# BATA-STABLE-V0 checkpoint-553 to 560",
        "",
        "## Verdict",
        "",
        f"`{result['verdict']}`",
        "",
        "Three independent four-A800 continuations produced exact raw LoRA, full",
        "effective B@A, optimizer, scheduler, and four-rank RNG state at step560.",
        "The pilot retained the original 1106-step scheduler horizon and stopped",
        "through a save-and-stop callback. No external evaluation was executed.",
        "",
        "## Stable recipe",
        "",
        "- Recovered 06:33 BATA source and original packed cache.",
        "- Historical checkpoint-553, four A800 ranks, GA16, 8K neat packing.",
        "- FA2 and Liger enabled; bf16 and pure_bf16 enabled.",
        "- Original AdamW, cosine schedule, LR, warmup, and weight decay; horizon 1106.",
        "- Strict deterministic algorithms, CUBLAS workspace contract, and",
        "  deterministic FlashAttention backward.",
        "- No DDP hook, gradient hashing, gradient reference, FA2 monkey patch, or",
        "  extra forward/backward.",
        "",
        "## Runs",
        "",
        "| Run | Step | Loss | Grad norm | LR | Runtime sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = result["states"][label]["scalar_step560"]
        lines.append(
            f"| {label} | 560 | {row['loss']:.12f} | {row['grad_norm']:.12f} | "
            f"{row['learning_rate']:.15g} | {runtime[label]['wall_seconds_rank_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise stability",
            "",
            "Every A/B, A/C, and B/C comparison has raw-LoRA cosine 1, relative L2",
            "0, effective-B@A cosine 1, relative L2 0, and exact optimizer, scheduler,",
            "and RNG fingerprints. Adam step is 560 for every state entry.",
            "",
            "## Historical trajectory observation",
            "",
            "| Scope | Update cosine | Progress | Residual |",
            "|---|---:|---:|---:|",
            f"| All modules | {direction['aggregate']['hist_update_cosine']:.9f} | "
            f"{direction['aggregate']['hist_progress']:.9f} | {direction['aggregate']['hist_residual']:.9f} |",
            f"| Attention | {direction['attention']['hist_update_cosine']:.9f} | "
            f"{direction['attention']['hist_progress']:.9f} | {direction['attention']['hist_residual']:.9f} |",
            f"| MLP | {direction['mlp']['hist_update_cosine']:.9f} | "
            f"{direction['mlp']['hist_progress']:.9f} | {direction['mlp']['hist_residual']:.9f} |",
            "",
            f"Stable step560 loss/grad norm are {scalar['loss']:.12f} / {scalar['grad_norm']:.12f}; "
            f"historical values are {historical['loss']:.12f} / {historical['grad_norm']:.12f}.",
            "These direction and scalar comparisons are observations only, not the",
            "repeatability gate and not evidence of external competition score.",
            "",
            "## Scope",
            "",
            "This result authorizes review of a possible checkpoint-553 to 1106",
            "continuation. It does not itself authorize or execute that continuation.",
            "No dataset, token value, weight, checkpoint, optimizer/RNG byte, server",
            "path, or credential is included in the public artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private-result", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    private = json.loads(args.private_result.read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    result = public_result(private, manifest, args.root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": "PASS", "verdict": result["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
