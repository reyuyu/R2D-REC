#!/usr/bin/env python3
"""Publish path-free BATA-STABLE-V0 base-to-step553 evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


LABELS = ("STABLE553-A", "STABLE553-B")


def metric_summary(rows: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    values = [row[metric] for row in rows.values()]
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def public_result(private: dict[str, Any], manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if private.get("verdict") not in {"STABLE553_EXACT", "STABLE553_NUMERIC", "STABLE553_FAIL"}:
        raise RuntimeError("stable553 result has no publishable verdict")
    result = copy.deepcopy(private)
    result.pop("checkpoints", None)
    result["historical_checkpoint"].pop("path", None)
    result["contract"] = {
        "scope": manifest["scope"],
        "pristine_source_sha256": manifest["pristine_source_sha256"],
        "runtime_source_sha256": manifest["runtime_source_sha256"],
        "base_model_sha256": manifest["base_model_sha256"],
        "base_sha_manifest_sha256": manifest["base_sha_manifest_sha256"],
        "dataset_contract": manifest["dataset_contract"],
        "cache_contract": manifest["cache_contract"],
        "llamafactory_contract": manifest["llamafactory_contract"],
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
            "rank_count": 4,
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
    layers = result["historical553_comparison"]["layers"]
    result["historical553_comparison"]["layer_summary"] = {
        metric: metric_summary(layers, metric) for metric in ("cosine", "relative_l2", "difference_l2")
    }
    result["scope"] = {
        "start_mode": "fresh_base",
        "start_step": 0,
        "stop_step": 553,
        "scheduler_horizon": 1106,
        "continued_to_step1106": False,
        "external_evaluation_executed": False,
        "heavy_forensic_instrumentation_enabled": False,
    }
    serialized = json.dumps(result, sort_keys=True)
    if any(value in serialized for value in ("/root/", "/data/", "private-uuid")):
        raise RuntimeError("public stable553 result contains a server path or private identifier")
    return result


def markdown(result: dict[str, Any]) -> str:
    pair = result["pairwise_repeatability"]
    historical = result["historical553_comparison"]
    lines = [
        "# BATA-STABLE-V0 base to checkpoint-553",
        "",
        "## Verdict",
        "",
        f"`{result['verdict']}`",
        "",
        f"`FIRST_EPOCH_REPEATABILITY = {result['first_epoch_repeatability']}`",
        "",
        "Two independent four-A800 runs started from the same SHA-pinned base model,",
        "retained the original 1106-step scheduler horizon, and used a callback to",
        "save and stop at optimizer step553. No second-epoch continuation or external",
        "evaluation was executed.",
        "",
        "## Recipe",
        "",
        "- Recovered 2026-08-12 06:33 BATA source and historical packed cache.",
        "- 222,001 source segments and 35,380 packed rows; four A800 ranks.",
        "- Micro batch 1, GA16, global batch 64, 8K neat packing.",
        "- FA2 and Liger enabled; bf16 and pure_bf16 enabled.",
        "- LoRA r32/alpha64/dropout0.05/all; AdamW LR 2e-4, cosine, warmup 0.03.",
        "- Seed 20260806, weight decay 0.01, fractional GC 0.4.",
        "- Strict PyTorch determinism, deterministic FA2 backward, and fixed CUBLAS workspace.",
        "- No resume checkpoint and no forensic gradient/DDP instrumentation.",
        "",
        "## Runs",
        "",
        "| Run | Step | Latest logged step | Loss | Grad norm | LR | Epoch | Runtime sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        state = result["states"][label]
        scalar = state["latest_scalar"]
        runtime = result["runtime"][label]["wall_seconds_rank_max"]
        lines.append(
            f"| {label} | {state['global_step']} | {scalar['step']} | {scalar['loss']:.12f} | "
            f"{scalar['grad_norm']:.12f} | {scalar['learning_rate']:.15g} | "
            f"{scalar['epoch']:.12f} | {runtime:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Repeatability",
            "",
            "| Evidence | Result |",
            "|---|---|",
            f"| Adapter file SHA | {'exact' if pair['adapter_file_sha_exact'] else 'different'} |",
            f"| Canonical LoRA SHA | {'exact' if pair['adapter_canonical_sha_exact'] else 'different'} |",
            f"| Raw LoRA cosine / relative L2 | {pair['raw_lora']['cosine']:.12g} / {pair['raw_lora']['relative_l2']:.12g} |",
            f"| Effective B@A cosine / relative L2 | {pair['effective_ba']['cosine']:.12g} / {pair['effective_ba']['relative_l2']:.12g} |",
            f"| Optimizer | {'exact' if pair['optimizer_exact'] else 'different'} |",
            f"| Scheduler | {'exact' if pair['scheduler_exact'] else 'different'} |",
            f"| Four-rank RNG | {'exact' if pair['rng_exact'] else 'different'} |",
            f"| Shared scalar logs | {'exact' if pair['scalar_logs']['exact'] else 'different'} |",
            "",
            "## Historical checkpoint-553 comparison",
            "",
            f"Raw LoRA cosine / relative L2: {historical['raw_lora']['cosine']:.9f} / "
            f"{historical['raw_lora']['relative_l2']:.9f}.",
            "",
            f"Effective B@A cosine / relative L2: {historical['effective_ba']['cosine']:.9f} / "
            f"{historical['effective_ba']['relative_l2']:.9f}.",
            "",
            "| Projection | B@A cosine | Relative L2 |",
            "|---|---:|---:|",
        ]
    )
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        row = historical["projections"][name]
        lines.append(f"| {name} | {row['cosine']:.9f} | {row['relative_l2']:.9f} |")
    trajectory = historical["scalar_trajectory"]
    layer = historical["layer_summary"]
    lines.extend(
        [
            "",
            f"Layerwise B@A cosine min/mean/max: {layer['cosine']['min']:.9f} / "
            f"{layer['cosine']['mean']:.9f} / {layer['cosine']['max']:.9f}.",
            "",
            f"Historical scalar comparison uses {trajectory['common_logging_step_count']} common logging steps. "
            f"Mean/max absolute loss delta: {trajectory['mean_abs_loss_delta']:.9f} / "
            f"{trajectory['max_abs_loss_delta']:.9f}; mean/max grad-norm delta: "
            f"{trajectory['mean_abs_grad_norm_delta']:.9f} / {trajectory['max_abs_grad_norm_delta']:.9f}. "
            f"First clearly divergent step: {trajectory['first_clearly_divergent_logging_step']}.",
            "",
            "Historical similarity is observational and does not affect the repeatability verdict.",
            "",
            "## Evaluation",
            "",
            "`NOT_EXECUTED`",
            "",
            "No reliable historical checkpoint-553 result under a separately proven identical",
            "external-evaluator contract was used in this phase.",
            "",
            "## Scope",
            "",
            "The runs stopped at checkpoint-553. They did not continue to step1106 and did",
            "not start GR_REC, GRPO-TK, or MC_USER. Public artifacts contain no dataset rows,",
            "tokens, model/optimizer/RNG bytes, raw examples, credentials, or server paths.",
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
