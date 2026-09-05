#!/usr/bin/env python3
"""Fail-closed artifact checks for the deterministic SFT -> GRPO-1 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SFT_MODEL_SHA256 = "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"
SFT_CONFIG_SHA256 = "78451878177a5443d87440940c54177da6e833e8ab4c97dab0e263a4c97162e2"
GRPO_STEP = 500
GRPO_ADAPTER_SHA256 = "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"
GRPO_REQUIRED = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "lineage.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def verify_sft(output: Path) -> dict:
    required = {
        "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "trainer_state.json", "training_args.bin",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"SFT output is incomplete: {missing}")
    checkpoints = sorted(path.name for path in output.glob("checkpoint-*") if path.is_dir())
    if checkpoints:
        raise RuntimeError(f"SFT emitted unexpected intermediate checkpoints: {checkpoints}")
    model_sha = sha256(output / "model.safetensors")
    config_sha = sha256(output / "config.json")
    if model_sha != SFT_MODEL_SHA256:
        raise RuntimeError(f"SFT model SHA mismatch: {model_sha}")
    if config_sha != SFT_CONFIG_SHA256:
        raise RuntimeError(f"SFT config SHA mismatch: {config_sha}")
    state = load_json(output / "trainer_state.json")
    if int(state.get("global_step", -1)) != 475 or int(state.get("max_steps", -1)) != 475:
        raise RuntimeError("SFT trainer state must finish at global_step=max_steps=475")
    return {
        "status": "PASS",
        "stage": "sft",
        "global_step": 475,
        "model_sha256": model_sha,
        "config_sha256": config_sha,
        "retained_model_state": str(output.resolve()),
        "verified_at": utc_now(),
    }


def verify_grpo(output: Path) -> dict:
    summary_path = output / "summary.json"
    checkpoint = output / f"checkpoint-{GRPO_STEP}"
    if not summary_path.is_file():
        raise RuntimeError("GRPO summary.json is missing")
    summary = load_json(summary_path)
    if summary.get("status") != "PASS" or int(summary.get("global_step", -1)) != GRPO_STEP:
        raise RuntimeError("GRPO summary does not report PASS at step 500")
    checkpoints = sorted(path.name for path in output.glob("checkpoint-*") if path.is_dir())
    if checkpoints != [f"checkpoint-{GRPO_STEP}"]:
        raise RuntimeError(f"GRPO must retain only checkpoint-500, got {checkpoints}")
    missing = sorted(name for name in GRPO_REQUIRED if not (checkpoint / name).is_file())
    if missing:
        raise RuntimeError(f"GRPO checkpoint is incomplete: {missing}")
    forbidden = sorted(path.name for path in checkpoint.glob("model*.safetensors"))
    if forbidden:
        raise RuntimeError(f"GRPO checkpoint is not adapter-only: {forbidden}")
    state = load_json(checkpoint / "trainer_state.json")
    if int(state.get("global_step", -1)) != GRPO_STEP or int(state.get("max_steps", -1)) != GRPO_STEP:
        raise RuntimeError("GRPO trainer state must finish at global_step=max_steps=500")
    lineage = load_json(checkpoint / "lineage.json")
    if lineage.get("parent_base_sha256") != SFT_MODEL_SHA256:
        raise RuntimeError("GRPO lineage does not point to the verified SFT model")
    adapter_sha = sha256(checkpoint / "adapter_model.safetensors")
    if adapter_sha != GRPO_ADAPTER_SHA256:
        raise RuntimeError(f"GRPO adapter SHA mismatch: {adapter_sha}")
    return {
        "status": "PASS",
        "stage": "grpo1",
        "global_step": GRPO_STEP,
        "adapter_sha256": adapter_sha,
        "parent_model_sha256": lineage["parent_base_sha256"],
        "retained_model_state": str(checkpoint.resolve()),
        "verified_at": utc_now(),
    }


def final_report(run_root: Path, source_commit: str) -> dict:
    sft = load_json(run_root / "state/01_sft.PASS.json")
    grpo = load_json(run_root / "state/02_grpo1.PASS.json")
    if sft.get("status") != "PASS" or grpo.get("status") != "PASS":
        raise RuntimeError("both stage reports must be PASS")
    checkpoint_links = sorted(path.name for path in (run_root / "checkpoints").iterdir())
    if checkpoint_links != ["01_sft_final", "02_grpo1_step500"]:
        raise RuntimeError(f"exactly two retained model-state links are required: {checkpoint_links}")
    return {
        "status": "PASS",
        "contract": "DETERMINISTIC_SFT_TO_GRPO1_500",
        "source_commit": source_commit,
        "retained_model_state_count": 2,
        "retained_model_states": checkpoint_links,
        "sft": sft,
        "grpo1": grpo,
        "completed_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("verify-sft", "verify-grpo"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)
    final = sub.add_parser("final-report")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--source-commit", required=True)
    final.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-sft":
        result = verify_sft(args.output.resolve())
    elif args.command == "verify-grpo":
        result = verify_grpo(args.output.resolve())
    else:
        result = final_report(args.run_root.resolve(), args.source_commit)
    write_json(args.report.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
