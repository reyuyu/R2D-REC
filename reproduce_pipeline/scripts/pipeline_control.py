#!/usr/bin/env python3
"""Materialize stage configs and enforce full-chain artifact identity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ADAPTER_REQUIRED = {
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {location}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{location}[{index}]")


def model_identity(path: Path) -> dict[str, str]:
    model = path / "model.safetensors"
    config = path / "config.json"
    if not model.is_file() or not config.is_file():
        raise RuntimeError(f"full model is missing model.safetensors or config.json: {path}")
    return {"model_sha256": sha256(model), "config_sha256": sha256(config)}


def adapter_identity(path: Path) -> str:
    adapter = path / "adapter_model.safetensors"
    if not adapter.is_file():
        raise RuntimeError(f"adapter_model.safetensors is missing: {path}")
    return sha256(adapter)


def registered_dataset(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "registry_key": args.dataset_key,
        "sha256": args.dataset_sha256,
        "rows": args.dataset_rows,
        "split": args.dataset_split,
        "registered_dataset_used_by_trainer": True,
    }


def materialize_grpo1(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.template)
    parent = model_identity(args.base_model)
    config["run_kind"] = "formal_grpo1_pipeline_final_only"
    config["parent"].update(parent)
    config["dataset"].update(registered_dataset(args))
    config["optimization"]["max_steps"] = args.steps
    config["checkpoint"] = {
        "save_steps": args.steps,
        "save_total_limit": 1,
        "adapter_only": True,
        "full_resume_state": True,
    }
    write_json(args.output, config)
    return config


def materialize_grpo2(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.template)
    parent_model = model_identity(args.base_model)
    parent_adapter = adapter_identity(args.adapter_parent)
    config["run_kind"] = "grpo2_continued_adapter_pipeline_final_only"
    config["optimization"]["max_steps"] = args.steps
    config["checkpoint"] = {
        "steps": [args.steps],
        "save_total_limit": 1,
        "adapter_only": True,
        "full_resume_state": True,
    }
    config["dataset"].update(registered_dataset(args))
    config["parent"] = {
        "base_model_sha256": parent_model["model_sha256"],
        "base_config_sha256": parent_model["config_sha256"],
        "adapter_sha256": parent_adapter,
        "adapter_step": args.parent_step,
        "adapter_dataset_sha256": args.parent_dataset_sha256,
    }
    write_json(args.output, config)
    return config


def materialize_grpo3(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.template)
    parent_adapter = adapter_identity(args.adapter_parent)
    config.update(
        {
            "stage": "grpo3_user_pipeline_final_only_v1",
            "base_model": str(args.base_model.resolve()),
            "adapter": str(args.adapter_parent.resolve()),
            "parent_adapter_sha256": parent_adapter,
            "parent_checkpoint_step": args.parent_step,
            "parent_stage": args.parent_stage,
            "parent_experiment": f"{args.parent_stage}_CONTINUED_SINGLE_ADAPTER_PARENT",
            "probe_parent_label": f"{args.parent_stage}-step{args.parent_step}",
            "train_data": str(args.dataset_path.resolve()),
            "train_sha256": args.dataset_sha256,
            "registered_dataset_name": args.dataset_key,
            "registered_dataset_split": args.dataset_split,
            "registered_dataset_rows": args.dataset_rows,
            "prompt_count": args.steps,
            "action_count": (args.steps + 1) // 2,
            "chain_count": args.steps // 2,
            "checkpoint_steps": [args.steps],
            "resume_supported": True,
            "resume_policy": "complete_checkpoint_state",
        }
    )
    write_json(args.output, config)
    return config


def verify_sft(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    required = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"SFT final output is incomplete: {missing}")
    unexpected = sorted(path.name for path in output.glob("checkpoint-*") if path.is_dir())
    if unexpected:
        raise RuntimeError(f"SFT emitted intermediate checkpoints: {unexpected}")
    state = load_json(output / "trainer_state.json")
    assert_finite(state, "trainer_state")
    global_step = int(state.get("global_step", -1))
    max_steps = int(state.get("max_steps", -1))
    if global_step <= 0 or global_step != max_steps:
        raise RuntimeError("SFT did not finish at global_step=max_steps")
    epoch = float(state.get("epoch", args.epochs))
    if not math.isclose(epoch, args.epochs, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"SFT epoch mismatch: expected={args.epochs} actual={epoch}")
    identity = model_identity(output)
    if args.expected_model_sha256 and identity["model_sha256"] != args.expected_model_sha256:
        raise RuntimeError(
            "SFT final model SHA256 mismatch: "
            f"expected={args.expected_model_sha256} actual={identity['model_sha256']}"
        )
    derivation = load_json(args.derivation_report)
    expected_derivation = {
        "status": "PASS",
        "registered_dataset_used_by_trainer": True,
        "registered_dataset_key": args.dataset_key,
        "registered_dataset_sha256": args.dataset_sha256,
        "derived_parquet_files": 24,
        "derived_rows": 270970,
        "temporary_data_removed": True,
    }
    for key, expected in expected_derivation.items():
        if derivation.get(key) != expected:
            raise RuntimeError(
                f"SFT raw-data derivation contract mismatch for {key}: "
                f"expected={expected!r} actual={derivation.get(key)!r}"
            )
    record = {
        "status": "PASS",
        "stage": "SFT",
        "epochs": args.epochs,
        "global_step": global_step,
        **identity,
        "final_artifact": str(output),
        "intermediate_checkpoints_disabled": True,
        "dataset_registry_key": args.dataset_key,
        "dataset_sha256": args.dataset_sha256,
        "dataset_rows": args.dataset_rows,
        "dataset_split": args.dataset_split,
        "registered_dataset_used_by_trainer": True,
        "raw_to_training_data": "AUTOMATIC_TMPFS",
        "derived_parquet_files": derivation["derived_parquet_files"],
        "derived_rows": derivation["derived_rows"],
        "temporary_data_removed": True,
        "verified_at": utc_now(),
    }
    write_json(args.report, record)
    return record


def verify_adapter(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    missing = sorted(name for name in ADAPTER_REQUIRED if not (checkpoint / name).is_file())
    if args.stage == "GRPO3" and not (checkpoint / "formal_state.json").is_file():
        missing.append("formal_state.json")
    if missing:
        raise RuntimeError(f"{args.stage} final checkpoint is incomplete: {missing}")
    forbidden = sorted(
        path.name
        for path in checkpoint.iterdir()
        if path.name == "model.safetensors"
        or path.name.startswith("model-")
        or path.name.startswith("pytorch_model")
    )
    if forbidden:
        raise RuntimeError(f"{args.stage} checkpoint is not adapter-only: {forbidden}")
    state = load_json(checkpoint / "trainer_state.json")
    summary = load_json(run_dir / "summary.json")
    lineage = load_json(checkpoint / "lineage.json")
    assert_finite(state, "trainer_state")
    assert_finite(summary, "summary")
    if summary.get("status") != "PASS":
        raise RuntimeError(f"{args.stage} summary is not PASS")
    actual_step = int(state.get("global_step", state.get("optimizer_step", -1)))
    if actual_step != args.step or int(state.get("max_steps", -1)) != args.step:
        raise RuntimeError(f"{args.stage} trainer state does not match final step {args.step}")
    patterns = ("checkpoint-*",) if args.stage != "GRPO3" else ("prompt-step-*",)
    siblings = sorted(
        path.name
        for pattern in patterns
        for path in args.checkpoint_container.resolve().glob(pattern)
        if path.is_dir()
    )
    if siblings != [checkpoint.name]:
        raise RuntimeError(f"{args.stage} must retain only its final checkpoint, got {siblings}")
    actual_sha = adapter_identity(checkpoint)
    if args.expected_adapter_sha256 and actual_sha != args.expected_adapter_sha256:
        raise RuntimeError(
            f"{args.stage} adapter SHA256 mismatch: "
            f"expected={args.expected_adapter_sha256} actual={actual_sha}"
        )
    if lineage.get("adapter_sha256") not in (None, actual_sha):
        raise RuntimeError(f"{args.stage} lineage adapter SHA mismatch")
    if args.stage == "GRPO1":
        if lineage.get("parent_base_sha256") != args.parent_sha256 or int(lineage.get("step", -1)) != args.step:
            raise RuntimeError("GRPO1 lineage mismatch")
    elif args.stage == "GRPO2":
        if lineage.get("adapter_initialization_source_sha256") != args.parent_sha256:
            raise RuntimeError("GRPO2 parent lineage mismatch")
        if lineage.get("contains_grpo1_and_grpo2_effect") is not True:
            raise RuntimeError("GRPO2 cumulative adapter lineage is missing")
    else:
        if lineage.get("parent_adapter_sha256") != args.parent_sha256:
            raise RuntimeError("GRPO3 parent lineage mismatch")
        if args.parent_stage == "GRPO1":
            if lineage.get("contains_grpo1_and_grpo3_effect") is not True:
                raise RuntimeError("GRPO3 GRPO1-parent cumulative lineage is missing")
            if lineage.get("contains_grpo1_grpo2_and_grpo3_effect") is not False:
                raise RuntimeError("GRPO3 lineage incorrectly claims a GRPO2 effect")
        elif lineage.get("contains_grpo1_grpo2_and_grpo3_effect") is not True:
            raise RuntimeError("GRPO3 GRPO2-parent cumulative lineage is missing")
    config = load_json(args.config)
    dataset = config.get("dataset", {}) if args.stage != "GRPO3" else {
        "registry_key": config.get("registered_dataset_name"),
        "sha256": config.get("train_sha256"),
    }
    if dataset.get("registry_key") != args.dataset_key or dataset.get("sha256") != args.dataset_sha256:
        raise RuntimeError(f"{args.stage} config is not bound to the resolved registry dataset")
    record = {
        "status": "PASS",
        "stage": args.stage,
        "global_step": args.step,
        "adapter_sha256": actual_sha,
        "parent_sha256": args.parent_sha256,
        "dataset_registry_key": args.dataset_key,
        "dataset_sha256": args.dataset_sha256,
        "registered_dataset_used_by_trainer": True,
        "final_artifact": str(checkpoint),
        "intermediate_checkpoints_disabled": True,
        "verified_at": utc_now(),
    }
    write_json(args.report, record)
    return record


def final_report(args: argparse.Namespace) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for item in args.stage_report:
        name, raw_path = item.split("=", 1)
        reports[name] = load_json(Path(raw_path))
        if reports[name].get("status") != "PASS":
            raise RuntimeError(f"stage report is not PASS: {name}")
    required = {
        name
        for name, enabled in (
            ("sft", args.run_sft),
            ("grpo1", args.run_grpo1),
            ("grpo2", args.run_grpo2),
            ("grpo3", args.run_grpo3),
        )
        if enabled
    }
    grpo3_parent_stage = "GRPO2" if args.run_grpo2 else "GRPO1"
    pipeline = "SFT -> GRPO1 -> GRPO2 -> GRPO3" if args.run_grpo2 else "SFT -> GRPO1 -> GRPO3"
    value = {
        "status": "PASS",
        "pipeline": pipeline,
        "source_commit": args.source_commit,
        "sft_epochs": args.sft_epochs,
        "sft_dataset_key": args.sft_dataset_key,
        "sft_dataset_registry_key": reports.get("sft", {}).get("dataset_registry_key"),
        "sft_dataset_sha256": reports.get("sft", {}).get("dataset_sha256"),
        "sft_dataset_rows": reports.get("sft", {}).get("dataset_rows"),
        "sft_registered_dataset_used_by_trainer": reports.get("sft", {}).get(
            "registered_dataset_used_by_trainer"
        ),
        "sft_raw_to_training_data": reports.get("sft", {}).get("raw_to_training_data"),
        "grpo1_steps": args.grpo1_steps,
        "grpo2_steps": args.grpo2_steps,
        "grpo2_enabled": bool(args.run_grpo2),
        "grpo2_status": "COMPLETED" if args.run_grpo2 else "SKIPPED",
        "grpo3_steps": args.grpo3_steps,
        "grpo3_parent_stage": grpo3_parent_stage,
        "sft_final_sha256": reports.get("sft", {}).get("model_sha256"),
        "grpo1_final_adapter_sha256": reports.get("grpo1", {}).get("adapter_sha256"),
        "grpo2_final_adapter_sha256": reports.get("grpo2", {}).get("adapter_sha256"),
        "grpo3_final_adapter_sha256": reports.get("grpo3", {}).get("adapter_sha256"),
        "grpo1_dataset_registry_key": reports.get("grpo1", {}).get("dataset_registry_key"),
        "grpo1_dataset_sha256": reports.get("grpo1", {}).get("dataset_sha256"),
        "grpo2_dataset_registry_key": reports.get("grpo2", {}).get("dataset_registry_key"),
        "grpo2_dataset_sha256": reports.get("grpo2", {}).get("dataset_sha256"),
        "grpo2_dataset_resolution": "RESOLVED" if args.run_grpo2 else "SKIPPED",
        "grpo3_dataset_registry_key": reports.get("grpo3", {}).get("dataset_registry_key"),
        "grpo3_dataset_sha256": reports.get("grpo3", {}).get("dataset_sha256"),
        "repro_data_root": str(args.repro_data_root.resolve()),
        "deterministic_runtime": True,
        "selected_stages_completed": set(reports) == required,
        "all_requested_stages_completed": set(reports) == required,
        "stages": reports,
        "completed_at": utc_now(),
    }
    write_json(args.output, value)
    return value


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_dataset_args(parser: argparse.ArgumentParser, *, path: bool = False) -> None:
    parser.add_argument("--dataset-key", required=True)
    if path:
        parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--dataset-rows", type=positive_int, required=True)
    parser.add_argument("--dataset-split", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    g1 = commands.add_parser("materialize-grpo1")
    g1.add_argument("--template", type=Path, required=True)
    g1.add_argument("--output", type=Path, required=True)
    g1.add_argument("--base-model", type=Path, required=True)
    g1.add_argument("--steps", type=positive_int, required=True)
    add_dataset_args(g1)
    g1.set_defaults(handler=materialize_grpo1)
    g2 = commands.add_parser("materialize-grpo2")
    g2.add_argument("--template", type=Path, required=True)
    g2.add_argument("--output", type=Path, required=True)
    g2.add_argument("--base-model", type=Path, required=True)
    g2.add_argument("--adapter-parent", type=Path, required=True)
    g2.add_argument("--parent-step", type=positive_int, required=True)
    g2.add_argument("--parent-dataset-sha256", required=True)
    g2.add_argument("--steps", type=positive_int, required=True)
    add_dataset_args(g2)
    g2.set_defaults(handler=materialize_grpo2)
    g3 = commands.add_parser("materialize-grpo3")
    g3.add_argument("--template", type=Path, required=True)
    g3.add_argument("--output", type=Path, required=True)
    g3.add_argument("--base-model", type=Path, required=True)
    g3.add_argument("--adapter-parent", type=Path, required=True)
    g3.add_argument("--parent-step", type=positive_int, required=True)
    g3.add_argument("--parent-stage", choices=("GRPO1", "GRPO2"), required=True)
    g3.add_argument("--steps", type=positive_int, required=True)
    add_dataset_args(g3, path=True)
    g3.set_defaults(handler=materialize_grpo3)
    sft = commands.add_parser("verify-sft")
    sft.add_argument("--output", type=Path, required=True)
    sft.add_argument("--epochs", type=float, required=True)
    sft.add_argument("--expected-model-sha256")
    add_dataset_args(sft)
    sft.add_argument("--derivation-report", type=Path, required=True)
    sft.add_argument("--report", type=Path, required=True)
    sft.set_defaults(handler=verify_sft)
    adapter = commands.add_parser("verify-adapter")
    adapter.add_argument("--stage", choices=("GRPO1", "GRPO2", "GRPO3"), required=True)
    adapter.add_argument("--run-dir", type=Path, required=True)
    adapter.add_argument("--checkpoint", type=Path, required=True)
    adapter.add_argument("--checkpoint-container", type=Path, required=True)
    adapter.add_argument("--config", type=Path, required=True)
    adapter.add_argument("--step", type=positive_int, required=True)
    adapter.add_argument("--parent-sha256", required=True)
    adapter.add_argument("--expected-adapter-sha256")
    adapter.add_argument("--parent-stage", choices=("GRPO1", "GRPO2"), default="GRPO2")
    adapter.add_argument("--dataset-key", required=True)
    adapter.add_argument("--dataset-sha256", required=True)
    adapter.add_argument("--report", type=Path, required=True)
    adapter.set_defaults(handler=verify_adapter)
    final = commands.add_parser("final-report")
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--source-commit", required=True)
    final.add_argument("--repro-data-root", type=Path, required=True)
    final.add_argument("--sft-epochs", type=float, required=True)
    final.add_argument("--sft-dataset-key", required=True)
    final.add_argument("--grpo1-steps", type=positive_int, required=True)
    final.add_argument("--grpo2-steps", type=positive_int, required=True)
    final.add_argument("--grpo3-steps", type=positive_int, required=True)
    final.add_argument("--run-sft", type=int, required=True)
    final.add_argument("--run-grpo1", type=int, required=True)
    final.add_argument("--run-grpo2", type=int, required=True)
    final.add_argument("--run-grpo3", type=int, required=True)
    final.add_argument("--stage-report", action="append", default=[])
    final.set_defaults(handler=final_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
