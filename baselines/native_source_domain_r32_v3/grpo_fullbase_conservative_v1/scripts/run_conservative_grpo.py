#!/usr/bin/env python3
"""Isolated full-model-parent GR_REC smoke/pilot runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LEGACY_SCRIPTS = HERE.parents[1] / "grpo" / "scripts"
for entry in (str(HERE), str(LEGACY_SCRIPTS)):
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

import trl_import_fix  # noqa: F401,E402
from checkpointing import LineageCheckpointCallback, assert_training_checkpoint, write_json_atomic  # noqa: E402
from evidence import StepEvidenceCallback, write_rank_summary  # noqa: E402
from grpo_probe import FixedProbeEvaluator, load_probe_records  # noqa: E402
from grpo_run_support import audit_sampler, count_raw_groups  # noqa: E402
from grpo_trl_trainer import (  # noqa: E402
    M_NO,
    M_THINK,
    RouteAwareRepeatSampler,
    build_route_dataset,
    make_nothink_reward_func,
    make_think_reward_func,
)
from modeling import (  # noqa: E402
    assert_optimizer_lora_only,
    base_tensor_fingerprints,
    load_training_parent,
    sampled_parameter_fingerprint,
)
from monitor.writer import monitor_from_env  # noqa: E402
from parent_contract import ParentSpec, file_sha256, validate_parent_files, validate_sha256  # noqa: E402
from retention import MilestoneRetentionCallback, verify_step0_parity, write_retention_summary  # noqa: E402
from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config, parameter_sha256  # noqa: E402
from trainer import FullBaseConservativeTrainer  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, encoding="utf-8"
    ).strip()


def working_tree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=HERE, text=True, encoding="utf-8"
    ).strip()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--monitor-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-mode", choices=("full_model", "adapter"), default="full_model")
    parser.add_argument("--parent-adapter")
    parser.add_argument("--parent-adapter-sha256")
    parser.add_argument("--execute", action="store_true")
    return parser


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", run_id):
        raise ValueError("invalid run ID")


def _parent_spec(args: argparse.Namespace, config: dict) -> ParentSpec:
    parent = config["parent"]
    if args.parent_mode != parent["parent_mode"]:
        raise RuntimeError(
            f"parent mode mismatch: config={parent['parent_mode']} cli={args.parent_mode}"
        )
    return ParentSpec(
        parent_mode=args.parent_mode,
        base_model_path=args.base_model,
        base_model_sha256=parent["model_sha256"],
        config_sha256=parent["config_sha256"],
        parent_adapter_path=args.parent_adapter,
        parent_adapter_sha256=args.parent_adapter_sha256,
        lineage=parent["model_role"],
    )


def preflight(args: argparse.Namespace) -> dict:
    _validate_run_id(args.run_id)
    config_path = Path(args.config).resolve()
    data_path = Path(args.data_path).resolve()
    config = _load_json(config_path)
    validate_sha256(config["dataset"]["sha256"], "dataset.sha256")
    spec = _parent_spec(args, config)
    parent_audit = validate_parent_files(spec, hash_weights=True)
    dataset_sha = file_sha256(data_path)
    if dataset_sha != config["dataset"]["sha256"]:
        raise RuntimeError(
            f"dataset SHA mismatch: expected {config['dataset']['sha256']}, got {dataset_sha}"
        )
    if not working_tree_clean():
        raise RuntimeError("working tree must be clean before GPU execution")
    output_dir = Path(args.output_root).resolve() / args.run_id
    monitor_dir = Path(args.monitor_root).resolve() / args.run_id
    for path in (output_dir, monitor_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"run directory is not empty: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "READY_TO_EXECUTE",
        "created_at": _utc_now(),
        "run_id": args.run_id,
        "run_kind": config["run_kind"],
        "parent": spec.public_dict(),
        "parent_audit": {**parent_audit, "base_model_path": "<runtime-supplied>"},
        "dataset_sha256": dataset_sha,
        "config_sha256": file_sha256(config_path),
        "git_commit": current_git_commit(),
        "working_tree_clean": True,
        "world_size": config["runtime"]["world_size"],
        "max_steps": config["optimization"]["max_steps"],
        "learning_rate": config["optimization"]["learning_rate"],
        "output_dir": str(output_dir),
        "monitor_dir": str(monitor_dir),
        "execute_required": True,
    }
    write_json_atomic(output_dir / "preflight.json", record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    print("READY_TO_EXECUTE", flush=True)
    return record


def _set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _assert_finite_history(history: list[dict]) -> None:
    for row in history:
        for key in ("loss", "grad_norm", "reward", "reward_std"):
            value = row.get(key)
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"non-finite {key} at step {row.get('step')}: {value}")


def _configure_monitor(run_id: str, monitor_root: Path) -> None:
    os.environ["GRPO_MONITOR"] = "1"
    os.environ["GRPO_DETAILED_MONITOR"] = "1"
    os.environ["GRPO_PARITY_AUDIT"] = "1"
    os.environ["GRPO_BEAM_RANK_BALANCE"] = "1"
    os.environ["GRPO_TRACE_EVERY"] = "1"
    os.environ["GRPO_RUN_ID"] = run_id
    os.environ["GRPO_MONITOR_DIR"] = str(monitor_root)


def _validate_runtime_environment(config: dict) -> None:
    expected = config["runtime"]["deterministic_environment"]
    mismatches = {
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"runtime environment contract mismatch: {mismatches}")


def execute(args: argparse.Namespace) -> dict | None:
    config = _load_json(args.config)
    _validate_runtime_environment(config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != int(config["runtime"]["world_size"]):
        raise RuntimeError(f"world size must be {config['runtime']['world_size']}, got {world}")
    torch.cuda.set_device(rank)
    output_dir = Path(args.output_root).resolve() / args.run_id
    monitor_root = Path(args.monitor_root).resolve()
    monitor_dir = monitor_root / args.run_id
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("execute requires a successful preflight record")
    preflight_record = _load_json(preflight_path)
    if preflight_record.get("status") != "READY_TO_EXECUTE":
        raise RuntimeError("preflight status is not READY_TO_EXECUTE")
    if preflight_record.get("git_commit") != current_git_commit() or not working_tree_clean():
        raise RuntimeError("code changed after preflight")
    if preflight_record.get("config_sha256") != file_sha256(args.config):
        raise RuntimeError("config changed after preflight")
    if preflight_record.get("dataset_sha256") != config["dataset"]["sha256"]:
        raise RuntimeError("dataset contract changed after preflight")
    if file_sha256(args.data_path) != config["dataset"]["sha256"]:
        raise RuntimeError("dataset bytes changed after preflight")
    unexpected = [
        path.name for path in output_dir.iterdir()
        if path.name != "preflight.json"
    ]
    if unexpected:
        raise RuntimeError(f"fresh execution output directory contains unexpected files: {unexpected}")

    _configure_monitor(args.run_id, monitor_root)
    spec = _parent_spec(args, config)
    model, tokenizer, trainable_audit = load_training_parent(
        spec,
        device=f"cuda:{rank}",
        dtype=torch.bfloat16,
        lora_seed=int(config["seeds"]["lora_initialization"]),
    )
    _set_runtime_seed(int(config["seeds"]["training"]) + rank)
    base_fingerprints_before = base_tensor_fingerprints(model)
    lora_fingerprint_before = sampled_parameter_fingerprint(
        model.named_parameters(), include_lora=True
    )
    lora_sha_before = parameter_sha256(model, True) if rank == 0 else None

    probe_ids = config["retention_probe"]["group_ids"]
    raw_groups = count_raw_groups(args.data_path)
    dataset = build_route_dataset(
        args.data_path,
        n_groups=raw_groups,
        seed=int(config["seeds"]["dataset"]),
        chunk=8,
        exclude_group_ids=probe_ids,
    )
    sampler = RouteAwareRepeatSampler(
        dataset, generation_batch_size=16, repeat_count=2, shuffle=False
    )
    sampler_audit = audit_sampler(dataset, sampler)
    cfg = make_grpo_config(
        str(output_dir),
        int(config["optimization"]["max_steps"]),
        float(config["optimization"]["learning_rate"]),
        int(config["seeds"]["training"]),
        save_strategy="steps",
        save_steps=int(config["checkpoint"]["save_steps"]),
        save_total_limit=int(config["checkpoint"]["save_total_limit"]),
    )
    monitor = monitor_from_env(args.run_id, rank)
    if monitor.enabled and rank == 0:
        monitor.write_manifest({
            "schema": "grpo_fullbase_conservative_v1",
            "run_id": args.run_id,
            "started_at": _utc_now(),
            "git_commit": current_git_commit(),
            "parent": spec.public_dict(),
            "dataset_sha256": config["dataset"]["sha256"],
            "config_sha256": preflight_record["config_sha256"],
            "world_size": world,
            "sampler_audit": sampler_audit,
            "optimization": config["optimization"],
            "seeds": config["seeds"],
            "retention_probe": config["retention_probe"],
            "trainable_audit": trainable_audit,
            "frozen_legacy_contract": config["frozen_legacy_contract"],
        })

    beam32_fn = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    trainer = FullBaseConservativeTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[
            make_nothink_reward_func(tokenizer=tokenizer),
            make_think_reward_func(beam32_fn=beam32_fn),
        ],
        monitor_writer=monitor,
    )
    trainer.create_optimizer()
    optimizer_audit = assert_optimizer_lora_only(trainer.model, trainer.optimizer)

    probe_records = load_probe_records(args.data_path, probe_ids)
    pure_evaluator = FixedProbeEvaluator(
        trainer=trainer,
        records=probe_records,
        group_ids=probe_ids,
        beam32_fn=beam32_fn,
        monitor=monitor,
        seed=int(config["seeds"]["probe"]),
        every_steps=2,
        probe_suite="step0_pure_full_sft",
    )
    retention_evaluator = FixedProbeEvaluator(
        trainer=trainer,
        records=probe_records,
        group_ids=probe_ids,
        beam32_fn=beam32_fn,
        monitor=monitor,
        seed=int(config["seeds"]["probe"]),
        every_steps=2,
        probe_suite="retention",
    )
    trainer.add_callback(MilestoneRetentionCallback(
        pure_evaluator, retention_evaluator,
        config["retention_probe"]["milestones"],
    ))
    trainer.add_callback(StepEvidenceCallback(output_dir, rank))
    trainer.add_callback(LineageCheckpointCallback({
        "parent_mode": spec.parent_mode,
        "parent_base_sha256": spec.base_model_sha256,
        "config_sha256": preflight_record["config_sha256"],
        "dataset_sha256": config["dataset"]["sha256"],
        "code_commit": current_git_commit(),
        "seed": int(config["seeds"]["training"]),
        "lora_seed": int(config["seeds"]["lora_initialization"]),
    }))

    started = time.time()
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    history = list(trainer.state.log_history)
    _assert_finite_history(history)
    base_fingerprints_after = base_tensor_fingerprints(model)
    if base_fingerprints_after != base_fingerprints_before:
        raise RuntimeError("in-memory base tensor fingerprint changed")
    lora_fingerprint_after = sampled_parameter_fingerprint(
        model.named_parameters(), include_lora=True
    )
    if lora_fingerprint_after == lora_fingerprint_before:
        raise RuntimeError("fresh LoRA fingerprint did not change")
    lora_sha_after = parameter_sha256(model, True) if rank == 0 else None
    if rank == 0 and lora_sha_after == lora_sha_before:
        raise RuntimeError("full LoRA hash did not change")

    summary = {
        "status": "PASS",
        "run_id": args.run_id,
        "run_kind": config["run_kind"],
        "rank": rank,
        "world_size": world,
        "global_step": int(trainer.state.global_step),
        "max_steps": int(config["optimization"]["max_steps"]),
        "train_runtime_sec": time.time() - started,
        "train_loss": float(result.training_loss),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(rank) // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(rank) // (1024 * 1024),
        "trainable_audit": trainable_audit,
        "optimizer_audit": optimizer_audit,
        "sampler_audit": sampler_audit,
        "base_tensor_fingerprints_before": base_fingerprints_before,
        "base_tensor_fingerprints_after": base_fingerprints_after,
        "base_tensor_unchanged": True,
        "lora_sample_fingerprint_before": lora_fingerprint_before,
        "lora_sample_fingerprint_after": lora_fingerprint_after,
        "lora_sha256_before": lora_sha_before,
        "lora_sha256_after": lora_sha_after,
        "lora_changed": True,
        "rollouts": trainer._smoke_log,
        "parity_log": trainer._parity_log,
        "log_history": history,
    }
    write_rank_summary(output_dir / f"run-summary-rank{rank}.json", summary)
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
        checkpoint_audit = assert_training_checkpoint(final_checkpoint, world)
        base_sha_after = file_sha256(Path(args.base_model) / "model.safetensors")
        if base_sha_after != spec.base_model_sha256:
            raise RuntimeError("immutable full-SFT base file changed during GRPO")
        step0 = verify_step0_parity(monitor_dir / "probes.jsonl", probe_ids)
        retention = write_retention_summary(
            monitor_dir, output_dir / "retention-summary.json"
        )
        final = {
            "status": "PASS",
            "run_id": args.run_id,
            "run_kind": config["run_kind"],
            "global_step": int(trainer.state.global_step),
            "base_sha256_before": preflight_record["parent_audit"]["verified_model_sha256"],
            "base_sha256_after": base_sha_after,
            "base_unchanged": True,
            "lora_changed": True,
            "checkpoint": checkpoint_audit,
            "step0_parity": step0,
            "retention": retention,
            "completed_at": _utc_now(),
        }
        write_json_atomic(output_dir / "summary.json", final)
        print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
        return final
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        preflight(args)
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
