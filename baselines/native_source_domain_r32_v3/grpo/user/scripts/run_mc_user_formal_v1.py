#!/usr/bin/env python3
"""Formal continuous-run trainer for MC_USER_v1 Stage1-512."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from run_mc_user_e2e_smoke import (
    cleanup_generation_cache,
    load_beta_for_e2e,
    set_generation_mode,
    set_training_mode,
)
from run_mc_user_generation_smoke import (
    K,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    generate_k2_route,
    score_generated_route,
    validate_exact_policy_ids,
)
from run_mc_user_pilot_v1 import (
    advance_step_counts,
    assert_gpu_process_owned,
    assert_only_lora_trainable,
    candidate_record,
    claim_gpu_process_ownership,
    display_rollout_records,
    summarize_metrics,
    validate_prompt_metric,
)
from run_mc_user_real_smoke import (
    MEMORY_THRESHOLD_MIB,
    TRAIN_SHA256,
    batch_to_device,
    file_sha256,
    gpu_preflight,
    load_tokenizer,
    optimizer_state_is_lora_only,
    parameter_sha256,
    read_jsonl,
    validate_paths,
    validate_trainable,
)
from user_mc_train_step import mc_optimizer_step


OUTPUT_ROOT = Path("/data/GRPO_USER/runs/mc_user_v1_formal")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
FORMAL_RESUME = "LIMITATION"
FROZEN_CONFIG = {
    "experiment_type": "formal",
    "stage": "stage1_512",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": (
        "/data/outputs/baselines/native_source_domain_r32_v3/"
        "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/"
        "checkpoint-1106"
    ),
    "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
    "train_sha256": TRAIN_SHA256,
    "prompt_count": 512,
    "action_count": 256,
    "chain_count": 256,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_new_tokens": MAX_NEW_TOKENS,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "checkpoint_steps": [128, 256, 384, 512],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}


class MCFormalError(RuntimeError):
    pass


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_formal_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in FROZEN_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise MCFormalError(f"frozen Stage1 config mismatch: {mismatches}")
    return config


def _stable_seed_key(selection_seed: int, sample_id: str) -> str:
    return hashlib.sha256(
        f"{selection_seed}:{sample_id}".encode("utf-8")
    ).hexdigest()


def select_formal_rows(
    rows: Sequence[Mapping[str, Any]], selection_seed: int
) -> list[dict[str, Any]]:
    by_route: dict[str, dict[str, dict[str, Any]]] = {"action": {}, "chain": {}}
    for raw_row in rows:
        route = raw_row.get("route")
        sample_id = str(raw_row.get("sample_id", ""))
        if route not in by_route or not sample_id:
            continue
        if sample_id in by_route[route]:
            raise MCFormalError(f"duplicate {route} sample_id: {sample_id}")
        by_route[route][sample_id] = dict(raw_row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for route, required in (("action", 256), ("chain", 256)):
        ordered = sorted(
            by_route[route],
            key=lambda sample_id: (
                _stable_seed_key(selection_seed, sample_id),
                sample_id,
            ),
        )
        if len(ordered) < required:
            raise MCFormalError(f"insufficient unique {route} prompts")
        selected[route] = [by_route[route][sample_id] for sample_id in ordered[:required]]

    interleaved = [
        row
        for index in range(256)
        for row in (selected["action"][index], selected["chain"][index])
    ]
    validate_formal_selection(interleaved)
    return interleaved


def validate_formal_selection(rows: Sequence[Mapping[str, Any]]) -> None:
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    routes = [row.get("route") for row in rows]
    if len(rows) != 512 or len(set(sample_ids)) != 512 or not all(sample_ids):
        raise MCFormalError("formal Stage1 requires 512 unique sample IDs")
    if routes != [route for _ in range(256) for route in ("action", "chain")]:
        raise MCFormalError("formal Stage1 routes must strictly alternate Action/Chain")


def git_reproducibility_state(
    repo_root: Path,
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    commit = run_command(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    porcelain = run_command(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if porcelain.strip():
        raise MCFormalError("BLOCKED_DIRTY_WORKTREE")
    return {"git_commit": commit, "working_tree_clean": True}


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise MCFormalError("run-id contains unsupported characters")
    return run_id


def _protected_run_artifacts(run_dir: Path) -> list[str]:
    protected = [run_dir / "metrics.jsonl", run_dir / "summary.json"]
    checkpoint_dir = run_dir / "checkpoints"
    found = [path.name for path in protected if path.exists()]
    if checkpoint_dir.exists():
        found.append("checkpoints")
    return found


def assert_run_target_writable(run_dir: Path) -> None:
    protected = _protected_run_artifacts(run_dir)
    if protected:
        raise MCFormalError(
            f"BLOCKED_EXISTING_FORMAL_RUN: {sorted(protected)}"
        )


def build_manifest(
    *,
    run_id: str,
    config_path: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    git_commit: str,
) -> dict[str, Any]:
    validate_formal_selection(rows)
    return {
        "status": "READY_TO_EXECUTE",
        "run_kind": "user_grpo",
        "algorithm": "mc_user_v1",
        "experiment_type": "formal",
        "stage": "stage1_512",
        "run_id": run_id,
        "base_model": config["base_model"],
        "adapter": config["adapter"],
        "train_data": config["train_data"],
        "train_sha256": config["train_sha256"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "prompt_count": config["prompt_count"],
        "action_count": config["action_count"],
        "chain_count": config["chain_count"],
        "selection_seed": config["selection_seed"],
        "route_schedule": config["route_schedule"],
        "K": config["K"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_new_tokens": config["max_new_tokens"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "forward_batch_size": config["forward_batch_size"],
        "checkpoint_steps": list(config["checkpoint_steps"]),
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
        "FORMAL_RESUME": FORMAL_RESUME,
        "prompts": [
            {
                "prompt_step": index,
                "route": row["route"],
                "sample_id": row["sample_id"],
            }
            for index, row in enumerate(rows, 1)
        ],
    }


def run_preflight(
    args: argparse.Namespace,
    *,
    gpu_checker: Callable[..., Mapping[str, Any]] = gpu_preflight,
    git_checker: Callable[[Path], Mapping[str, Any]] = git_reproducibility_state,
    model_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    del model_loader  # An injected model loader is deliberately unusable pre-execute.
    config_path = args.config.expanduser().resolve()
    config = load_formal_config(config_path)
    config_sha256 = file_sha256(config_path)
    args.base_model = Path(config["base_model"])
    args.adapter = Path(config["adapter"])
    args.train_data = Path(config["train_data"])
    paths = validate_paths(args.base_model, args.adapter, args.train_data)
    if paths["train_sha256"] != config["train_sha256"]:
        raise MCFormalError("formal train SHA differs from frozen config")

    repo_root = Path(__file__).resolve().parents[5]
    git_state = dict(git_checker(repo_root))
    rows = select_formal_rows(
        read_jsonl(args.train_data), int(config["selection_seed"])
    )
    gpu = gpu_checker(args.gpu_id, args.memory_threshold_mib)
    run_id = _validate_run_id(args.run_id)
    output_root = Path(getattr(args, "output_root", OUTPUT_ROOT))
    run_dir = output_root / run_id
    assert_run_target_writable(run_dir)
    manifest = build_manifest(
        run_id=run_id,
        config_path=config_path,
        config_sha256=config_sha256,
        config=config,
        rows=rows,
        git_commit=str(git_state["git_commit"]),
    )
    existing_manifest = run_dir / "manifest.json"
    if existing_manifest.is_file():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        for key in ("config_sha256", "git_commit", "prompts"):
            if previous.get(key) != manifest.get(key):
                raise MCFormalError("BLOCKED_RUN_ID_CONTRACT_MISMATCH")
    _write_json(existing_manifest, manifest)
    preflight_record = {
        "status": "READY_TO_EXECUTE",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "gpu": gpu,
        "git_commit": git_state["git_commit"],
        "working_tree_clean": True,
        "config_sha256": config_sha256,
        "train_sha256": config["train_sha256"],
        "prompt_count": 512,
        "action_count": 256,
        "chain_count": 256,
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
        "FORMAL_RESUME": FORMAL_RESUME,
        "execute_required": True,
    }
    _write_json(run_dir / "preflight.json", preflight_record)
    return {
        **preflight_record,
        "config": config,
        "manifest": manifest,
        "rows": rows,
        "paths": paths,
        "run_dir": run_dir,
    }


def formal_candidate_record(
    candidate: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    marginal: Mapping[str, Any],
    route: str,
) -> dict[str, Any]:
    record = candidate_record(candidate, units, marginal, route)
    record.update(
        {
            "same_sign_overlap_token_count": int(
                candidate["same_sign_overlap_token_count"]
            ),
            "max_active_units_per_token": int(
                candidate["max_active_units_per_token"]
            ),
        }
    )
    return record


def run_formal_prompt_loop(
    rows: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    process_prompt: Callable[
        [Mapping[str, Any], torch.optim.Optimizer, int], Mapping[str, Any]
    ],
    checkpoint_writer: Callable[
        [int, int, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], None
    ],
    metric_writer: Callable[[Mapping[str, Any]], None],
    checkpoint_steps: Sequence[int],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    optimizer_step = 0
    checkpoints = set(int(step) for step in checkpoint_steps)
    for expected_step, row in enumerate(rows, 1):
        result = dict(process_prompt(row, optimizer, expected_step))
        prompt_step, optimizer_step = advance_step_counts(
            expected_step - 1, optimizer_step, result
        )
        record = {"prompt_step": prompt_step, "optimizer_step": optimizer_step, **result}
        validate_prompt_metric(record)
        records.append(record)
        metric_writer(record)
        if prompt_step in checkpoints:
            checkpoint_writer(prompt_step, optimizer_step, rows[:prompt_step], records)
    return records, optimizer_step


def save_formal_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    prompt_step: int,
    optimizer_step: int,
    processed_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    *,
    selection_seed: int,
    config_sha256: str,
    train_sha256: str,
    git_commit: str,
) -> None:
    checkpoint_dir = run_dir / "checkpoints" / f"prompt-step-{prompt_step:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    forbidden = [
        path.name
        for path in checkpoint_dir.iterdir()
        if path.name == "model.safetensors"
        or path.name.startswith("model-")
        or path.name.startswith("pytorch_model")
    ]
    if forbidden:
        raise MCFormalError(f"formal checkpoint contains base weights: {forbidden}")
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    if not required.issubset({path.name for path in checkpoint_dir.iterdir()}):
        raise MCFormalError("formal checkpoint is missing adapter-only weights")
    state = {
        "prompt_step": prompt_step,
        "optimizer_step": optimizer_step,
        "processed_sample_ids": [row["sample_id"] for row in processed_rows],
        "selection_seed": selection_seed,
        "config_sha256": config_sha256,
        "train_sha256": train_sha256,
        "git_commit": git_commit,
        "running_metrics": summarize_metrics(records, wall_seconds),
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
        "FORMAL_RESUME": FORMAL_RESUME,
    }
    _write_json(checkpoint_dir / "formal_state.json", state)


def execute_formal(
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    model_loader: Callable[[argparse.Namespace], torch.nn.Module] = load_beta_for_e2e,
) -> dict[str, Any]:
    run_dir = Path(preflight["run_dir"])
    assert_run_target_writable(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise MCFormalError("CUDA_VISIBLE_DEVICES must equal the explicit --gpu-id")
    config = preflight["config"]
    seed = int(config["selection_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from transformers import set_seed

    set_seed(seed)
    device = torch.device("cuda:0")
    tokenizer = load_tokenizer(args.base_model)
    model = model_loader(args)
    trainable = validate_trainable(model)
    assert_only_lora_trainable(model)
    initial_base_hash, _ = parameter_sha256(model, lora=False)
    _, lora_tensor_count = parameter_sha256(model, lora=True)
    if len(trainable) != 504 or lora_tensor_count != 504:
        raise MCFormalError("formal BETA contract requires 504/504 LoRA tensors")
    gpu_owner_claim = claim_gpu_process_ownership(args.gpu_id)
    runtime_metadata = {
        "gpu_owner_claim": gpu_owner_claim,
        "container_pid": {"value": os.getpid(), "diagnostic_only": True},
    }
    runtime_manifest = dict(preflight["manifest"])
    runtime_manifest["runtime"] = runtime_metadata
    _write_json(run_dir / "manifest.json", runtime_manifest)

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    metrics_path = run_dir / "metrics.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    started = time.perf_counter()
    display_optimizer_step = 0

    def process_prompt(
        row: Mapping[str, Any],
        persistent_optimizer: torch.optim.Optimizer,
        prompt_step: int,
    ) -> Mapping[str, Any]:
        nonlocal display_optimizer_step
        if file_sha256(args.train_data) != config["train_sha256"]:
            raise MCFormalError("frozen train_3000 SHA changed during formal run")
        assert_only_lora_trainable(model)
        assert_gpu_process_owned(
            args.gpu_id,
            claimed_gpu_uuid=gpu_owner_claim["gpu_uuid"],
            claimed_host_pid=gpu_owner_claim["nvidia_host_pid"],
        )
        route = str(row["route"])
        set_generation_mode(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        generated_ids, _generation_contract = generate_k2_route(
            model, tokenizer, row, device
        )
        torch.cuda.synchronize(device)
        generation_wall = time.perf_counter() - generation_started
        generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        scored = score_generated_route(route, row, generated_ids, tokenizer)
        validate_exact_policy_ids(scored["policy_batch"], generated_ids)
        units = scored["rollout"]["credit_units_per_candidate"]
        marginals = scored["rollout"]["marginal_results"]
        candidates = [
            formal_candidate_record(
                scored["candidates"][index], units[index], marginals[index], route
            )
            for index in range(K)
        ]
        cleanup_generation_cache(device)
        set_training_mode(model)
        training_batch = batch_to_device(scored["policy_batch"], device)
        validate_exact_policy_ids(training_batch, generated_ids)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        training_started = time.perf_counter()
        update = mc_optimizer_step(
            model,
            persistent_optimizer,
            training_batch,
            training_batch["credit_units_per_candidate"],
            forward_batch_size=int(config["forward_batch_size"]),
        )
        torch.cuda.synchronize(device)
        training_wall = time.perf_counter() - training_started
        training_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        assert_only_lora_trainable(model)
        assert_gpu_process_owned(
            args.gpu_id,
            claimed_gpu_uuid=gpu_owner_claim["gpu_uuid"],
            claimed_host_pid=gpu_owner_claim["nvidia_host_pid"],
        )
        if (
            not bool(update["finite"])
            or not math.isfinite(float(update["loss"]))
            or not math.isfinite(float(update["grad_norm"]))
        ):
            raise MCFormalError("non-finite formal loss or gradient")
        metadata = update["objective_metadata"]
        if bool(update["optimizer_step_performed"]):
            display_optimizer_step += 1
        for display_record in display_rollout_records(
            prompt_step=prompt_step,
            optimizer_step=display_optimizer_step,
            route=route,
            sample_id=str(row["sample_id"]),
            candidates=scored["candidates"],
            units_per_candidate=units,
        ):
            _append_jsonl(rollouts_path, display_record)
        return {
            "route": route,
            "sample_id": row["sample_id"],
            "candidates": candidates,
            "active_unit_count": int(metadata["active_unit_count"]),
            "active_token_count": int(metadata["active_token_count"]),
            "active_token_assignment_count": int(
                metadata["active_token_assignment_count"]
            ),
            "loss": float(update["loss"]),
            "grad_norm": float(update["grad_norm"]),
            "skipped_update": bool(update["skipped_update"]),
            "optimizer_step_performed": bool(update["optimizer_step_performed"]),
            "generation_wall_seconds": generation_wall,
            "training_wall_seconds": training_wall,
            "generation_peak_vram_mib": generation_peak,
            "training_peak_vram_mib": training_peak,
        }

    def write_checkpoint(
        prompt_step: int,
        optimizer_step: int,
        processed: Sequence[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        save_formal_checkpoint(
            model,
            run_dir,
            prompt_step,
            optimizer_step,
            processed,
            records,
            time.perf_counter() - started,
            selection_seed=seed,
            config_sha256=preflight["config_sha256"],
            train_sha256=config["train_sha256"],
            git_commit=preflight["git_commit"],
        )

    try:
        records, optimizer_step = run_formal_prompt_loop(
            preflight["rows"],
            optimizer,
            process_prompt,
            write_checkpoint,
            lambda record: _append_jsonl(metrics_path, record),
            config["checkpoint_steps"],
        )
        final_base_hash, _ = parameter_sha256(model, lora=False)
        if final_base_hash != initial_base_hash:
            raise MCFormalError("frozen base hash changed during formal run")
        if optimizer_step and not optimizer_state_is_lora_only(optimizer, trainable):
            raise MCFormalError("optimizer state is not restricted to LoRA parameters")
        final_checkpoint = run_dir / "checkpoints" / "prompt-step-0512"
        if not final_checkpoint.is_dir():
            raise MCFormalError("formal final prompt-step-0512 checkpoint is missing")
        summary = summarize_metrics(records, time.perf_counter() - started)
        summary.update(
            {
                "run_id": preflight["run_id"],
                "prompt_step": len(records),
                "optimizer_step": optimizer_step,
                "base_hash_unchanged": True,
                "trainable_tensor_count": len(trainable),
                "lora_tensor_count": lora_tensor_count,
                "resume_supported": False,
                "resume_policy": "continuous_run_only",
                "FORMAL_RESUME": FORMAL_RESUME,
                **runtime_metadata,
            }
        )
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        stopped = {
            "status": "STOPPED",
            "run_id": preflight["run_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
            "resume_supported": False,
            "resume_policy": "continuous_run_only",
            "FORMAL_RESUME": FORMAL_RESUME,
            **runtime_metadata,
        }
        _write_json(run_dir / "summary.json", stopped)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB
    )
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: preflight[key]
        for key in (
            "status",
            "run_id",
            "run_dir",
            "gpu",
            "git_commit",
            "working_tree_clean",
            "config_sha256",
            "train_sha256",
            "prompt_count",
            "action_count",
            "chain_count",
            "resume_supported",
            "resume_policy",
            "FORMAL_RESUME",
            "execute_required",
        )
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Callable[..., Mapping[str, Any]] = run_preflight,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_formal,
) -> Mapping[str, Any]:
    args = build_parser().parse_args(argv)
    preflight = preflight_fn(args)
    if not args.execute:
        output = public_preflight(preflight)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        print("READY_TO_EXECUTE")
        return output
    output = execute_fn(preflight, args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    try:
        output = run_cli()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 1
    return 0 if output["status"] in {"PASS", "READY_TO_EXECUTE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
