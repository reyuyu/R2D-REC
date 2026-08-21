#!/usr/bin/env python3
"""Gated 32-prompt online pilot runner for MC_USER_v1."""

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
from run_mc_user_real_smoke import (
    ADAPTER,
    BASE_MODEL,
    MEMORY_THRESHOLD_MIB,
    TRAIN_DATA,
    TRAIN_SHA256,
    batch_to_device,
    file_sha256,
    gpu_preflight,
    load_tokenizer,
    optimizer_state_is_lora_only,
    parameter_sha256,
    read_jsonl,
    select_fixed_rows,
    validate_paths,
    validate_trainable,
)
from user_mc_train_step import mc_optimizer_step


SEED = 20260822
ACTION_COUNT = 16
CHAIN_COUNT = 16
PILOT_PROMPT_COUNT = ACTION_COUNT + CHAIN_COUNT
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
FORWARD_BATCH_SIZE = 1
CHECKPOINT_STEPS = (8, 16, 32)
OUTPUT_ROOT = Path("/data/GRPO_USER/runs/mc_user_v1_pilot32")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class MCPilotError(RuntimeError):
    pass


def _stable_seed_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def smoke_sample_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return rows shared by the fixed, generation, and E2E smoke fixtures."""

    fixed = select_fixed_rows(rows)
    return {str(fixed["action"]["sample_id"]), str(fixed["chain"]["sample_id"])}


def select_pilot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SEED,
    excluded_sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = set(excluded_sample_ids or ())
    by_route: dict[str, dict[str, dict[str, Any]]] = {"action": {}, "chain": {}}
    for raw_row in rows:
        route = raw_row.get("route")
        sample_id = str(raw_row.get("sample_id", ""))
        if route not in by_route or not sample_id or sample_id in excluded:
            continue
        if sample_id in by_route[route]:
            raise MCPilotError(f"duplicate {route} sample_id in train data: {sample_id}")
        by_route[route][sample_id] = dict(raw_row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for route, required in (("action", ACTION_COUNT), ("chain", CHAIN_COUNT)):
        ordered_ids = sorted(
            by_route[route], key=lambda sample_id: (_stable_seed_key(sample_id, seed), sample_id)
        )
        if len(ordered_ids) < required:
            raise MCPilotError(f"insufficient unique {route} prompts for pilot")
        selected[route] = [by_route[route][sample_id] for sample_id in ordered_ids[:required]]

    interleaved = []
    for index in range(ACTION_COUNT):
        interleaved.extend((selected["action"][index], selected["chain"][index]))
    validate_pilot_selection(interleaved, excluded)
    return interleaved


def validate_pilot_selection(
    rows: Sequence[Mapping[str, Any]], excluded_sample_ids: set[str] | None = None
) -> None:
    excluded = set(excluded_sample_ids or ())
    sample_ids = [str(row["sample_id"]) for row in rows]
    expected_routes = [route for _ in range(ACTION_COUNT) for route in ("action", "chain")]
    if len(rows) != PILOT_PROMPT_COUNT or len(set(sample_ids)) != PILOT_PROMPT_COUNT:
        raise MCPilotError("pilot selection must contain exactly 32 unique prompts")
    if [row.get("route") for row in rows] != expected_routes:
        raise MCPilotError("pilot selection must strictly alternate Action and Chain")
    if set(sample_ids) & excluded:
        raise MCPilotError("pilot selection overlaps a prior smoke fixture")


def pilot_config() -> dict[str, Any]:
    return {
        "prompt_count": PILOT_PROMPT_COUNT,
        "action_count": ACTION_COUNT,
        "chain_count": CHAIN_COUNT,
        "K": K,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "checkpoint_prompt_steps": list(CHECKPOINT_STEPS),
        "seed": SEED,
    }


def build_manifest(
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    excluded: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_pilot_selection(rows, excluded)
    return {
        "status": "READY_TO_EXECUTE",
        "run_id": run_id,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "train_data": str(args.train_data),
        "train_sha256": TRAIN_SHA256,
        "selection_method": "sha256(seed:sample_id), then sample_id",
        "excluded_smoke_sample_ids": sorted(excluded),
        "config": pilot_config(),
        "prompts": [
            {
                "prompt_step": index,
                "route": row["route"],
                "sample_id": row["sample_id"],
            }
            for index, row in enumerate(rows, 1)
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _default_run_id() -> str:
    return time.strftime("MC-USER-PILOT32-%Y%m%d-%H%M%S", time.localtime())


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise MCPilotError("run-id contains unsupported characters")
    return run_id


def run_preflight(
    args: argparse.Namespace,
    *,
    gpu_checker: Callable[..., Mapping[str, Any]] = gpu_preflight,
) -> dict[str, Any]:
    paths = validate_paths(args.base_model, args.adapter, args.train_data)
    rows = read_jsonl(args.train_data)
    excluded = smoke_sample_ids(rows)
    selection = select_pilot_rows(rows, seed=SEED, excluded_sample_ids=excluded)
    gpu = gpu_checker(args.gpu_id, args.memory_threshold_mib)
    run_id = _validate_run_id(args.run_id or _default_run_id())
    run_dir = args.output_root / run_id
    manifest = build_manifest(run_id, selection, excluded, args)
    _write_json(run_dir / "manifest.json", manifest)
    return {
        "status": "READY_TO_EXECUTE",
        "paths": paths,
        "gpu": gpu,
        "rows": selection,
        "excluded_sample_ids": sorted(excluded),
        "manifest": manifest,
        "run_dir": run_dir,
    }


def assert_only_lora_trainable(model: torch.nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != 504 or any("lora_" not in name.lower() for name in trainable):
        raise MCPilotError("expected exactly 504 trainable LoRA tensors")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name.lower()
    ):
        raise MCPilotError("a base parameter became trainable")


def assert_gpu_process_owned(
    gpu_id: int,
    *,
    current_pid: int | None = None,
    run_command: Callable[..., Any] = subprocess.run,
) -> None:
    current_pid = os.getpid() if current_pid is None else current_pid
    gpu_rows = run_command(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    uuid_by_index = {
        int(index.strip()): uuid.strip()
        for index, uuid in (line.split(",", 1) for line in gpu_rows.splitlines() if line.strip())
    }
    if gpu_id not in uuid_by_index:
        raise MCPilotError(f"GPU {gpu_id} disappeared")
    process_rows = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    foreign = []
    for line in process_rows.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used = [part.strip() for part in line.split(",", 3)]
        if uuid == uuid_by_index[gpu_id] and int(pid) != current_pid:
            foreign.append({"pid": int(pid), "process_name": process_name, "used": used})
    if foreign:
        raise MCPilotError(f"GPU {gpu_id} acquired a foreign compute process")


def candidate_record(
    candidate: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    marginal: Mapping[str, Any],
    route: str,
) -> dict[str, Any]:
    record = {
        "candidate_index": int(candidate["candidate_index"]),
        "generated_token_count": int(candidate["generated_token_count"]),
        "format_valid": bool(candidate["format_valid"]),
        "reward": float(candidate["full_reward"]),
        "projection_required": bool(candidate["projection_required"]),
        "positive_unit_count": sum(float(unit["delta"]) > 0 for unit in units),
        "negative_unit_count": sum(float(unit["delta"]) < 0 for unit in units),
        "zero_unit_count": sum(float(unit["delta"]) == 0 for unit in units),
        "positive_credit_mass": sum(
            float(unit["delta"]) for unit in units if float(unit["delta"]) > 0
        ),
        "negative_credit_mass": sum(
            abs(float(unit["delta"])) for unit in units if float(unit["delta"]) < 0
        ),
        "overlap_token_count": int(candidate["overlap_token_count"]),
        "mixed_sign_overlap_token_count": int(candidate["mixed_sign_overlap_token_count"]),
    }
    if route == "action":
        record["predicted_sid_unit_count"] = len(marginal["credits"]) if marginal["valid"] else 0
    else:
        record.update(
            {
                "predicted_event_count": len(marginal["credits"]) if marginal["valid"] else 0,
                "full_action_alignment": float(marginal.get("full_action_alignment", 0.0)),
                "full_logic_alignment": float(marginal.get("full_logic_alignment", 0.0)),
            }
        )
    return record


def validate_prompt_metric(record: Mapping[str, Any]) -> None:
    required = {
        "prompt_step", "optimizer_step", "route", "sample_id", "candidates",
        "active_unit_count", "active_token_count", "active_token_assignment_count",
        "loss", "grad_norm", "skipped_update", "optimizer_step_performed",
        "generation_wall_seconds", "training_wall_seconds",
        "generation_peak_vram_mib", "training_peak_vram_mib",
    }
    missing = required - set(record)
    if missing:
        raise MCPilotError(f"prompt metric is missing fields: {sorted(missing)}")
    if len(record["candidates"]) != K:
        raise MCPilotError("each pilot prompt must retain exactly K=2 candidates")


def summarize_metrics(records: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    candidates = [candidate for record in records for candidate in record["candidates"]]
    action = [candidate for record in records if record["route"] == "action" for candidate in record["candidates"]]
    chain = [candidate for record in records if record["route"] == "chain" for candidate in record["candidates"]]
    losses = [float(record["loss"]) for record in records if not record["skipped_update"]]
    grads = [float(record["grad_norm"]) for record in records if not record["skipped_update"]]

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def total(items: Sequence[Mapping[str, Any]], key: str) -> float:
        return sum(float(item[key]) for item in items)

    summary = {
        "status": "PASS",
        "prompt_count": len(records),
        "optimizer_update_count": sum(bool(record["optimizer_step_performed"]) for record in records),
        "valid_candidate_rate": ratio(sum(bool(item["format_valid"]) for item in candidates), len(candidates)),
        "projection_required_rate": ratio(sum(bool(item["projection_required"]) for item in candidates), len(candidates)),
        "overlap_candidate_rate": ratio(sum(int(item["overlap_token_count"]) > 0 for item in candidates), len(candidates)),
        "no_credit_prompt_rate": ratio(sum(bool(record["skipped_update"]) for record in records), len(records)),
        "grad_norm": {"mean": sum(grads) / len(grads) if grads else 0.0, "max": max(grads, default=0.0)},
        "loss": {
            "mean": sum(losses) / len(losses) if losses else 0.0,
            "min": min(losses, default=0.0),
            "max": max(losses, default=0.0),
        },
        "peak_vram_mib": max(
            (max(float(record["generation_peak_vram_mib"]), float(record["training_peak_vram_mib"])) for record in records),
            default=0.0,
        ),
        "wall_seconds": wall_seconds,
        "action": {
            "candidate_count": len(action),
            "candidate_with_negative_unit_count": sum(int(item["negative_unit_count"]) > 0 for item in action),
            "negative_candidate_rate": ratio(sum(int(item["negative_unit_count"]) > 0 for item in action), len(action)),
            "total_positive_credit_mass": total(action, "positive_credit_mass"),
            "total_negative_credit_mass": total(action, "negative_credit_mass"),
            "mean_predicted_sid_count": total(action, "predicted_sid_unit_count") / len(action) if action else 0.0,
            "mean_reward": total(action, "reward") / len(action) if action else 0.0,
        },
        "chain": {
            "candidate_count": len(chain),
            "candidate_with_negative_unit_count": sum(int(item["negative_unit_count"]) > 0 for item in chain),
            "negative_candidate_rate": ratio(sum(int(item["negative_unit_count"]) > 0 for item in chain), len(chain)),
            "total_positive_credit_mass": total(chain, "positive_credit_mass"),
            "total_negative_credit_mass": total(chain, "negative_credit_mass"),
            "mean_predicted_event_count": total(chain, "predicted_event_count") / len(chain) if chain else 0.0,
            "mean_reward": total(chain, "reward") / len(chain) if chain else 0.0,
            "mean_action_alignment": total(chain, "full_action_alignment") / len(chain) if chain else 0.0,
            "mean_logic_alignment": total(chain, "full_logic_alignment") / len(chain) if chain else 0.0,
        },
    }
    return summary


def advance_step_counts(prompt_step: int, optimizer_step: int, update: Mapping[str, Any]) -> tuple[int, int]:
    prompt_step += 1
    if bool(update["optimizer_step_performed"]):
        optimizer_step += 1
    if bool(update["skipped_update"]) == bool(update["optimizer_step_performed"]):
        raise MCPilotError("optimizer step accounting is inconsistent")
    return prompt_step, optimizer_step


def run_prompt_loop(
    rows: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    process_prompt: Callable[[Mapping[str, Any], torch.optim.Optimizer, int], Mapping[str, Any]],
    checkpoint_writer: Callable[[int, int, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], None],
    metric_writer: Callable[[Mapping[str, Any]], None],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    optimizer_step = 0
    for expected_step, row in enumerate(rows, 1):
        result = dict(process_prompt(row, optimizer, expected_step))
        prompt_step, optimizer_step = advance_step_counts(expected_step - 1, optimizer_step, result)
        record = {"prompt_step": prompt_step, "optimizer_step": optimizer_step, **result}
        validate_prompt_metric(record)
        records.append(record)
        metric_writer(record)
        if prompt_step in CHECKPOINT_STEPS:
            checkpoint_writer(prompt_step, optimizer_step, rows[:prompt_step], records)
    return records, optimizer_step


def save_adapter_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    prompt_step: int,
    optimizer_step: int,
    processed_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    wall_seconds: float,
) -> None:
    checkpoint_dir = run_dir / "checkpoints" / f"prompt-step-{prompt_step:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    forbidden = (checkpoint_dir / "model.safetensors", checkpoint_dir / "pytorch_model.bin")
    if any(path.exists() for path in forbidden):
        raise MCPilotError("checkpoint unexpectedly contains base-model weights")
    state = {
        "prompt_step": prompt_step,
        "optimizer_step": optimizer_step,
        "sample_ids_processed": [row["sample_id"] for row in processed_rows],
        "seed": SEED,
        "config": pilot_config(),
        "running_metrics": summarize_metrics(records, wall_seconds),
    }
    _write_json(checkpoint_dir / "pilot_state.json", state)


def execute_pilot(
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    model_loader: Callable[[argparse.Namespace], torch.nn.Module] = load_beta_for_e2e,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise MCPilotError("CUDA_VISIBLE_DEVICES must equal the explicit --gpu-id")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from transformers import set_seed

    set_seed(SEED)
    device = torch.device("cuda:0")
    tokenizer = load_tokenizer(args.base_model)
    model = model_loader(args)
    trainable = validate_trainable(model)
    assert_only_lora_trainable(model)
    initial_base_hash, _ = parameter_sha256(model, lora=False)
    _, lora_tensor_count = parameter_sha256(model, lora=True)
    if len(trainable) != 504 or lora_tensor_count != 504:
        raise MCPilotError("BETA LoRA structure is not the frozen 504-tensor contract")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    run_dir = Path(preflight["run_dir"])
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()

    def process_prompt(row: Mapping[str, Any], persistent_optimizer: torch.optim.Optimizer, prompt_step: int) -> Mapping[str, Any]:
        if file_sha256(args.train_data) != TRAIN_SHA256:
            raise MCPilotError("frozen train_3000 SHA changed during pilot")
        assert_only_lora_trainable(model)
        assert_gpu_process_owned(args.gpu_id)
        route = str(row["route"])
        set_generation_mode(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        generated_ids, _ = generate_k2_route(model, tokenizer, row, device)
        torch.cuda.synchronize(device)
        generation_wall = time.perf_counter() - generation_started
        generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        scored = score_generated_route(route, row, generated_ids, tokenizer)
        validate_exact_policy_ids(scored["policy_batch"], generated_ids)
        units = scored["rollout"]["credit_units_per_candidate"]
        marginals = scored["rollout"]["marginal_results"]
        candidates = [
            candidate_record(scored["candidates"][index], units[index], marginals[index], route)
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
            forward_batch_size=FORWARD_BATCH_SIZE,
        )
        torch.cuda.synchronize(device)
        training_wall = time.perf_counter() - training_started
        training_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        assert_only_lora_trainable(model)
        assert_gpu_process_owned(args.gpu_id)
        if not bool(update["finite"]) or not math.isfinite(float(update["loss"])) or not math.isfinite(float(update["grad_norm"])):
            raise MCPilotError("non-finite pilot loss or gradient")
        metadata = update["objective_metadata"]
        return {
            "route": route,
            "sample_id": row["sample_id"],
            "candidates": candidates,
            "active_unit_count": int(metadata["active_unit_count"]),
            "active_token_count": int(metadata["active_token_count"]),
            "active_token_assignment_count": int(metadata["active_token_assignment_count"]),
            "loss": float(update["loss"]),
            "grad_norm": float(update["grad_norm"]),
            "skipped_update": bool(update["skipped_update"]),
            "optimizer_step_performed": bool(update["optimizer_step_performed"]),
            "generation_wall_seconds": generation_wall,
            "training_wall_seconds": training_wall,
            "generation_peak_vram_mib": generation_peak,
            "training_peak_vram_mib": training_peak,
        }

    def write_checkpoint(prompt_step: int, optimizer_step: int, processed: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> None:
        save_adapter_checkpoint(
            model, run_dir, prompt_step, optimizer_step, processed, records, time.perf_counter() - started
        )

    try:
        records, optimizer_step = run_prompt_loop(
            preflight["rows"],
            optimizer,
            process_prompt,
            write_checkpoint,
            lambda record: _append_jsonl(metrics_path, record),
        )
        final_base_hash, _ = parameter_sha256(model, lora=False)
        if final_base_hash != initial_base_hash:
            raise MCPilotError("frozen base hash changed during pilot")
        if not optimizer_state_is_lora_only(optimizer, trainable) and optimizer_step:
            raise MCPilotError("optimizer state is not restricted to LoRA parameters")
        summary = summarize_metrics(records, time.perf_counter() - started)
        summary.update(
            {
                "run_id": preflight["manifest"]["run_id"],
                "prompt_step": len(records),
                "optimizer_step": optimizer_step,
                "trainable_tensor_count": len(trainable),
                "lora_tensor_count": lora_tensor_count,
                "base_hash_unchanged": True,
            }
        )
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        stopped = {
            "status": "STOPPED",
            "run_id": preflight["manifest"]["run_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "summary.json", stopped)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--base-model", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--adapter", type=Path, default=Path(ADAPTER))
    parser.add_argument("--train-data", type=Path, default=Path(TRAIN_DATA))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB)
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    manifest = preflight["manifest"]
    return {
        "status": "READY_TO_EXECUTE",
        "run_id": manifest["run_id"],
        "run_dir": str(preflight["run_dir"]),
        "gpu": preflight["gpu"],
        "train_sha256": manifest["train_sha256"],
        "prompt_count": len(manifest["prompts"]),
        "action_count": sum(item["route"] == "action" for item in manifest["prompts"]),
        "chain_count": sum(item["route"] == "chain" for item in manifest["prompts"]),
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Callable[..., Mapping[str, Any]] = run_preflight,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_pilot,
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
