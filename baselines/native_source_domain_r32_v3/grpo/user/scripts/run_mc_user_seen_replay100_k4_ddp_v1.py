#!/usr/bin/env python3
"""Replay the first 100 seen Stage1 prompts from its final LoRA on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from run_mc_user_e2e_smoke import cleanup_generation_cache, set_generation_mode, set_training_mode
from run_mc_user_formal_k4_ddp_v1 import (
    K,
    WORLD_SIZE,
    MCK4Error,
    RUN_ID_RE,
    _all_gpu_preflight,
    _gather,
    _rank_hashes,
    assert_ddp_gpu_process_owned,
    candidate_seed,
    claim_ddp_gpu_process_ownership,
    distributed_mc_optimizer_step,
    generate_one_candidate,
    load_beta_for_rank,
    score_rank_candidate,
)
from run_mc_user_formal_v1 import (
    FORMAL_RESUME,
    _append_jsonl,
    _write_json,
    assert_run_target_writable,
    candidate_evaluator_means,
    git_reproducibility_state,
    save_formal_checkpoint,
    summarize_metrics,
)
from run_mc_user_formal_probe_sidecar_v1 import validate_adapter_only
from run_mc_user_real_smoke import (
    MEMORY_THRESHOLD_MIB,
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


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_user_seen_replay100_k4_ddp.json"
OUTPUT_ROOT = Path("/data/GRPO_USER/runs/mc_user_v1_replay100")
FROZEN_CONFIG = {
    "experiment_type": "continuation",
    "stage": "stage2_seen_replay100_k4",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": "/data/GRPO_USER/runs/mc_user_v1_formal/MC-USER-FORMAL-S1-512-K4-20260822-155521/checkpoints/prompt-step-0512",
    "parent_run_dir": "/data/GRPO_USER/runs/mc_user_v1_formal/MC-USER-FORMAL-S1-512-K4-20260822-155521",
    "parent_run_id": "MC-USER-FORMAL-S1-512-K4-20260822-155521",
    "parent_manifest_sha256": "e990528e7ecb7c0a8064722265e005cbab0eec0b9da9ab88a696caead7bfb9e6",
    "parent_summary_sha256": "3110808371f2b273ec2408b393ef99ab86b361692eeb52717e7e3a9cd6bbfd99",
    "parent_adapter_sha256": "e74fce3456d2f0e14807963f0909039d27955eda5a924bae8b84a1fb5e11e6ad",
    "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "selection_mode": "parent_manifest_prefix",
    "source_prompt_start": 1,
    "prompt_count": 100,
    "action_count": 50,
    "chain_count": 50,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "checkpoint_steps": [50, 100],
    "optimizer_state_policy": "fresh_adamw",
    "expected_optimizer_updates": 100,
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}


class MCReplayError(MCK4Error):
    pass


def load_replay_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in FROZEN_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise MCReplayError(f"frozen Replay100 config mismatch: {mismatches}")
    return config


def load_parent_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = Path(config["parent_run_dir"])
    manifest_path, summary_path = parent / "manifest.json", parent / "summary.json"
    if file_sha256(manifest_path) != config["parent_manifest_sha256"]:
        raise MCReplayError("parent manifest SHA mismatch")
    if file_sha256(summary_path) != config["parent_summary_sha256"]:
        raise MCReplayError("parent summary SHA mismatch")
    if file_sha256(Path(config["adapter"]) / "adapter_model.safetensors") != config["parent_adapter_sha256"]:
        raise MCReplayError("parent adapter SHA mismatch")
    validate_adapter_only(Path(config["adapter"]), 512)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prompts = manifest.get("prompts", [])
    if manifest.get("run_id") != config["parent_run_id"] or summary.get("status") != "PASS":
        raise MCReplayError("parent run is not the frozen PASS run")
    if len(prompts) != 512 or len({row.get("sample_id") for row in prompts}) != 512:
        raise MCReplayError("parent manifest must contain 512 unique prompts")
    if summary.get("base_hash_unchanged") is not True or summary.get("lora_tensor_count") != 504:
        raise MCReplayError("parent integrity contract failed")
    return manifest, summary


def select_parent_prefix_rows(
    train_rows: Sequence[Mapping[str, Any]],
    parent_manifest: Mapping[str, Any],
    prompt_count: int = 100,
) -> list[dict[str, Any]]:
    source = list(parent_manifest["prompts"][:prompt_count])
    indexed = {str(row["sample_id"]): row for row in train_rows}
    if len(source) != prompt_count or any(str(item["sample_id"]) not in indexed for item in source):
        raise MCReplayError("parent prefix cannot be reconstructed from frozen train data")
    selected = [dict(indexed[str(item["sample_id"])]) for item in source]
    expected_routes = ["action" if index % 2 == 0 else "chain" for index in range(prompt_count)]
    if [str(row["route"]) for row in selected] != expected_routes:
        raise MCReplayError("parent prefix route order is not strict alternating")
    if [str(row["sample_id"]) for row in selected] != [str(item["sample_id"]) for item in source]:
        raise MCReplayError("replay sample order differs from parent manifest")
    return selected


def build_manifest(
    run_id: str,
    config_path: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    git_commit: str,
) -> dict[str, Any]:
    return {
        "status": "READY_TO_EXECUTE",
        "run_kind": "user_grpo",
        "algorithm": "mc_user_v1",
        "experiment_type": config["experiment_type"],
        "stage": config["stage"],
        "run_id": run_id,
        "parent_run_id": config["parent_run_id"],
        "parent_run_dir": config["parent_run_dir"],
        "parent_checkpoint": config["adapter"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "parent_summary_sha256": config["parent_summary_sha256"],
        "parent_adapter_sha256": config["parent_adapter_sha256"],
        "base_model": config["base_model"],
        "adapter": config["adapter"],
        "train_data": config["train_data"],
        "train_sha256": config["train_sha256"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "prompt_count": len(rows),
        "action_count": sum(row["route"] == "action" for row in rows),
        "chain_count": sum(row["route"] == "chain" for row in rows),
        "selection_mode": config["selection_mode"],
        "source_prompt_start": config["source_prompt_start"],
        "selection_seed": config["selection_seed"],
        "route_schedule": config["route_schedule"],
        "replay_of_seen_samples": True,
        "comparison_contract": "same_sample_order_same_prompt_step_same_candidate_seed_rule",
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_new_tokens": config["max_new_tokens"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "forward_batch_size": config["forward_batch_size"],
        "checkpoint_steps": list(config["checkpoint_steps"]),
        "optimizer_state_policy": config["optimizer_state_policy"],
        "expected_optimizer_updates": config["expected_optimizer_updates"],
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
        "FORMAL_RESUME": FORMAL_RESUME,
        "prompts": [
            {
                "prompt_step": index,
                "source_prompt_step": index,
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
    git_checker: Callable[..., Mapping[str, Any]] = git_reproducibility_state,
) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = load_replay_config(config_path)
    paths = validate_paths(Path(config["base_model"]), Path(config["adapter"]), Path(config["train_data"]))
    if paths["train_sha256"] != config["train_sha256"]:
        raise MCReplayError("frozen train SHA mismatch")
    parent_manifest, _ = load_parent_contract(config)
    rows = select_parent_prefix_rows(read_jsonl(Path(config["train_data"])), parent_manifest, int(config["prompt_count"]))
    if len(rows) != 100 or sum(row["route"] == "action" for row in rows) != 50:
        raise MCReplayError("Replay100 must be 50 Action + 50 Chain")
    repo_root = Path(__file__).resolve().parents[5]
    git_state = dict(git_checker(repo_root))
    gpus = _all_gpu_preflight(args.memory_threshold_mib, checker=gpu_checker)
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise MCReplayError("run-id contains unsupported characters")
    run_dir = args.output_root / args.run_id
    assert_run_target_writable(run_dir)
    config_sha256 = file_sha256(config_path)
    manifest = build_manifest(args.run_id, config_path, config_sha256, config, rows, str(git_state["git_commit"]))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise MCReplayError("BLOCKED_RUN_ID_CONTRACT_MISMATCH")
    _write_json(manifest_path, manifest)
    preflight = {
        "status": "READY_TO_EXECUTE",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "gpus": gpus,
        "git_commit": git_state["git_commit"],
        "working_tree_clean": True,
        "config_sha256": config_sha256,
        "train_sha256": config["train_sha256"],
        "parent_adapter_sha256": config["parent_adapter_sha256"],
        "prompt_count": 100,
        "action_count": 50,
        "chain_count": 50,
        "expected_optimizer_updates": 100,
        "optimizer_state_policy": "fresh_adamw",
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "execute_required": True,
    }
    _write_json(run_dir / "preflight.json", preflight)
    return preflight


def _load_execute_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    run_dir = args.output_root / args.run_id
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config_path = args.config.expanduser().resolve()
    config = load_replay_config(config_path)
    if preflight.get("status") != "READY_TO_EXECUTE" or manifest.get("config_sha256") != file_sha256(config_path):
        raise MCReplayError("execute preflight/config contract mismatch")
    if git_reproducibility_state(Path(__file__).resolve().parents[5])["git_commit"] != manifest.get("git_commit"):
        raise MCReplayError("execute git commit differs from preflight")
    if file_sha256(Path(config["train_data"])) != config["train_sha256"]:
        raise MCReplayError("execute train SHA differs from preflight")
    parent_manifest, _ = load_parent_contract(config)
    rows = select_parent_prefix_rows(read_jsonl(Path(config["train_data"])), parent_manifest, 100)
    return preflight, config, rows, run_dir


def execute_distributed(args: argparse.Namespace) -> dict[str, Any] | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3" or int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise MCReplayError("execute requires CUDA_VISIBLE_DEVICES=0,1,2,3 and world_size=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    preflight, config, rows, run_dir = _load_execute_contract(args)
    dist.barrier()
    tokenizer = load_tokenizer(config["base_model"])
    model = load_beta_for_rank(config, local_rank)
    trainable = validate_trainable(model)
    if len(trainable) != 504:
        raise MCReplayError("Replay100 requires 504 LoRA trainable tensors")
    dist.barrier()
    owner = claim_ddp_gpu_process_ownership(local_rank)
    initial_base_hash = parameter_sha256(model, lora=False)[0] if rank == 0 else None
    initial_lora_hash = parameter_sha256(model, lora=True)[0] if rank == 0 else None
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    _rank_hashes(ddp.module)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    records: list[dict[str, Any]] = []
    optimizer_step = 0
    started = time.perf_counter()
    metrics_path, rollouts_path = run_dir / "metrics.jsonl", run_dir / "rollouts.jsonl"
    try:
        for prompt_step, row in enumerate(rows, 1):
            assert_ddp_gpu_process_owned(owner)
            seed = candidate_seed(int(config["selection_seed"]), prompt_step, str(row["sample_id"]), rank)
            set_generation_mode(ddp.module)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            generation_started = time.perf_counter()
            generated_ids = generate_one_candidate(ddp.module, tokenizer, row, device, seed, config)
            torch.cuda.synchronize(device)
            generation_wall = time.perf_counter() - generation_started
            generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            policy_batch, metric_candidate, rollout_record = score_rank_candidate(str(row["route"]), row, generated_ids, tokenizer, rank)
            cleanup_generation_cache(device)
            set_training_mode(ddp.module)
            training_batch = batch_to_device(policy_batch, device)
            torch.cuda.reset_peak_memory_stats(device)
            training_started = time.perf_counter()
            update = distributed_mc_optimizer_step(ddp, optimizer, training_batch, training_batch["credit_units_per_candidate"], forward_batch_size=int(config["forward_batch_size"]))
            torch.cuda.synchronize(device)
            training_wall = time.perf_counter() - training_started
            training_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if update["optimizer_step_performed"]:
                optimizer_step += 1
            local = {
                "rank": rank,
                "candidate_seed": seed,
                "candidate": metric_candidate,
                "rollout": rollout_record,
                "metadata": {key: update["objective_metadata"][key] for key in ("active_unit_count", "active_token_count", "active_token_assignment_count")},
                "generation_wall_seconds": generation_wall,
                "training_wall_seconds": training_wall,
                "generation_peak_vram_mib": generation_peak,
                "training_peak_vram_mib": training_peak,
            }
            gathered = _gather(local)
            if rank == 0:
                candidates = [item["candidate"] for item in sorted(gathered, key=lambda value: value["rank"])]
                record = {
                    "prompt_step": prompt_step,
                    "source_prompt_step": prompt_step,
                    "optimizer_step": optimizer_step,
                    "step": prompt_step,
                    "route": row["route"],
                    "sample_id": row["sample_id"],
                    "candidates": candidates,
                    **candidate_evaluator_means(str(row["route"]), candidates),
                    "active_unit_count": sum(item["metadata"]["active_unit_count"] for item in gathered),
                    "active_token_count": sum(item["metadata"]["active_token_count"] for item in gathered),
                    "active_token_assignment_count": sum(item["metadata"]["active_token_assignment_count"] for item in gathered),
                    "loss": float(update["loss"]),
                    "grad_norm": float(update["grad_norm"]),
                    "skipped_update": bool(update["skipped_update"]),
                    "optimizer_step_performed": bool(update["optimizer_step_performed"]),
                    "generation_wall_seconds": max(item["generation_wall_seconds"] for item in gathered),
                    "training_wall_seconds": max(item["training_wall_seconds"] for item in gathered),
                    "generation_peak_vram_mib": max(item["generation_peak_vram_mib"] for item in gathered),
                    "training_peak_vram_mib": max(item["training_peak_vram_mib"] for item in gathered),
                }
                records.append(record)
                _append_jsonl(metrics_path, record)
                for item in sorted(gathered, key=lambda value: value["rank"]):
                    _append_jsonl(rollouts_path, dict(item["rollout"], prompt_step=prompt_step, source_prompt_step=prompt_step, optimizer_step=optimizer_step, rank=item["rank"], candidate_index=item["rank"], candidate_seed=item["candidate_seed"]))
            if prompt_step in set(config["checkpoint_steps"]):
                dist.barrier()
                _rank_hashes(ddp.module)
                if rank == 0:
                    save_formal_checkpoint(ddp.module, run_dir, prompt_step, optimizer_step, rows[:prompt_step], records, time.perf_counter() - started, selection_seed=int(config["selection_seed"]), config_sha256=preflight["config_sha256"], train_sha256=config["train_sha256"], git_commit=preflight["git_commit"])
                dist.barrier()
        hashes = _rank_hashes(ddp.module)
        if rank == 0:
            final_base_hash = parameter_sha256(ddp.module, lora=False)[0]
            final_lora_hash = parameter_sha256(ddp.module, lora=True)[0]
            summary = summarize_metrics(records, time.perf_counter() - started)
            summary.update({
                "status": "PASS" if optimizer_step == int(config["expected_optimizer_updates"]) else "PASS_WITH_SKIPPED_UPDATES",
                "run_id": args.run_id,
                "parent_run_id": config["parent_run_id"],
                "prompt_step": len(rows),
                "optimizer_step": optimizer_step,
                "expected_optimizer_updates": int(config["expected_optimizer_updates"]),
                "metrics_row_count": len(records),
                "rollout_row_count": len(rows) * K,
                "trainable_tensor_count": len(trainable),
                "lora_tensor_count": hashes[0]["lora_count"],
                "rank_lora_hashes_equal": True,
                "base_hash_unchanged": initial_base_hash == final_base_hash,
                "lora_hash_changed": initial_lora_hash != final_lora_hash,
                "optimizer_state_policy": "fresh_adamw",
                "optimizer_state_lora_only": optimizer_state_is_lora_only(optimizer, trainable),
                "replay_of_seen_samples": True,
                "comparison_contract": "same_sample_order_same_prompt_step_same_candidate_seed_rule",
                "parallelism": "candidate_parallel",
                "K": K,
                "world_size": WORLD_SIZE,
            })
            if not summary["base_hash_unchanged"] or not summary["lora_hash_changed"]:
                raise MCReplayError("Replay100 parameter integrity contract failed")
            _write_json(run_dir / "summary.json", summary)
            return summary
        return None
    except Exception as exc:
        if rank == 0:
            _write_json(run_dir / "summary.json", {"status": "STOPPED", "run_id": args.run_id, "error": f"{type(exc).__name__}: {exc}", "prompt_step": len(records), "optimizer_step": optimizer_step})
        raise
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.execute:
        return execute_distributed(args)
    result = run_preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("READY_TO_EXECUTE")
    return result


if __name__ == "__main__":
    main()
