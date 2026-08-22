#!/usr/bin/env python3
"""Four-GPU candidate-parallel formal trainer for MC_USER hybrid GRPO."""

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from run_mc_user_e2e_smoke import cleanup_generation_cache, set_generation_mode, set_training_mode
from run_mc_user_formal_v1 import (
    FORMAL_RESUME,
    OUTPUT_ROOT,
    _append_jsonl,
    _write_json,
    assert_run_target_writable,
    candidate_evaluator_means,
    git_reproducibility_state,
    save_formal_checkpoint,
    select_formal_rows,
    summarize_metrics,
)
from run_mc_user_generation_smoke import _candidate_overlap_metadata, stop_token_ids
from run_mc_user_pilot_v1 import (
    _selected_gpu_processes,
    assert_only_lora_trainable,
    candidate_record,
    display_rollout_records,
)
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
from run_user_grpo_smoke import render_prompt, trim_at_stop
from user_mc_batch import make_mc_policy_batch
from user_action_reward import score_action
from user_chain_reward import score_chain
from user_mc_hybrid_distributed import distributed_hybrid_optimizer_step
from user_mc_rollout import prepare_mc_scored_rollout
from user_mc_train_step import _gradient_norm


WORLD_SIZE = 4
K = 4
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
ALGORITHM = "mc_user_hybrid_grpo_v1"
OUTPUT_ROOT = Path("/data/GRPO_USER/runs/mc_user_v1_hybrid_formal")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_user_formal_stage1_512_hybrid_k4.json"
FROZEN_CONFIG = {
    "experiment_type": "formal",
    "stage": "stage1_512_hybrid_k4",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": "/data/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500",
    "parent_experiment": "GR_REC_v1",
    "parent_checkpoint_step": 1500,
    "parent_recorded_external_score": 1.3510,
    "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "prompt_count": 512,
    "action_count": 256,
    "chain_count": 256,
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
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [128, 256, 384, 512],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}


class MCK4Error(RuntimeError):
    pass


def load_k4_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in FROZEN_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise MCK4Error(f"frozen K4 config mismatch: {mismatches}")
    return config


def validate_checkpoint_root(path: Path, *, minimum_free_bytes: int = 2 * 1024**3) -> Path:
    declared = str(path).replace("\\", "/")
    if declared == "/data" or declared.startswith("/data/"):
        raise MCK4Error("checkpoint-root must not be under /data")
    root = path.expanduser().resolve()
    if root == Path("/data") or Path("/data") in root.parents:
        raise MCK4Error("checkpoint-root must not be under /data")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise MCK4Error("checkpoint-root is not writable")
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    if free < minimum_free_bytes:
        raise MCK4Error("checkpoint-root has insufficient free space")
    return root


def candidate_seed(selection_seed: int, prompt_step: int, sample_id: str, candidate_index: int) -> int:
    raw = f"{selection_seed}:{prompt_step}:{sample_id}:{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def probe_queue_value() -> dict[str, Any]:
    return {
        "status": "PENDING_TRAINING_CHECKPOINTS",
        "mode": "POST_TRAINING_INFERENCE_ONLY",
        "training_blocked_by_probe": False,
        "items": [
            {"step": step, "label": "BETA" if step == 0 else str(step), "status": "pending" if step == 0 else "waiting", "available_for_probe": step == 0}
            for step in (0, 128, 256, 384, 512)
        ],
    }


def update_probe_queue(queue: Mapping[str, Any], checkpoint_step: int) -> dict[str, Any]:
    updated = json.loads(json.dumps(queue))
    found = False
    for item in updated["items"]:
        if int(item["step"]) == int(checkpoint_step):
            item["status"] = "ready"
            item["available_for_probe"] = True
            found = True
    if not found:
        raise MCK4Error(f"probe queue lacks checkpoint {checkpoint_step}")
    return updated


def _all_gpu_preflight(threshold: int, checker: Callable[..., Mapping[str, Any]] = gpu_preflight) -> list[dict[str, Any]]:
    return [dict(checker(gpu_id, threshold)) for gpu_id in range(WORLD_SIZE)]


def build_manifest(
    run_id: str,
    config_path: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    git_commit: str,
    *,
    smoke_prompts: int = 0,
) -> dict[str, Any]:
    selected = list(rows[:smoke_prompts] if smoke_prompts else rows)
    return {
        "status": "READY_TO_EXECUTE",
        "run_kind": "user_grpo",
        "algorithm": ALGORITHM,
        "experiment_type": "formal_k4_smoke" if smoke_prompts else "formal",
        "stage": "stage1_512_k4_smoke" if smoke_prompts else config["stage"],
        "run_id": run_id,
        "base_model": config["base_model"],
        "adapter": config["adapter"],
        "parent_experiment": config["parent_experiment"],
        "parent_checkpoint_step": config["parent_checkpoint_step"],
        "parent_recorded_external_score": config["parent_recorded_external_score"],
        "train_data": config["train_data"],
        "train_sha256": config["train_sha256"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "prompt_count": len(selected),
        "max_steps": len(selected),
        "action_count": sum(row["route"] == "action" for row in selected),
        "chain_count": sum(row["route"] == "chain" for row in selected),
        "selection_seed": config["selection_seed"],
        "route_schedule": config["route_schedule"],
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_new_tokens": config["max_new_tokens"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "forward_batch_size": config["forward_batch_size"],
        "sequence_weight": config["sequence_weight"],
        "local_weight": config["local_weight"],
        "checkpoint_steps": [] if smoke_prompts else list(config["checkpoint_steps"]),
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
        "FORMAL_RESUME": FORMAL_RESUME,
        "prompts": [
            {"prompt_step": index, "route": row["route"], "sample_id": row["sample_id"]}
            for index, row in enumerate(selected, 1)
        ],
    }


def run_preflight(args: argparse.Namespace, *, gpu_checker: Callable[..., Mapping[str, Any]] = gpu_preflight, git_checker: Callable[..., Mapping[str, Any]] = git_reproducibility_state) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = load_k4_config(config_path)
    paths = validate_paths(Path(config["base_model"]), Path(config["adapter"]), Path(config["train_data"]))
    if paths["train_sha256"] != config["train_sha256"]:
        raise MCK4Error("frozen train SHA mismatch")
    repo_root = Path(__file__).resolve().parents[5]
    git_state = dict(git_checker(repo_root))
    rows = select_formal_rows(read_jsonl(Path(config["train_data"])), int(config["selection_seed"]))
    if args.smoke_prompts not in (0, 2):
        raise MCK4Error("smoke-prompts must be 0 or 2")
    gpus = _all_gpu_preflight(args.memory_threshold_mib, checker=gpu_checker)
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise MCK4Error("run-id contains unsupported characters")
    run_dir = args.output_root / args.run_id
    checkpoint_root = validate_checkpoint_root(args.checkpoint_root)
    assert_run_target_writable(run_dir)
    config_sha256 = file_sha256(config_path)
    manifest = build_manifest(args.run_id, config_path, config_sha256, config, rows, str(git_state["git_commit"]), smoke_prompts=args.smoke_prompts)
    manifest["checkpoint_root"] = str(checkpoint_root)
    manifest["checkpoint_run_dir"] = str(checkpoint_root / args.run_id)
    existing = run_dir / "manifest.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous != manifest:
            raise MCK4Error("BLOCKED_RUN_ID_CONTRACT_MISMATCH")
    _write_json(existing, manifest)
    preflight = {
        "status": "READY_TO_EXECUTE",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "gpus": gpus,
        "git_commit": git_state["git_commit"],
        "working_tree_clean": True,
        "config_sha256": config_sha256,
        "train_sha256": config["train_sha256"],
        "prompt_count": manifest["prompt_count"],
        "action_count": manifest["action_count"],
        "chain_count": manifest["chain_count"],
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "checkpoint_root": str(checkpoint_root),
        "execute_required": True,
    }
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(run_dir / "evaluations" / "user_light_probe" / "probe_queue.json", probe_queue_value())
    return preflight


def _load_execute_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    run_dir = args.output_root / args.run_id
    preflight_path, manifest_path = run_dir / "preflight.json", run_dir / "manifest.json"
    if not preflight_path.is_file() or not manifest_path.is_file():
        raise MCK4Error("execute requires existing preflight.json and manifest.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY_TO_EXECUTE" or manifest.get("git_commit") != preflight.get("git_commit"):
        raise MCK4Error("preflight contract mismatch")
    if manifest.get("K") != K or manifest.get("world_size") != WORLD_SIZE or manifest.get("parallelism") != "candidate_parallel":
        raise MCK4Error("execute manifest is not K4 candidate-parallel")
    checkpoint_root = validate_checkpoint_root(args.checkpoint_root)
    if str(checkpoint_root) != manifest.get("checkpoint_root"):
        raise MCK4Error("execute checkpoint-root differs from preflight")
    config_path = args.config.expanduser().resolve()
    config = load_k4_config(config_path)
    if file_sha256(config_path) != manifest.get("config_sha256"):
        raise MCK4Error("execute config SHA differs from preflight")
    if file_sha256(Path(config["train_data"])) != manifest.get("train_sha256"):
        raise MCK4Error("execute train SHA differs from preflight")
    repo_root = Path(__file__).resolve().parents[5]
    git_state = git_reproducibility_state(repo_root)
    if git_state["git_commit"] != manifest.get("git_commit"):
        raise MCK4Error("execute git commit differs from preflight")
    selected_ids = [item["sample_id"] for item in manifest["prompts"]]
    all_rows = {str(row["sample_id"]): row for row in read_jsonl(Path(config["train_data"]))}
    rows = [dict(all_rows[sample_id]) for sample_id in selected_ids]
    return preflight, config, rows, run_dir


def load_beta_for_rank(config: Mapping[str, Any], local_rank: int) -> torch.nn.Module:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, config["adapter"], is_trainable=True, local_files_only=True)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    return model


@torch.inference_mode()
def generate_one_candidate(model: torch.nn.Module, tokenizer: Any, row: Mapping[str, Any], device: torch.device, seed: int, config: Mapping[str, Any]) -> list[int]:
    prompt = render_prompt(tokenizer, row)
    encoded = tokenizer([prompt], add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt")
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    stops = stop_token_ids(tokenizer)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            num_return_sequences=1,
            max_new_tokens=int(config["max_new_tokens"]),
            eos_token_id=sorted(stops),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return trim_at_stop(output[0, prompt_width:].cpu().tolist(), stops)


def score_rank_candidate(route: str, row: Mapping[str, Any], generated_ids: Sequence[int], tokenizer: Any, candidate_index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rollout = prepare_mc_scored_rollout([row], [generated_ids], tokenizer, candidates_per_prompt=1)
    policy_batch = make_mc_policy_batch(tokenizer, rollout)
    marginal = rollout["marginal_results"][0]
    units = rollout["credit_units_per_candidate"][0]
    overlap = _candidate_overlap_metadata(units, len(generated_ids))
    raw = {
        "candidate_index": candidate_index,
        "completion": rollout["completions"][0],
        "generated_token_count": len(generated_ids),
        "format_valid": bool(marginal["valid"]),
        "full_reward": float(marginal["full_reward"]),
        "projection_required": bool(rollout["projection_per_candidate"][0]["projection_required"]),
        **{key: int(overlap[key]) for key in ("overlap_token_count", "same_sign_overlap_token_count", "mixed_sign_overlap_token_count", "max_active_units_per_token")},
    }
    if route == "action":
        evaluator = score_action(raw["completion"], dict(row))
        raw.update(f1=float(evaluator.f1), precision=float(evaluator.precision), recall=float(evaluator.recall), predicted_sid_count=len(evaluator.pred_sids_unique))
    else:
        evaluator = score_chain(raw["completion"], dict(row))
        raw["full_action_alignment"] = float(evaluator.action_f1)
        raw["full_logic_alignment"] = float(evaluator.logic_f1)
        raw["predicted_event_count"] = len(evaluator.predicted_events)
    metric_candidate = candidate_record(raw, units, marginal, route)
    rollout_record = display_rollout_records(prompt_step=0, optimizer_step=0, route=route, sample_id=str(row["sample_id"]), candidates=[raw], units_per_candidate=[units])[0]
    if route == "action":
        evaluator_fields = {
            "precision": raw["precision"],
            "recall": raw["recall"],
            "predicted_sid_count": raw["predicted_sid_count"],
        }
    else:
        evaluator_fields = {"predicted_event_count": raw["predicted_event_count"]}
    metric_candidate.update(evaluator_fields)
    rollout_record.update(evaluator_fields)
    return policy_batch, metric_candidate, rollout_record


def _gather(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    return [dict(item) for item in gathered]


def append_rank0_prompt_artifacts(
    rank: int,
    metrics_path: Path,
    rollouts_path: Path,
    metric: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
) -> None:
    if rank != 0:
        return
    _append_jsonl(metrics_path, metric)
    for rollout in rollouts:
        _append_jsonl(rollouts_path, rollout)


def validate_ddp_owner_claims(claims: Sequence[Mapping[str, Any]]) -> None:
    if len(claims) != WORLD_SIZE or any(not claim.get("nvidia_host_pids") for claim in claims):
        raise MCK4Error("GPU_OWNER_NOT_VISIBLE")
    pid_sets = [set(map(int, claim["nvidia_host_pids"])) for claim in claims]
    if len(set().union(*pid_sets)) != WORLD_SIZE:
        raise MCK4Error("GPU_FOREIGN_PROCESS_AFTER_LOAD")


def claim_ddp_gpu_process_ownership(gpu_id: int) -> dict[str, Any]:
    gpu_uuid, processes = _selected_gpu_processes(gpu_id)
    local_claim = {
        "gpu_id": gpu_id,
        "gpu_uuid": gpu_uuid,
        "nvidia_host_pids": sorted(int(item["nvidia_host_pid"]) for item in processes),
        "processes": processes,
    }
    claims = _gather(local_claim)
    validate_ddp_owner_claims(claims)
    return local_claim


def assert_ddp_gpu_process_owned(claim: Mapping[str, Any]) -> None:
    gpu_uuid, processes = _selected_gpu_processes(int(claim["gpu_id"]))
    current_pids = sorted(int(item["nvidia_host_pid"]) for item in processes)
    if gpu_uuid != claim["gpu_uuid"] or current_pids != list(claim["nvidia_host_pids"]):
        raise MCK4Error("GPU_OWNERSHIP_CHANGED")


def _rank_hashes(model: torch.nn.Module) -> list[dict[str, Any]]:
    lora_hash, lora_count = parameter_sha256(model, lora=True)
    gathered = _gather({"lora_hash": lora_hash, "lora_count": lora_count})
    if len({item["lora_hash"] for item in gathered}) != 1:
        raise MCK4Error("rank LoRA parameters diverged")
    return gathered


def execute_distributed(args: argparse.Namespace) -> dict[str, Any] | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise MCK4Error("execute requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise MCK4Error("execute requires torchrun world_size=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank not in range(WORLD_SIZE):
        raise MCK4Error("LOCAL_RANK must be 0..3")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # This is a single-node job. Binding bootstrap traffic to loopback avoids
    # the development container's unroutable external interface.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    preflight, config, rows, run_dir = _load_execute_contract(args)
    checkpoint_run_dir = Path(preflight["checkpoint_root"]) / args.run_id
    dist.barrier()
    tokenizer = load_tokenizer(config["base_model"])
    model = load_beta_for_rank(config, local_rank)
    trainable = validate_trainable(model)
    if len(trainable) != 504:
        raise MCK4Error("K4 formal requires 504 LoRA trainable tensors")
    dist.barrier()
    owner = claim_ddp_gpu_process_ownership(local_rank)
    initial_base_hash = parameter_sha256(model, lora=False)[0] if rank == 0 else None
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    _rank_hashes(ddp.module)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    records: list[dict[str, Any]] = []
    optimizer_step = 0
    started = time.perf_counter()
    metrics_path, rollouts_path = run_dir / "metrics.jsonl", run_dir / "rollouts.jsonl"
    queue_path = run_dir / "evaluations" / "user_light_probe" / "probe_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
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
            update = distributed_hybrid_optimizer_step(
                ddp,
                optimizer,
                training_batch,
                metric_candidate["reward"],
                training_batch["credit_units_per_candidate"],
                sequence_weight=float(config["sequence_weight"]),
                local_weight=float(config["local_weight"]),
                forward_batch_size=int(config["forward_batch_size"]),
            )
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
                "metadata": {
                    "active_unit_count": update["local_objective_metadata"]["active_unit_count"],
                    "active_token_count": update["local_objective_metadata"]["active_token_count"],
                    "active_token_assignment_count": update["local_objective_metadata"]["active_token_assignment_count"],
                },
                "hybrid": {key: update[key] for key in (
                    "local_reward", "group_reward_mean", "group_reward_std",
                    "group_reward_spread", "sequence_advantage", "sequence_loss",
                    "local_loss", "total_loss",
                )},
                "grad_norm": float(update["grad_norm"]),
                "generation_wall_seconds": generation_wall,
                "training_wall_seconds": training_wall,
                "generation_peak_vram_mib": generation_peak,
                "training_peak_vram_mib": training_peak,
            }
            gathered = _gather(local)
            if rank == 0:
                candidates = [item["candidate"] for item in sorted(gathered, key=lambda value: value["rank"])]
                for candidate, item in zip(candidates, sorted(gathered, key=lambda value: value["rank"])):
                    candidate.update(item["hybrid"])
                evaluator = candidate_evaluator_means(str(row["route"]), candidates)
                hybrid_mean = {
                    key: sum(float(item["hybrid"][key]) for item in gathered) / K
                    for key in ("sequence_loss", "local_loss", "total_loss")
                }
                if row["route"] == "action":
                    route_means = {
                        "candidate_mean_precision": sum(float(item["precision"]) for item in candidates) / K,
                        "candidate_mean_recall": sum(float(item["recall"]) for item in candidates) / K,
                        "candidate_mean_sid_count": sum(float(item["predicted_sid_count"]) for item in candidates) / K,
                        "candidate_mean_total": None,
                        "candidate_mean_event_count": None,
                    }
                else:
                    route_means = {
                        "candidate_mean_precision": None,
                        "candidate_mean_recall": None,
                        "candidate_mean_sid_count": None,
                        "candidate_mean_total": sum(float(item["reward"]) for item in candidates) / K,
                        "candidate_mean_event_count": sum(float(item["predicted_event_count"]) for item in candidates) / K,
                    }
                record = {
                    "prompt_step": prompt_step,
                    "optimizer_step": optimizer_step,
                    "step": prompt_step,
                    "route": row["route"],
                    "sample_id": row["sample_id"],
                    "candidates": candidates,
                    **evaluator,
                    "active_unit_count": sum(item["metadata"]["active_unit_count"] for item in gathered),
                    "active_token_count": sum(item["metadata"]["active_token_count"] for item in gathered),
                    "active_token_assignment_count": sum(item["metadata"]["active_token_assignment_count"] for item in gathered),
                    "algorithm": ALGORITHM,
                    "group_reward_mean": float(update["group_reward_mean"]),
                    "group_reward_std": float(update["group_reward_std"]),
                    "group_reward_min": min(float(item["reward"]) for item in candidates),
                    "group_reward_max": max(float(item["reward"]) for item in candidates),
                    "group_reward_spread": float(update["group_reward_spread"]),
                    **hybrid_mean,
                    **route_means,
                    "loss": hybrid_mean["total_loss"],
                    "grad_norm": float(update["grad_norm"]),
                    "skipped_update": bool(update["skipped_update"]),
                    "optimizer_step_performed": bool(update["optimizer_step_performed"]),
                    "generation_wall_seconds": max(item["generation_wall_seconds"] for item in gathered),
                    "training_wall_seconds": max(item["training_wall_seconds"] for item in gathered),
                    "generation_peak_vram_mib": max(item["generation_peak_vram_mib"] for item in gathered),
                    "training_peak_vram_mib": max(item["training_peak_vram_mib"] for item in gathered),
                }
                records.append(record)
                display_rows = []
                for item in sorted(gathered, key=lambda value: value["rank"]):
                    display = dict(item["rollout"], prompt_step=prompt_step, optimizer_step=optimizer_step, rank=item["rank"], candidate_index=item["rank"], candidate_seed=item["candidate_seed"], **item["hybrid"])
                    display["completion_token_count"] = display["generated_token_count"]
                    display["reward"] = display["full_reward"]
                    if display["route"] == "chain":
                        display["total_reward"] = display["full_reward"]
                        display["action_alignment"] = display["full_action_alignment"]
                        display["logic_alignment"] = display["full_logic_alignment"]
                    display_rows.append(display)
                append_rank0_prompt_artifacts(rank, metrics_path, rollouts_path, record, display_rows)
            if prompt_step in set(config["checkpoint_steps"]) and not args.smoke_prompts:
                dist.barrier()
                _rank_hashes(ddp.module)
                if rank == 0:
                    save_formal_checkpoint(ddp.module, checkpoint_run_dir, prompt_step, optimizer_step, rows[:prompt_step], records, time.perf_counter() - started, selection_seed=int(config["selection_seed"]), config_sha256=preflight["config_sha256"], train_sha256=config["train_sha256"], git_commit=preflight["git_commit"])
                    queue = update_probe_queue(queue, prompt_step)
                    _write_json(queue_path, queue)
                dist.barrier()
        hashes = _rank_hashes(ddp.module)
        if rank == 0:
            final_base_hash = parameter_sha256(ddp.module, lora=False)[0]
            summary = summarize_metrics(records, time.perf_counter() - started)
            summary.update({
                "status": "PASS",
                "run_id": args.run_id,
                "prompt_step": len(rows),
                "optimizer_step": optimizer_step,
                "metrics_row_count": len(records),
                "rollout_row_count": len(rows) * K,
                "trainable_tensor_count": len(trainable),
                "lora_tensor_count": hashes[0]["lora_count"],
                "rank_lora_hashes_equal": True,
                "base_hash_unchanged": initial_base_hash == final_base_hash,
                "optimizer_state_lora_only": optimizer_state_is_lora_only(optimizer, trainable),
                "parallelism": "candidate_parallel",
                "K": K,
                "world_size": WORLD_SIZE,
                "algorithm": ALGORITHM,
                "checkpoint_root": preflight["checkpoint_root"],
                "sequence_weight": config["sequence_weight"],
                "local_weight": config["local_weight"],
            })
            if not summary["base_hash_unchanged"]:
                raise MCK4Error("base parameters changed")
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
    parser.add_argument("--smoke-prompts", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
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
