#!/usr/bin/env python3
"""Four-GPU same-prompt candidate-parallel formal trainer for MC_USER_v1."""

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
from user_mc_policy import compute_mc_model_loss
from user_mc_rollout import prepare_mc_scored_rollout
from user_mc_train_step import _gradient_norm


WORLD_SIZE = 4
K = 4
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_user_formal_stage1_512_k4_ddp.json"
FROZEN_CONFIG = {
    "experiment_type": "formal",
    "stage": "stage1_512_k4",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": (
        "/data/outputs/baselines/native_source_domain_r32_v3/"
        "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/"
        "checkpoint-1106"
    ),
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
        "algorithm": "mc_user_v1",
        "experiment_type": "formal_k4_smoke" if smoke_prompts else "formal",
        "stage": "stage1_512_k4_smoke" if smoke_prompts else config["stage"],
        "run_id": run_id,
        "base_model": config["base_model"],
        "adapter": config["adapter"],
        "train_data": config["train_data"],
        "train_sha256": config["train_sha256"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "prompt_count": len(selected),
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
    assert_run_target_writable(run_dir)
    config_sha256 = file_sha256(config_path)
    manifest = build_manifest(args.run_id, config_path, config_sha256, config, rows, str(git_state["git_commit"]), smoke_prompts=args.smoke_prompts)
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
        raw["f1"] = float(marginal["full_reward"])
    else:
        raw["full_action_alignment"] = float(marginal.get("full_action_alignment", 0.0))
        raw["full_logic_alignment"] = float(marginal.get("full_logic_alignment", 0.0))
    metric_candidate = candidate_record(raw, units, marginal, route)
    rollout_record = display_rollout_records(prompt_step=0, optimizer_step=0, route=route, sample_id=str(row["sample_id"]), candidates=[raw], units_per_candidate=[units])[0]
    return policy_batch, metric_candidate, rollout_record


def distributed_mc_optimizer_step(ddp_model: DistributedDataParallel, optimizer: torch.optim.Optimizer, batch: Mapping[str, torch.Tensor], units: Sequence[Sequence[Mapping[str, Any]]], *, forward_batch_size: int = 1) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    loss, metadata, per_token_logps = compute_mc_model_loss(ddp_model, batch, units, forward_batch_size=forward_batch_size)
    if not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(per_token_logps).all()):
        raise MCK4Error("non-finite distributed loss/logps")
    device = loss.device
    local_active = torch.tensor(int(metadata["active_unit_count"]), device=device, dtype=torch.long)
    global_active = local_active.clone()
    dist.all_reduce(global_active, op=dist.ReduceOp.SUM)
    reduced_loss = loss.detach().double()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss /= dist.get_world_size()
    if int(global_active.item()) == 0:
        return {"loss": float(reduced_loss), "grad_norm": 0.0, "finite": True, "objective_metadata": metadata, "global_active_unit_count": 0, "skipped_update": True, "optimizer_step_performed": False}
    loss.backward()
    parameters = [parameter for parameter in ddp_model.module.parameters() if parameter.requires_grad]
    grad_norm = _gradient_norm(parameters)
    grad_bounds = torch.tensor([grad_norm, grad_norm], device=device, dtype=torch.float64)
    dist.all_reduce(grad_bounds[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(grad_bounds[1:], op=dist.ReduceOp.MAX)
    grad_norm_spread = float(grad_bounds[1] - grad_bounds[0])
    if grad_norm_spread > 1e-6:
        raise MCK4Error(f"DDP gradient norms diverged across ranks: {grad_norm_spread}")
    optimizer.step()
    return {"loss": float(reduced_loss), "grad_norm": grad_norm, "grad_norm_spread": grad_norm_spread, "finite": True, "objective_metadata": metadata, "global_active_unit_count": int(global_active.item()), "skipped_update": False, "optimizer_step_performed": True}


def _gather(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    return [dict(item) for item in gathered]


def validate_ddp_owner_claims(claims: Sequence[Mapping[str, Any]]) -> None:
    if len(claims) != WORLD_SIZE or any(not claim.get("nvidia_host_pids") for claim in claims):
        raise MCK4Error("GPU_OWNER_NOT_VISIBLE")
    pid_sets = [set(map(int, claim["nvidia_host_pids"])) for claim in claims]
    one_process_per_gpu = all(len(pids) == 1 for pids in pid_sets) and len(
        {next(iter(pids)) for pids in pid_sets}
    ) == WORLD_SIZE
    all_ranks_visible_per_gpu = all(pids == pid_sets[0] for pids in pid_sets) and len(
        pid_sets[0]
    ) == WORLD_SIZE
    if not one_process_per_gpu and not all_ranks_visible_per_gpu:
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
                "grad_norm": float(update["grad_norm"]),
                "generation_wall_seconds": generation_wall,
                "training_wall_seconds": training_wall,
                "generation_peak_vram_mib": generation_peak,
                "training_peak_vram_mib": training_peak,
            }
            gathered = _gather(local)
            if rank == 0:
                candidates = [item["candidate"] for item in sorted(gathered, key=lambda value: value["rank"])]
                evaluator = candidate_evaluator_means(str(row["route"]), candidates)
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
                    display = dict(item["rollout"], prompt_step=prompt_step, optimizer_step=optimizer_step, rank=item["rank"], candidate_index=item["rank"], candidate_seed=item["candidate_seed"])
                    _append_jsonl(rollouts_path, display)
            if prompt_step in set(config["checkpoint_steps"]) and not args.smoke_prompts:
                dist.barrier()
                _rank_hashes(ddp.module)
                if rank == 0:
                    save_formal_checkpoint(ddp.module, run_dir, prompt_step, optimizer_step, rows[:prompt_step], records, time.perf_counter() - started, selection_seed=int(config["selection_seed"]), config_sha256=preflight["config_sha256"], train_sha256=config["train_sha256"], git_commit=preflight["git_commit"])
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
