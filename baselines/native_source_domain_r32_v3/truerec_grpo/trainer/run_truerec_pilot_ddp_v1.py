"""Four-GPU rank-sharded production entrypoint for one logical TrueRec G8 group."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY  # noqa: E402
from beta_single_group_loss_audit import lora_parameter_sha  # noqa: E402
from beta_streaming_backward_audit import gradient_audit  # noqa: E402
from checkpoint_ddp_v1 import load_distributed_checkpoint, save_distributed_checkpoint  # noqa: E402
from checkpoint_v1 import DEFAULT_ORDER_SEED  # noqa: E402
from distributed_trainer_v1 import (  # noqa: E402
    DDP_WORLD_SIZE, LOCAL_G, DistributedTrueRecGRPOTrainerV1,
    all_rank_values_equal, distributed_fail_if,
)
from policy_scoring_v1 import parameter_versions, score_full_sequences  # noqa: E402
from production_memory_hardening import WORST_CONTEXT_TOKEN_COUNT, WORST_DOMAIN, WORST_GROUP_ID  # noqa: E402
from real_three_group_resume_smoke import (  # noqa: E402
    DATASET_IDENTITY, LoraCheckpointState, ORDER_SHA256, base_parameter_versions,
    load_pilot_order, load_runtime, optimizer_step_value,
)
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, GENERATION_KWARGS,
    GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    build_rescored_rollout_group, extract_generation_artifacts,
)
from run_truerec_pilot_v1 import select_streaming_microbatch_size  # noqa: E402


NORMAL_GROUP_ID = "2652ca78a590e57a5af87ab38557d9c03e15c4cfc3f04e907f9effd4ddaa6643"
NORMAL_CONTEXT_TOKENS = 1735
RUN_SEED = 440104
MIN_FREE_GIB = 70.0


class DDPProductionError(RuntimeError):
    pass


def seed_rank(rank: int) -> None:
    seed = RUN_SEED + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def gather_rank_objects(value: Any) -> list[Any]:
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def global_ratio_stats(local_current, local_candidates, device: torch.device) -> dict[str, float | int]:
    differences = []
    ratios = []
    for row, candidate in zip(local_current, local_candidates):
        for current, old in zip(row, candidate.old_logps):
            differences.append(abs(float(current) - float(old)))
            ratios.append(math.exp(float(current) - float(old)))
    sums = torch.tensor([sum(differences), sum(ratios), len(ratios)], dtype=torch.float64, device=device)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    extrema = torch.tensor([max(differences), min(ratios), max(ratios)], dtype=torch.float64, device=device)
    maximum = extrema[:1].clone(); minimum_ratio = extrema[1:2].clone(); maximum_ratio = extrema[2:].clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(minimum_ratio, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum_ratio, op=dist.ReduceOp.MAX)
    count = int(sums[2].item())
    return {
        "abs_mean": float(sums[0].item() / count), "abs_max": float(maximum.item()),
        "ratio_mean": float(sums[1].item() / count), "ratio_min": float(minimum_ratio.item()),
        "ratio_max": float(maximum_ratio.item()), "token_count": count,
    }


def optimizer_driver_state() -> dict[str, int | bool]:
    return {
        "business_groups_seen": 1, "optimizer_steps": 1, "global_step": 1,
        "rollouts_completed": 1, "old_rescores_completed": 1,
        "streaming_backwards_completed": 1, "groups_in_accumulation_window": 0, "failed": False,
    }


def initialize() -> tuple[int, torch.device, dict[str, Any]]:
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != DDP_WORLD_SIZE:
        raise DDPProductionError("production runner requires exactly four ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    resource = {
        "rank": rank, "local_rank": local_rank, "gpu_name": torch.cuda.get_device_name(device),
        "free_memory_at_start_gb": free / (1024 ** 3), "total_memory_gb": total / (1024 ** 3),
    }
    distributed_fail_if(resource["free_memory_at_start_gb"] < MIN_FREE_GIB, "a rank lacks 70 GiB free memory", device)
    seed_rank(rank)
    return rank, device, resource


def run_one_group(group_id: str, expected_context: int, expected_mb: int, device: torch.device) -> tuple[dict[str, Any], Any, Any]:
    rank = dist.get_rank()
    records, order, order_info = load_pilot_order()
    record = records[group_id]
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    ddp.eval(); model.eval()
    context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
    selected_mb = select_streaming_microbatch_size(len(context_ids))
    identity_ok = (
        len(context_ids) == expected_context and selected_mb == expected_mb
        and all_rank_values_equal(group_id) and all_rank_values_equal(selected_mb)
    )
    distributed_fail_if(not identity_ok, "group/context/microbatch agreement failed", device)
    torch.cuda.reset_peak_memory_stats(device)
    versions = parameter_versions(model)
    local_generation_kwargs = dict(GENERATION_KWARGS); local_generation_kwargs["num_return_sequences"] = LOCAL_G
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        eos_token_id=list(FORMAL_EOS_TOKEN_IDS), pad_token_id=FORMAL_PAD_TOKEN_ID,
        return_dict_in_generate=True, output_scores=True, **local_generation_kwargs,
    )
    artifacts = extract_generation_artifacts(context_ids, output, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
    distributed_fail_if(parameter_versions(model) != versions, "policy changed during distributed rollout", device)
    gathered_generation = gather_rank_objects({
        "completion_ids": artifacts.completion_ids,
        "generation_score_logps": artifacts.generation_score_logps,
    })
    global_completion_ids = tuple(row for payload in gathered_generation for row in payload["completion_ids"])
    global_generation_logps = tuple(row for payload in gathered_generation for row in payload["generation_score_logps"])
    distributed_fail_if(len(global_completion_ids) != 8, "global completion count is not eight", device)

    local_old = score_full_sequences(
        model, context_ids, artifacts.completion_ids, FORMAL_PAD_TOKEN_ID, device,
        grad_enabled=False, trainable_parameters=(), scoring_microbatch_size=selected_mb,
    )
    gathered_old = gather_rank_objects(local_old.logps)
    global_old = tuple(row for rank_rows in gathered_old for row in rank_rows)
    group = build_rescored_rollout_group(
        record, context_ids, global_completion_ids, global_old, global_generation_logps,
        renderer.tokenizer.convert_ids_to_tokens,
    )
    distributed_fail_if(
        tuple(candidate.sample_index for candidate in group.candidates) != tuple(range(8)),
        "global candidate indices are not rank-major 0..7", device,
    )

    lora_before, _ = lora_parameter_sha(model)
    base_before = base_parameter_versions(model)
    optimizer.zero_grad(set_to_none=True)
    trainer = DistributedTrueRecGRPOTrainerV1(
        ddp, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID,
        device=device, streaming_microbatch_size=selected_mb,
    )
    backward = trainer.backward_global_group(group)
    local_start = rank * LOCAL_G
    local_candidates = group.candidates[local_start:local_start + LOCAL_G]
    ratio = global_ratio_stats(backward.local_current_logps, local_candidates, device)
    gradients = gradient_audit(model)
    expected_calls = LOCAL_G // selected_mb
    loss_finite = all(math.isfinite(value) for value in (
        backward.global_frontier_value, backward.global_hpr_value_raw,
        backward.global_hpr_value_weighted, backward.global_total_value,
    ))
    gradient_finite = (
        gradients["lora_params_with_nonzero_grad"] > 0 and math.isfinite(gradients["lora_grad_norm"])
        and gradients["lora_grad_norm"] > 0 and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == gradients["inf_grad_count"] == 0
    )
    local_failure = not (
        ratio["abs_max"] <= 1e-6 and backward.physical_forward_calls == expected_calls
        and backward.physical_backward_calls == expected_calls and backward.hpr_extra_forward_calls == 0
        and loss_finite and gradient_finite and all_rank_values_equal(backward.runtime_plan_hash)
    )
    distributed_fail_if(local_failure, "pre-optimizer distributed gate failed", device)
    optimizer.step()
    torch.cuda.synchronize(device)
    lora_after, _ = lora_parameter_sha(model)
    base_mutation = base_parameter_versions(model) != base_before
    lora_hashes = gather_rank_objects(lora_after)
    distributed_fail_if(lora_before == lora_after or base_mutation or len(set(lora_hashes)) != 1, "parameter update agreement failed", device)
    local_memory = {
        "rank": rank,
        "rank_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "rank_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "rank_allocated_after_group_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
    }
    rank_memory = gather_rank_objects(local_memory)
    report = {
        "status": "PASS", "model_family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT),
        "provenance": provenance, "group_id": group_id, "context_token_count": len(context_ids),
        "world_size": DDP_WORLD_SIZE, "global_G": 8, "local_G_per_rank": LOCAL_G,
        "global_candidate_indices": list(range(8)), "selected_microbatch_size": selected_mb,
        "old_current_microbatch_matched": True, "local_old_forward_calls": expected_calls,
        "local_current_forward_calls": backward.physical_forward_calls,
        "local_current_backward_calls": backward.physical_backward_calls,
        "hpr_extra_forward_calls": backward.hpr_extra_forward_calls,
        "runtime_plan_hash": backward.runtime_plan_hash,
        "ratio": ratio,
        "loss": {
            "frontier": backward.global_frontier_value, "hpr_raw": backward.global_hpr_value_raw,
            "hpr_weighted": backward.global_hpr_value_weighted, "total": backward.global_total_value,
            "finite": loss_finite,
        },
        "gradient": gradients, "gradient_finite": gradient_finite,
        "lora_changed": lora_before != lora_after, "base_parameter_mutation": base_mutation,
        "optimizer_steps": 1, "optimizer": optimizer_audit, "dropout_active": dropout["rl_dropout_active"],
        "trainability": trainability, "rank_memory": rank_memory,
        "max_rank_peak_allocated_gb": max(item["rank_peak_allocated_gb"] for item in rank_memory),
        "max_rank_peak_reserved_gb": max(item["rank_peak_reserved_gb"] for item in rank_memory),
        "OOM": False, "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "order": order_info, "formal_pilot4096_started": False, "evaluation_started": False,
        "next_experiment_started": False,
    }
    return report, model, optimizer


def atomic_rank0_json(path: Path, value: Any) -> None:
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("normal-save", "normal-resume", "worst"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    rank, device, resource = initialize()
    try:
        if args.scenario == "normal-save":
            records, order, order_info = load_pilot_order()
            distributed_fail_if(order[0] != NORMAL_GROUP_ID, "normal group is not frozen order index zero", device)
            report, model, optimizer = run_one_group(NORMAL_GROUP_ID, NORMAL_CONTEXT_TOKENS, 2, device)
            checkpoint = save_distributed_checkpoint(
                args.checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer,
                driver_state=optimizer_driver_state(), dataset_identity=DATASET_IDENTITY,
                epoch=0, next_group_index=1, order=order_info, device=device,
            )
            report.update({"checkpoint_saved": True, "checkpoint": checkpoint, "resource_at_start": resource})
            atomic_rank0_json(args.output_dir / "normal_process_a.json", report)
        elif args.scenario == "normal-resume":
            records, order, _ = load_pilot_order()
            model, optimizer, _, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
            restored = load_distributed_checkpoint(
                args.checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer,
                current_dataset_identity=DATASET_IDENTITY, current_group_ids=list(records), device=device,
            )
            local = {
                "rank": rank, "rng_restored": restored["rng_restored"],
                "model_restore_exact": restored["model_restore_exact"],
                "optimizer_step": optimizer_step_value(optimizer),
            }
            ranks = gather_rank_objects(local)
            passed = (
                restored["global_step"] == 1 and restored["next_group_index"] == 1
                and all(item["model_restore_exact"] for item in ranks)
                and all(all(item["rng_restored"].values()) for item in ranks)
                and all(item["optimizer_step"] == 1 for item in ranks)
            )
            distributed_fail_if(not passed, "distributed checkpoint restore gate failed", device)
            report = {
                "status": "PASS", "world_size": 4, "global_step": 1, "cursor": 1,
                "model_restore_exact": True, "adamw_restore": True, "all_rank_rng_restore": True,
                "rank_restore": ranks, "provenance": provenance, "dropout_active": dropout["rl_dropout_active"],
                "trainability": trainability, "optimizer": optimizer_audit,
                "group_index_1_started": False, "optimizer_steps_after_resume": 0,
                "formal_pilot4096_started": False, "next_experiment_started": False,
            }
            atomic_rank0_json(args.output_dir / "normal_resume_summary.json", report)
        else:
            report, _, _ = run_one_group(WORST_GROUP_ID, WORST_CONTEXT_TOKEN_COUNT, 1, device)
            report["resource_at_start"] = resource
            report["checkpoint_saved"] = False
            atomic_rank0_json(args.output_dir / "worst_context_summary.json", report)
    finally:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
