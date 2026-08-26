"""Four-GPU rank-sharded production entrypoint for one logical TrueRec G8 group."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

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
from monitoring_v1 import gather_detached_monitoring, local_payload_from_backward  # noqa: E402
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


def build_pre_optimizer_gate_diagnostic(
    *, ratio: dict[str, float | int], selected_mb: int, expected_calls: int,
    losses: dict[str, float], rank_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe the existing pre-optimizer gate using detached scalar state."""
    loss_finite = {name: math.isfinite(float(value)) for name, value in losses.items()}
    runtime_hashes = [str(state["runtime_plan_hash"]) for state in rank_states]
    runtime_equal = len(set(runtime_hashes)) == 1
    failed: set[str] = set()
    if float(ratio["abs_max"]) > 1e-6:
        failed.add("RATIO_ABS_MAX")
    for state in rank_states:
        calls = state["calls"]
        if int(calls["forward_calls"]) != expected_calls:
            failed.add("FORWARD_CALL_COUNT")
        if int(calls["backward_calls"]) != expected_calls:
            failed.add("BACKWARD_CALL_COUNT")
        if int(calls["hpr_extra_forward_calls"]) != 0:
            failed.add("HPR_EXTRA_FORWARD_CALLS")
        gradient = state["gradient"]
        norm = float(gradient["lora_grad_norm"])
        if not math.isfinite(norm):
            failed.add("LORA_GRAD_NORM_NONFINITE")
        elif int(gradient["lora_params_with_nonzero_grad"]) == 0 or norm <= 0:
            failed.add("ZERO_GRADIENT")
        if int(gradient["base_params_with_grad"]) != 0:
            failed.add("BASE_GRADIENT")
        if int(gradient["nan_grad_count"]) != 0:
            failed.add("NAN_GRADIENT")
        if int(gradient["inf_grad_count"]) != 0:
            failed.add("INF_GRADIENT")
    if not all(loss_finite.values()):
        failed.add("LOSS_NONFINITE")
    if not runtime_equal:
        failed.add("RUNTIME_PLAN_HASH_MISMATCH")
    call_rows = [dict(state["calls"], rank=int(state["rank"])) for state in rank_states]
    gradient_rows = [dict(state["gradient"], rank=int(state["rank"])) for state in rank_states]
    return {
        "ratio_gate": {
            "abs_mean": float(ratio["abs_mean"]), "abs_max": float(ratio["abs_max"]),
            "ratio_min": float(ratio["ratio_min"]), "ratio_max": float(ratio["ratio_max"]),
            "pass": float(ratio["abs_max"]) <= 1e-6,
        },
        "call_gate": {
            "selected_mb": int(selected_mb), "expected_calls": int(expected_calls),
            "forward_calls": [row["forward_calls"] for row in call_rows],
            "backward_calls": [row["backward_calls"] for row in call_rows],
            "hpr_extra_forward_calls": [row["hpr_extra_forward_calls"] for row in call_rows],
            "per_rank": call_rows,
            "pass": not failed.intersection({
                "FORWARD_CALL_COUNT", "BACKWARD_CALL_COUNT", "HPR_EXTRA_FORWARD_CALLS",
            }),
        },
        "loss_gate": {
            **{name: float(value) for name, value in losses.items()},
            "finite": loss_finite, "pass": all(loss_finite.values()),
        },
        "gradient_gate": {
            **{
                name: [row[name] for row in gradient_rows]
                for name in (
                    "lora_params_with_grad", "lora_params_with_nonzero_grad", "lora_grad_norm",
                    "base_params_with_grad", "nan_grad_count", "inf_grad_count",
                )
            },
            "per_rank": gradient_rows,
            "pass": not failed.intersection({
                "LORA_GRAD_NORM_NONFINITE", "ZERO_GRADIENT", "BASE_GRADIENT",
                "NAN_GRADIENT", "INF_GRADIENT",
            }),
        },
        "runtime_plan_gate": {
            "each_rank_runtime_plan_hash": runtime_hashes,
            "all_equal": runtime_equal, "pass": runtime_equal,
        },
        "EXACT_FAILED_SUBGATES": sorted(failed),
        "pass": not failed,
    }


def optimizer_driver_state() -> dict[str, int | bool]:
    return {
        "business_groups_seen": 1, "optimizer_steps": 1, "global_step": 1,
        "rollouts_completed": 1, "old_rescores_completed": 1,
        "streaming_backwards_completed": 1, "groups_in_accumulation_window": 0, "failed": False,
    }


def distributed_restore_gate(restored: dict[str, Any], ranks: list[dict[str, Any]]) -> bool:
    return (
        restored["driver_state"]["global_step"] == 1
        and restored["next_group_index"] == 1
        and all(item["model_restore_exact"] for item in ranks)
        and all(all(item["rng_restored"].values()) for item in ranks)
        and all(item["optimizer_step"] == 1 for item in ranks)
    )


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


def run_loaded_group(
    group_id: str, record: dict[str, Any], model, ddp, optimizer, renderer,
    device: torch.device, *, expected_context: int | None = None,
    expected_mb: int | None = None, provenance: dict[str, Any] | None = None,
    dropout: dict[str, Any] | None = None, trainability: dict[str, Any] | None = None,
    optimizer_audit: dict[str, Any] | None = None, strict_parameter_audit: bool = False,
    pre_optimizer_diagnostic_callback: Callable[[dict[str, Any], Any, Any, list[dict[str, Any]]], None] | None = None,
    stop_before_optimizer: bool = False,
) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]] | None]:
    """Execute the already-gated global-G8 update against a long-lived DDP policy."""
    rank = dist.get_rank()
    ddp.eval(); model.eval()
    started = time.perf_counter()
    context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
    selected_mb = select_streaming_microbatch_size(len(context_ids))
    identity_ok = (
        (expected_context is None or len(context_ids) == expected_context)
        and (expected_mb is None or selected_mb == expected_mb)
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

    lora_before = base_before = None
    if strict_parameter_audit:
        lora_before, _ = lora_parameter_sha(model)
        base_before = base_parameter_versions(model)
    optimizer.zero_grad(set_to_none=True)
    trainer = DistributedTrueRecGRPOTrainerV1(
        ddp, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID,
        device=device, streaming_microbatch_size=selected_mb,
    )
    backward = trainer.backward_global_group(group)
    rank_monitoring = gather_detached_monitoring(local_payload_from_backward(backward))
    local_start = rank * LOCAL_G
    local_candidates = group.candidates[local_start:local_start + LOCAL_G]
    ratio = global_ratio_stats(backward.local_current_logps, local_candidates, device)
    gradients = gradient_audit(model)
    expected_calls = LOCAL_G // selected_mb
    losses = {
        "frontier": backward.global_frontier_value,
        "hpr_raw": backward.global_hpr_value_raw,
        "hpr_weighted": backward.global_hpr_value_weighted,
        "total": backward.global_total_value,
    }
    loss_finite = all(math.isfinite(value) for value in losses.values())
    gradient_finite = (
        gradients["lora_params_with_nonzero_grad"] > 0 and math.isfinite(gradients["lora_grad_norm"])
        and gradients["lora_grad_norm"] > 0 and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == gradients["inf_grad_count"] == 0
    )
    rank_states = gather_rank_objects({
        "rank": rank,
        "calls": {
            "forward_calls": backward.physical_forward_calls,
            "backward_calls": backward.physical_backward_calls,
            "hpr_extra_forward_calls": backward.hpr_extra_forward_calls,
        },
        "gradient": gradients,
        "runtime_plan_hash": backward.runtime_plan_hash,
    })
    gate_diagnostic = build_pre_optimizer_gate_diagnostic(
        ratio=ratio, selected_mb=selected_mb, expected_calls=expected_calls,
        losses=losses, rank_states=rank_states,
    )
    local_failure = not gate_diagnostic["pass"]
    callback_failure = False
    if pre_optimizer_diagnostic_callback is not None and rank == 0:
        try:
            pre_optimizer_diagnostic_callback(gate_diagnostic, group, backward, rank_monitoring)
        except Exception:
            callback_failure = True
    if pre_optimizer_diagnostic_callback is not None:
        distributed_fail_if(callback_failure, "pre-optimizer diagnostic callback failed", device)
    distributed_fail_if(local_failure, "pre-optimizer distributed gate failed", device)
    if stop_before_optimizer:
        raise DDPProductionError("requested diagnostic stop before optimizer")
    optimizer.step()
    torch.cuda.synchronize(device)
    if strict_parameter_audit:
        lora_after, _ = lora_parameter_sha(model)
        base_mutation = base_parameter_versions(model) != base_before
        lora_hashes = gather_rank_objects(lora_after)
        update_failure = lora_before == lora_after or base_mutation or len(set(lora_hashes)) != 1
    else:
        lora_after = None
        base_mutation = False
        update_failure = parameter_versions(model) == versions
    distributed_fail_if(update_failure, "parameter update agreement failed", device)
    local_memory = {
        "rank": rank,
        "healthy": True,
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "rank_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "rank_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "rank_allocated_after_group_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
    }
    rank_memory = gather_rank_objects(local_memory)
    wall_time_seconds = time.perf_counter() - started
    report = {
        "status": "PASS", "model_family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT),
        "provenance": provenance or {}, "group_id": group_id, "context_token_count": len(context_ids),
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
        "lora_changed": True if not strict_parameter_audit else lora_before != lora_after,
        "base_parameter_mutation": base_mutation,
        "optimizer_steps": 1, "optimizer": optimizer_audit or {},
        "dropout_active": (dropout or {}).get("rl_dropout_active", False),
        "trainability": trainability or {}, "rank_memory": rank_memory,
        "wall_time_seconds": wall_time_seconds,
        "max_rank_peak_allocated_gb": max(item["rank_peak_allocated_gb"] for item in rank_memory),
        "max_rank_peak_reserved_gb": max(item["rank_peak_reserved_gb"] for item in rank_memory),
        "OOM": False, "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "formal_pilot4096_started": False, "evaluation_started": False, "next_experiment_started": False,
    }
    return report, group, backward, rank_monitoring


def run_one_group(group_id: str, expected_context: int, expected_mb: int, device: torch.device) -> tuple[dict[str, Any], Any, Any]:
    records, _, order_info = load_pilot_order()
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    report, _, _, _ = run_loaded_group(
        group_id, records[group_id], model, ddp, optimizer, renderer, device,
        expected_context=expected_context, expected_mb=expected_mb, provenance=provenance,
        dropout=dropout, trainability=trainability, optimizer_audit=optimizer_audit,
        strict_parameter_audit=True,
    )
    report["order"] = order_info
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
            passed = distributed_restore_gate(restored, ranks)
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
