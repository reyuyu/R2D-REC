#!/usr/bin/env python3
"""Real generation-to-one-step correctness smoke for MC_USER_v1."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from run_mc_user_generation_smoke import (
    K,
    MAX_NEW_TOKENS,
    SEED,
    TEMPERATURE,
    TOP_P,
    generate_k2_route,
    run_preflight,
    score_generated_route,
    validate_exact_policy_ids,
)
from run_mc_user_real_smoke import (
    ADAPTER,
    BASE_MODEL,
    MEMORY_THRESHOLD_MIB,
    TRAIN_DATA,
    batch_to_device,
    lora_delta,
    lora_snapshot,
    optimizer_state_is_lora_only,
    parameter_sha256,
    restore_lora,
    validate_trainable,
)
from user_mc_train_step import mc_optimizer_step


LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
FORWARD_BATCH_SIZE = 1
RESULT_OUTPUT = "/data/GRPO_USER/results/mc_user_v1_e2e_smoke.json"


class MCE2ESmokeError(RuntimeError):
    pass


def set_generation_mode(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True


def cleanup_generation_cache(
    device: torch.device,
    *,
    collect_fn: Callable[[], Any] = gc.collect,
    empty_cache_fn: Callable[[], Any] = torch.cuda.empty_cache,
    allocated_fn: Callable[[torch.device], int] = torch.cuda.memory_allocated,
    reserved_fn: Callable[[torch.device], int] = torch.cuda.memory_reserved,
) -> dict[str, float]:
    collect_fn()
    empty_cache_fn()
    return {
        "allocated_mib_after_cleanup": allocated_fn(device) / (1024 * 1024),
        "reserved_mib_after_cleanup": reserved_fn(device) / (1024 * 1024),
    }


def set_training_mode(model: torch.nn.Module) -> None:
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()


def restore_identical_route_start(
    model: torch.nn.Module,
    initial_lora: Mapping[str, torch.Tensor],
    initial_lora_hash: str,
    *,
    restore_fn: Callable[..., Any] = restore_lora,
    hash_fn: Callable[..., tuple[str, int]] = parameter_sha256,
) -> None:
    restore_fn(model, initial_lora)
    restored_hash, _ = hash_fn(model, lora=True)
    if restored_hash != initial_lora_hash:
        raise MCE2ESmokeError("route did not restore the identical initial LoRA state")


def validate_update_contract(
    update: Mapping[str, Any],
    *,
    base_hash_before: str,
    base_hash_after: str,
    lora_hash_before: str,
    lora_hash_after: str,
    lora_delta_l2: float,
    lora_delta_max_abs: float,
    optimizer_state_lora_only: bool,
    optimizer_state_empty: bool,
) -> dict[str, Any]:
    if base_hash_before != base_hash_after:
        raise MCE2ESmokeError("one-step route changed frozen base parameters")
    active = int(update["active_unit_count"]) > 0
    if active:
        if not bool(update["finite"]) or not math.isfinite(float(update["loss"])):
            raise MCE2ESmokeError("active route loss is not finite")
        if float(update["grad_norm"]) <= 0.0:
            raise MCE2ESmokeError("active route gradient norm is not positive")
        if bool(update["skipped_update"]) or not bool(
            update["optimizer_step_performed"]
        ):
            raise MCE2ESmokeError("active route did not perform one optimizer step")
        if lora_hash_before == lora_hash_after:
            raise MCE2ESmokeError("active route LoRA hash did not change")
        if lora_delta_l2 <= 0.0 or lora_delta_max_abs <= 0.0:
            raise MCE2ESmokeError("active route LoRA delta is zero")
        if not optimizer_state_lora_only:
            raise MCE2ESmokeError("active route optimizer state is not LoRA-only")
    else:
        if not bool(update["skipped_update"]) or bool(
            update["optimizer_step_performed"]
        ):
            raise MCE2ESmokeError("no-credit route did not skip optimizer step")
        if lora_hash_before != lora_hash_after:
            raise MCE2ESmokeError("no-credit route changed LoRA parameters")
        if lora_delta_l2 != 0.0 or lora_delta_max_abs != 0.0:
            raise MCE2ESmokeError("no-credit route has a non-zero LoRA delta")
        if not optimizer_state_empty:
            raise MCE2ESmokeError("no-credit route advanced optimizer state")
    return {
        "active_credit": active,
        "base_hash_unchanged": True,
        "update_contract_passed": True,
    }


def candidate_credit_record(
    candidate: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    marginal_result: Mapping[str, Any],
    *,
    route: str,
) -> dict[str, Any]:
    positive_mass = sum(
        float(unit["delta"]) for unit in units if float(unit["delta"]) > 0.0
    )
    negative_mass = sum(
        abs(float(unit["delta"])) for unit in units if float(unit["delta"]) < 0.0
    )
    record = {
        "candidate_index": int(candidate["candidate_index"]),
        "format_valid": bool(candidate["format_valid"]),
        "reward": float(candidate["full_reward"]),
        "generated_token_count": int(candidate["generated_token_count"]),
        "positive_unit_count": int(candidate["positive_unit_count"]),
        "negative_unit_count": int(candidate["negative_unit_count"]),
        "zero_unit_count": int(candidate["zero_unit_count"]),
        "positive_credit_mass": positive_mass,
        "negative_credit_mass": negative_mass,
        "overlap_token_count": int(candidate["overlap_token_count"]),
        "mixed_sign_overlap_token_count": int(
            candidate["mixed_sign_overlap_token_count"]
        ),
        "projection_required": bool(candidate["projection_required"]),
    }
    if route == "chain":
        record["predicted_event_count"] = (
            len(marginal_result["credits"]) if marginal_result["valid"] else 0
        )
    return record


def load_beta_for_e2e(args: argparse.Namespace) -> torch.nn.Module:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base, str(args.adapter), is_trainable=True, local_files_only=True
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    set_generation_mode(model)
    return model


def run_route_e2e(
    route: str,
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    initial_base_hash: str,
    initial_lora_hash: str,
    initial_lora: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    restore_identical_route_start(
        model, initial_lora, initial_lora_hash
    )
    route_lora_before = lora_snapshot(model)
    base_hash_before, _ = parameter_sha256(model, lora=False)
    if base_hash_before != initial_base_hash:
        raise MCE2ESmokeError("base hash drifted before route execution")

    set_generation_mode(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    generated_ids, generation_contract = generate_k2_route(
        model, tokenizer, row, device
    )
    torch.cuda.synchronize(device)
    generation_wall = time.perf_counter() - generation_started
    generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    scored = score_generated_route(route, row, generated_ids, tokenizer)
    validate_exact_policy_ids(scored["policy_batch"], generated_ids)
    units = scored["rollout"]["credit_units_per_candidate"]
    candidate_records = [
        candidate_credit_record(
            scored["candidates"][index],
            units[index],
            scored["rollout"]["marginal_results"][index],
            route=route,
        )
        for index in range(K)
    ]

    # Generated tensors are local to generate_k2_route; only exact CPU ID lists
    # remain here. Collect Python references before releasing the CUDA cache.
    cache_cleanup = cleanup_generation_cache(device)
    lora_hash_pre_update, _ = parameter_sha256(model, lora=True)
    if lora_hash_pre_update != initial_lora_hash:
        raise MCE2ESmokeError("generation or scoring changed LoRA parameters")
    set_training_mode(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    training_batch = batch_to_device(scored["policy_batch"], device)
    validate_exact_policy_ids(training_batch, generated_ids)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    update = mc_optimizer_step(
        model,
        optimizer,
        training_batch,
        training_batch["credit_units_per_candidate"],
        forward_batch_size=FORWARD_BATCH_SIZE,
    )
    torch.cuda.synchronize(device)
    training_wall = time.perf_counter() - training_started
    training_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    base_hash_after, _ = parameter_sha256(model, lora=False)
    lora_hash_after, _ = parameter_sha256(model, lora=True)
    delta_l2, delta_max = lora_delta(model, route_lora_before)
    update_contract = validate_update_contract(
        update,
        base_hash_before=base_hash_before,
        base_hash_after=base_hash_after,
        lora_hash_before=initial_lora_hash,
        lora_hash_after=lora_hash_after,
        lora_delta_l2=delta_l2,
        lora_delta_max_abs=delta_max,
        optimizer_state_lora_only=optimizer_state_is_lora_only(optimizer, trainable),
        optimizer_state_empty=not bool(optimizer.state),
    )
    metadata = update["objective_metadata"]
    return {
        "route": route,
        **generation_contract,
        "candidate_count": K,
        "candidates": candidate_records,
        "exact_generated_ids_preserved": True,
        "generation_peak_vram_mib": generation_peak,
        "generation_wall_seconds": generation_wall,
        "cache_cleanup": cache_cleanup,
        "training_peak_vram_mib": training_peak,
        "training_wall_seconds": training_wall,
        "loss": float(update["loss"]),
        "grad_norm": float(update["grad_norm"]),
        "finite": bool(update["finite"]),
        "active_unit_count": int(metadata["active_unit_count"]),
        "active_token_count": int(metadata["active_token_count"]),
        "active_token_assignment_count": int(
            metadata["active_token_assignment_count"]
        ),
        "skipped_update": bool(update["skipped_update"]),
        "optimizer_step_performed": bool(update["optimizer_step_performed"]),
        "optimizer_state_lora_only": (
            optimizer_state_is_lora_only(optimizer, trainable)
            if metadata["active_unit_count"] > 0
            else None
        ),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "lora_hash_before": initial_lora_hash,
        "lora_hash_pre_update": lora_hash_pre_update,
        "lora_hash_after": lora_hash_after,
        "lora_delta_l2": delta_l2,
        "lora_delta_max_abs": delta_max,
        **update_contract,
    }


def execute_e2e_smoke(
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    model_loader: Callable[[argparse.Namespace], torch.nn.Module] = load_beta_for_e2e,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise MCE2ESmokeError(
            "CUDA_VISIBLE_DEVICES must equal the explicit --gpu-id"
        )
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from transformers import set_seed

    set_seed(SEED)
    device = torch.device("cuda:0")
    model = model_loader(args)
    trainable = validate_trainable(model)
    initial_base_hash, base_tensor_count = parameter_sha256(model, lora=False)
    initial_lora_hash, lora_tensor_count = parameter_sha256(model, lora=True)
    initial_lora = lora_snapshot(model)
    started = time.perf_counter()
    routes = {}
    for route in ("action", "chain"):
        routes[route] = run_route_e2e(
            route,
            model,
            preflight["tokenizer"],
            preflight["selection"][route],
            trainable,
            initial_base_hash,
            initial_lora_hash,
            initial_lora,
            device,
        )
    output = {
        "status": "PASS",
        "config": {
            "K": K,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "forward_batch_size": FORWARD_BATCH_SIZE,
            "optimizer_steps_per_route": 1,
        },
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "selected_sample_ids": preflight["selected_sample_ids"],
        "trainable_tensor_count": len(trainable),
        "base_tensor_count": base_tensor_count,
        "lora_tensor_count": lora_tensor_count,
        "routes": routes,
        "combined_process_peak_vram_mib": max(
            max(route["generation_peak_vram_mib"], route["training_peak_vram_mib"])
            for route in routes.values()
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-model", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--adapter", type=Path, default=Path(ADAPTER))
    parser.add_argument("--train-data", type=Path, default=Path(TRAIN_DATA))
    parser.add_argument("--result-output", type=Path, default=Path(RESULT_OUTPUT))
    parser.add_argument(
        "--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB
    )
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": preflight["status"],
        "gpu": preflight["gpu"],
        "selected_sample_ids": preflight["selected_sample_ids"],
        "prompt_token_counts": preflight["prompt_token_counts"],
        "train_sha256": preflight["paths"]["train_sha256"],
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Callable[..., Mapping[str, Any]] = run_preflight,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_e2e_smoke,
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
