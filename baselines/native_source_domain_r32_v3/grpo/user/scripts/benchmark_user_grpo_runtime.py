#!/usr/bin/env python3
"""Three-part four-GPU runtime benchmark for the frozen GR_USER_v1 objective."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from run_user_grpo_smoke import (
    ADAPTER,
    BASE_MODEL,
    EXPECTED_DATA_SHA,
    G,
    MAX_NEW_TOKENS,
    PILOT,
    SEED,
    TEMPERATURE,
    TOP_P,
    dataset_sha,
    generate_route,
    grad_norm,
    read_jsonl,
    render_prompt,
)
from user_grpo_trainer import UserGRPOTrainer, prepare_scored_rollout


CONFIGS = (
    {"name": "P0", "length_bucketing": False, "forward_batch_size": 1},
    {"name": "P1", "length_bucketing": True, "forward_batch_size": 1},
)
ACTION_BUCKETS = ((2, 5), (6, 10), (11, 20), (21, 10**9))
ROUTES = ("action", "chain")
GENERATION_WARMUP_PROMPTS = 8
GENERATION_TIMED_REPEATS = 1
CACHED_GLOBAL_PROMPTS = 8
CACHED_WARMUP_STEPS = 2
CACHED_TIMED_STEPS = 5
TINY_PROMPTS = 8
PARITY_RELATIVE_L2_LIMIT = 0.02
PARITY_COSINE_LIMIT = 0.9999
UPDATE_COSINE_LIMIT = 0.995


def _spread(rows, count):
    ordered = sorted(rows, key=lambda row: (row["prompt_token_count"], row["sample_id"]))
    if len(ordered) < count:
        raise ValueError("selection bucket is too small")
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def select_benchmark_rows(rows):
    """Select a deterministic 20 Action + 20 Chain fixture."""
    action_rows = [row for row in rows if row["route"] == "action"]
    chain_rows = [row for row in rows if row["route"] == "chain"]
    action = []
    for low, high in ACTION_BUCKETS:
        action.extend(_spread([row for row in action_rows if low <= row["gold_sid_count"] <= high], 5))
    chain = []
    for event_count in (2, 3, 4, 5):
        chain.extend(_spread([row for row in chain_rows if row["gold_event_count"] == event_count], 5))
    selected = {"action": action, "chain": chain}
    sample_ids = [row["sample_id"] for route in ROUTES for row in selected[route]]
    if len(sample_ids) != 40 or len(set(sample_ids)) != 40:
        raise AssertionError("benchmark selection must contain 40 unique prompts")
    return selected


def schedule_rows(rows, *, length_bucketing, world_size=4, prompts_per_rank=2):
    """Build synchronized rounds and keep complete G=4 groups on one rank."""
    ordered = list(rows)
    if length_bucketing:
        ordered.sort(key=lambda row: (row["prompt_token_count"], row["sample_id"]))
    width = world_size * prompts_per_rank
    rounds = []
    for start in range(0, len(ordered), width):
        global_batch = ordered[start : start + width]
        if len(global_batch) <= world_size:
            per_rank = [[global_batch[rank]] if rank < len(global_batch) else [] for rank in range(world_size)]
        else:
            per_rank = [
                global_batch[rank * prompts_per_rank : (rank + 1) * prompts_per_rank]
                for rank in range(world_size)
            ]
        rounds.append(per_rank)
    flattened = [row["sample_id"] for round_rows in rounds for rank_rows in round_rows for row in rank_rows]
    if collections.Counter(flattened) != collections.Counter(row["sample_id"] for row in rows):
        raise AssertionError("scheduling changed prompt membership")
    return rounds


def padding_counts(rows):
    if not rows:
        return 0, 0
    true_tokens = sum(int(row["prompt_token_count"]) for row in rows)
    padded_total = max(int(row["prompt_token_count"]) for row in rows) * len(rows)
    return padded_total - true_tokens, true_tokens


def _seed(offset=0):
    value = SEED + offset
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _all_reduce_scalar(value, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(float(value), device=torch.cuda.current_device(), dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor)


def _all_gather_object(value):
    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return values


def _trainable(model):
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not values or any("lora_" not in name.lower() for name, _ in values):
        raise RuntimeError("only LoRA parameters may be trainable")
    return values


def _restore_lora(model, snapshot):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.copy_(snapshot[name].to(parameter.device, dtype=parameter.dtype))


def _snapshot_tensors(named_parameters):
    return {name: parameter.detach().float().cpu().clone() for name, parameter in named_parameters}


def _snapshot_gradients(named_parameters):
    gradients = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"trainable parameter has no gradient: {name}")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    return gradients


def _tensor_delta(named_parameters, before):
    return {
        name: parameter.detach().float().cpu() - before[name]
        for name, parameter in named_parameters
    }


def _compare_vectors(reference, candidate):
    difference_squared = reference_squared = candidate_squared = dot = 0.0
    max_abs_error = 0.0
    for name, expected in reference.items():
        actual = candidate[name]
        max_abs_error = max(max_abs_error, float((expected - actual).abs().max()))
        expected64 = expected.double()
        actual64 = actual.double()
        difference_squared += float(((expected64 - actual64) ** 2).sum())
        reference_squared += float((expected64**2).sum())
        candidate_squared += float((actual64**2).sum())
        dot += float((expected64 * actual64).sum())
    return {
        "max_abs_error": max_abs_error,
        "relative_l2_error": math.sqrt(difference_squared / max(reference_squared, 1e-30)),
        "cosine_similarity": dot / max(math.sqrt(reference_squared * candidate_squared), 1e-30),
    }


def parity_accepted(metrics):
    return bool(
        metrics["reward_exact"]
        and metrics["sequence_advantage_exact"]
        and metrics["token_advantage_exact"]
        and metrics["penalty_mask_exact"]
        and metrics["loss_abs_error"] <= 2e-5
        and metrics["gradient"]["relative_l2_error"] <= PARITY_RELATIVE_L2_LIMIT
        and metrics["gradient"]["cosine_similarity"] >= PARITY_COSINE_LIMIT
        and metrics["one_step_update"]["cosine_similarity"] >= UPDATE_COSINE_LIMIT
    )


def _generation_pass(model, tokenizer, selected, config, rank, device, *, seed_offset, capture):
    model.eval()
    model.module.config.use_cache = True
    local_completions = {}
    local_padding = local_true = local_generated = 0
    _seed(seed_offset)
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for route in ROUTES:
        for per_rank in schedule_rows(selected[route], length_bucketing=config["length_bucketing"]):
            local_rows = per_rank[rank]
            if not local_rows:
                continue
            padded, true_tokens = padding_counts(local_rows)
            local_padding += padded
            local_true += true_tokens
            completions = generate_route(model.module, tokenizer, local_rows, device)
            local_generated += sum(len(ids) for ids in completions)
            if capture:
                for row_index, row in enumerate(local_rows):
                    local_completions[row["sample_id"]] = completions[row_index * G : (row_index + 1) * G]
    torch.cuda.synchronize(device)
    dist.barrier()
    wall_seconds = time.perf_counter() - started
    totals = {
        "prompt_padding_tokens": int(_all_reduce_scalar(local_padding)),
        "prompt_true_tokens": int(_all_reduce_scalar(local_true)),
        "generated_tokens": int(_all_reduce_scalar(local_generated)),
    }
    completions = None
    if capture:
        completions = {}
        for shard in _all_gather_object(local_completions):
            completions.update(shard)
    return wall_seconds, totals, completions


def benchmark_generation(model, tokenizer, selected, config, rank, device):
    warmup = {route: selected[route][: GENERATION_WARMUP_PROMPTS // 2] for route in ROUTES}
    _generation_pass(model, tokenizer, warmup, config, rank, device, seed_offset=100, capture=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    repeats = []
    captured = None
    for repeat in range(GENERATION_TIMED_REPEATS):
        wall, totals, completions = _generation_pass(
            model, tokenizer, selected, config, rank, device,
            seed_offset=1000 + repeat * 100, capture=True,
        )
        repeats.append({"wall_seconds": wall, **totals})
        if captured is None:
            captured = completions
    local_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peaks = _all_gather_object(local_peak)
    if rank != 0:
        return None, None
    wall = statistics.fmean(item["wall_seconds"] for item in repeats)
    true_tokens = statistics.fmean(item["prompt_true_tokens"] for item in repeats)
    padding_tokens = statistics.fmean(item["prompt_padding_tokens"] for item in repeats)
    generated_tokens = statistics.fmean(item["generated_tokens"] for item in repeats)
    result = {
        **config,
        "warmup_repeats": 1,
        "timed_repeats": GENERATION_TIMED_REPEATS,
        "prompts_per_repeat": 40,
        "generation_wall_seconds": wall,
        "seconds_per_prompt": wall / 40,
        "prompts_per_hour": 40 / wall * 3600,
        "generated_tokens_per_second": generated_tokens / wall,
        "prompt_true_tokens": int(true_tokens),
        "prompt_padding_tokens": int(padding_tokens),
        "prompt_padding_ratio": padding_tokens / (true_tokens + padding_tokens),
        "peak_vram_mib": max(peaks),
        "per_gpu_peak_vram_mib": peaks,
        "repeats": repeats,
    }
    return result, captured


def build_rollout_cache(selected, completions, tokenizer):
    groups = {}
    for route in ROUTES:
        for row in selected[route]:
            completion_ids = completions[row["sample_id"]]
            rollout = prepare_scored_rollout([row], completion_ids, tokenizer)
            groups[row["sample_id"]] = {
                "route": route,
                "completion_ids": completion_ids,
                "rewards": rollout["rewards"].tolist(),
                "sequence_advantages": rollout["sequence_advantages"].tolist(),
                "token_advantages": [
                    rollout["token_advantages"][index, : len(ids)].tolist()
                    for index, ids in enumerate(completion_ids)
                ],
                "penalty_masks": [
                    rollout["local_penalty_mask"][index, : len(ids)].tolist()
                    for index, ids in enumerate(completion_ids)
                ],
            }
    return {
        "contract_version": "gr_user_runtime_cached_rollout_v1r",
        "G": G,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "groups": groups,
    }


def validate_rollout_cache(cache, selected):
    expected = {row["sample_id"] for route in ROUTES for row in selected[route]}
    if cache.get("G") != G or set(cache.get("groups", {})) != expected:
        raise RuntimeError("cached rollout does not match benchmark selection")
    for sample_id, group in cache["groups"].items():
        lengths = [len(ids) for ids in group["completion_ids"]]
        if len(lengths) != G or len(group["rewards"]) != G:
            raise RuntimeError(f"invalid G=4 cache group: {sample_id}")
        for field in ("sequence_advantages", "token_advantages", "penalty_masks"):
            if len(group[field]) != G:
                raise RuntimeError(f"invalid cached field {field}: {sample_id}")
        if any(len(group["token_advantages"][index]) != length for index, length in enumerate(lengths)):
            raise RuntimeError(f"token advantage length mismatch: {sample_id}")
        if any(len(group["penalty_masks"][index]) != length for index, length in enumerate(lengths)):
            raise RuntimeError(f"penalty mask length mismatch: {sample_id}")


def make_cached_policy_batch(tokenizer, rows, cache, device):
    prompt_ids_list = []
    completion_ids_list = []
    token_advantages_list = []
    penalty_masks_list = []
    sequence_advantages = []
    rewards = []
    for row in rows:
        group = cache["groups"][row["sample_id"]]
        prompt_ids = tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False)
        for candidate in range(G):
            prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(group["completion_ids"][candidate])
            token_advantages_list.append(group["token_advantages"][candidate])
            penalty_masks_list.append(group["penalty_masks"][candidate])
            sequence_advantages.append(group["sequence_advantages"][candidate])
            rewards.append(group["rewards"][candidate])
    prompt_width = max(map(len, prompt_ids_list))
    completion_width = max(map(len, completion_ids_list))
    count = len(completion_ids_list)
    prompt_ids = torch.full((count, prompt_width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    prompt_mask = torch.zeros_like(prompt_ids)
    completion_ids = torch.full((count, completion_width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    completion_mask = torch.zeros((count, completion_width), dtype=torch.float32, device=device)
    token_advantages = torch.zeros((count, completion_width), dtype=torch.float32, device=device)
    penalty_mask = torch.zeros((count, completion_width), dtype=torch.bool, device=device)
    for index, (prompt, completion) in enumerate(zip(prompt_ids_list, completion_ids_list)):
        prompt_ids[index, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
        prompt_mask[index, -len(prompt) :] = 1
        completion_ids[index, : len(completion)] = torch.tensor(completion, dtype=torch.long, device=device)
        completion_mask[index, : len(completion)] = 1
        token_advantages[index, : len(completion)] = torch.tensor(token_advantages_list[index], device=device)
        penalty_mask[index, : len(completion)] = torch.tensor(penalty_masks_list[index], device=device)
    valid_tokens = int(prompt_mask.sum() + completion_mask.sum())
    padded_tokens = int(prompt_ids.numel() + completion_ids.numel() - valid_tokens)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "token_advantages": token_advantages,
        "local_penalty_mask": penalty_mask,
        "sequence_advantages": torch.tensor(sequence_advantages, dtype=torch.float32, device=device),
        "cache_rewards": rewards,
        "valid_policy_tokens": valid_tokens,
        "padded_policy_tokens": padded_tokens,
    }


def _policy_rows(selected, count_per_route):
    return {route: selected[route][:count_per_route] for route in ROUTES}


def _local_policy_batches(tokenizer, rows_by_route, cache, config, rank, device, prompts_per_rank):
    batches = {}
    for route in ROUTES:
        rounds = schedule_rows(
            rows_by_route[route],
            length_bucketing=config["length_bucketing"],
            prompts_per_rank=prompts_per_rank,
        )
        if len(rounds) != 1 or not rounds[0][rank]:
            raise AssertionError("policy fixture must form one non-empty batch per rank")
        batches[route] = make_cached_policy_batch(tokenizer, rounds[0][rank], cache, device)
    return batches


def _attach_old_logps(model, trainer, batches, device):
    model.eval()
    started = time.perf_counter()
    for batch in batches.values():
        combined = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        with torch.no_grad():
            batch["old_per_token_logps"] = trainer._get_user_per_token_logps(
                model, combined, attention, batch["completion_ids"].size(1)
            ).detach()
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _cached_optimizer_step(model, trainer, optimizer, trainable, batch, device):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = trainer._compute_loss(model, batch)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    raw_grad_norm = grad_norm([parameter for _, parameter in trainable])
    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    return {
        "loss": float(loss.detach()),
        "policy_forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "grad_norm": raw_grad_norm,
        "clip_fraction": trainer.last_loss_metrics["clip_fraction"],
        "policy_tokens": batch["valid_policy_tokens"],
        "policy_padding_tokens": batch["padded_policy_tokens"],
    }


def correctness_run(model, tokenizer, selected, cache, config, initial_lora, trainable, rank, device, learning_rate):
    _restore_lora(model, initial_lora)
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=learning_rate, weight_decay=0.0)
    rows_by_route = _policy_rows(selected, 4)
    batches = _local_policy_batches(tokenizer, rows_by_route, cache, config, rank, device, prompts_per_rank=1)
    _attach_old_logps(model, trainer, batches, device)
    optimizer.zero_grad(set_to_none=True)
    local_loss = 0.0
    for route in ROUTES:
        model.train()
        loss = trainer._compute_loss(model, batches[route]) * 0.5
        loss.backward()
        local_loss += float(loss.detach())
    gradients = _snapshot_gradients(trainable)
    before_update = _snapshot_tensors(trainable)
    optimizer.step()
    torch.cuda.synchronize(device)
    update = _tensor_delta(trainable, before_update)
    global_loss = _all_reduce_scalar(local_loss) / dist.get_world_size()
    signature = {
        sample_id: {
            field: cache["groups"][sample_id][field]
            for field in ("rewards", "sequence_advantages", "token_advantages", "penalty_masks")
        }
        for route in ROUTES for sample_id in [row["sample_id"] for row in rows_by_route[route]]
    }
    return {"loss": global_loss, "gradients": gradients, "update": update, "signature": signature}


def compare_correctness(reference, candidate):
    reward_exact = sequence_exact = token_exact = mask_exact = True
    for sample_id, expected in reference["signature"].items():
        actual = candidate["signature"][sample_id]
        reward_exact &= expected["rewards"] == actual["rewards"]
        sequence_exact &= expected["sequence_advantages"] == actual["sequence_advantages"]
        token_exact &= expected["token_advantages"] == actual["token_advantages"]
        mask_exact &= expected["penalty_masks"] == actual["penalty_masks"]
    result = {
        "reward_exact": bool(reward_exact),
        "sequence_advantage_exact": bool(sequence_exact),
        "token_advantage_exact": bool(token_exact),
        "penalty_mask_exact": bool(mask_exact),
        "loss_abs_error": abs(reference["loss"] - candidate["loss"]),
        "loss_tolerance": 2e-5,
        "gradient": _compare_vectors(reference["gradients"], candidate["gradients"]),
        "one_step_update": _compare_vectors(reference["update"], candidate["update"]),
        "acceptance": {
            "gradient_relative_l2_limit": PARITY_RELATIVE_L2_LIMIT,
            "gradient_cosine_limit": PARITY_COSINE_LIMIT,
            "one_step_update_cosine_limit": UPDATE_COSINE_LIMIT,
        },
    }
    result["passed"] = parity_accepted(result)
    return result


def benchmark_cached_policy(model, tokenizer, selected, cache, config, initial_lora, trainable, rank, device, learning_rate):
    rows_by_route = _policy_rows(selected, CACHED_GLOBAL_PROMPTS)
    _restore_lora(model, initial_lora)
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=learning_rate, weight_decay=0.0)
    batches = _local_policy_batches(tokenizer, rows_by_route, cache, config, rank, device, prompts_per_rank=2)
    _attach_old_logps(model, trainer, batches, device)
    for step in range(CACHED_WARMUP_STEPS):
        _cached_optimizer_step(model, trainer, optimizer, trainable, batches[ROUTES[step % 2]], device)

    _restore_lora(model, initial_lora)
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=learning_rate, weight_decay=0.0)
    batches = _local_policy_batches(tokenizer, rows_by_route, cache, config, rank, device, prompts_per_rank=2)
    old_logps_seconds = _attach_old_logps(model, trainer, batches, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    dist.barrier()
    for step in range(CACHED_TIMED_STEPS):
        route = ROUTES[step % 2]
        dist.barrier()
        started = time.perf_counter()
        local = _cached_optimizer_step(model, trainer, optimizer, trainable, batches[route], device)
        dist.barrier()
        wall = time.perf_counter() - started
        record = {
            "step": step + 1,
            "route": route,
            "wall_seconds": wall,
            "loss": _all_reduce_scalar(local["loss"]) / dist.get_world_size(),
            "policy_forward_seconds": _all_reduce_scalar(local["policy_forward_seconds"], dist.ReduceOp.MAX),
            "backward_seconds": _all_reduce_scalar(local["backward_seconds"], dist.ReduceOp.MAX),
            "optimizer_seconds": _all_reduce_scalar(local["optimizer_seconds"], dist.ReduceOp.MAX),
            "grad_norm": _all_reduce_scalar(local["grad_norm"], dist.ReduceOp.MAX),
            "clip_fraction": _all_reduce_scalar(local["clip_fraction"]) / dist.get_world_size(),
            "policy_tokens": int(_all_reduce_scalar(local["policy_tokens"])),
            "policy_padding_tokens": int(_all_reduce_scalar(local["policy_padding_tokens"])),
        }
        if rank == 0:
            records.append(record)
    local_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peaks = _all_gather_object(local_peak)
    old_logps_seconds = _all_reduce_scalar(old_logps_seconds, dist.ReduceOp.MAX)
    if rank != 0:
        return None
    total_wall = sum(record["wall_seconds"] for record in records)
    tokens = sum(record["policy_tokens"] for record in records)
    padding = sum(record["policy_padding_tokens"] for record in records)
    active = sum(record["policy_forward_seconds"] + record["backward_seconds"] for record in records)
    prompts = CACHED_GLOBAL_PROMPTS * CACHED_TIMED_STEPS
    return {
        **config,
        "warmup_steps": CACHED_WARMUP_STEPS,
        "timed_steps": CACHED_TIMED_STEPS,
        "global_prompts_per_step": CACHED_GLOBAL_PROMPTS,
        "seconds_per_step": total_wall / CACHED_TIMED_STEPS,
        "seconds_per_prompt": total_wall / prompts,
        "policy_tokens_per_second": tokens / active,
        "policy_forward_seconds": sum(record["policy_forward_seconds"] for record in records),
        "backward_seconds": sum(record["backward_seconds"] for record in records),
        "optimizer_seconds": sum(record["optimizer_seconds"] for record in records),
        "old_policy_cache_build_seconds": old_logps_seconds,
        "policy_padding_ratio": padding / (padding + tokens),
        "grad_norm_mean": statistics.fmean(record["grad_norm"] for record in records),
        "grad_norm_max": max(record["grad_norm"] for record in records),
        "clip_fraction_mean": statistics.fmean(record["clip_fraction"] for record in records),
        "peak_vram_mib": max(peaks),
        "per_gpu_peak_vram_mib": peaks,
        "steps": records,
    }


def benchmark_tiny_e2e(model, tokenizer, selected, config, initial_lora, trainable, rank, device, learning_rate):
    _restore_lora(model, initial_lora)
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=learning_rate, weight_decay=0.0)
    tiny = _policy_rows(selected, TINY_PROMPTS // 2)
    local_completions = {}
    local_generated = local_policy_tokens = 0
    _seed(9000)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    wall_started = time.perf_counter()
    generation_started = time.perf_counter()
    model.eval()
    for route in ROUTES:
        per_rank = schedule_rows(
            tiny[route], length_bucketing=config["length_bucketing"], prompts_per_rank=1
        )[0]
        local_rows = per_rank[rank]
        completions = generate_route(model.module, tokenizer, local_rows, device)
        local_generated += sum(len(ids) for ids in completions)
        local_completions[route] = (local_rows, completions)
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started

    reward_seconds = forward_seconds = backward_seconds = 0.0
    local_loss = local_grad_norm = local_clip = 0.0
    for route in ROUTES:
        local_rows, completions = local_completions[route]
        started = time.perf_counter()
        ephemeral = build_rollout_cache({route: local_rows, **{other: [] for other in ROUTES if other != route}}, {
            row["sample_id"]: completions[index * G : (index + 1) * G]
            for index, row in enumerate(local_rows)
        }, tokenizer)
        batch = make_cached_policy_batch(tokenizer, local_rows, ephemeral, device)
        reward_seconds += time.perf_counter() - started
        local_policy_tokens += batch["valid_policy_tokens"]
        combined = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        started = time.perf_counter()
        with torch.no_grad():
            batch["old_per_token_logps"] = trainer._get_user_per_token_logps(
                model, combined, attention, batch["completion_ids"].size(1)
            ).detach()
        model.train()
        loss = trainer._compute_loss(model, batch) * 0.5
        torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds += time.perf_counter() - started
        local_loss += float(loss.detach())
        local_clip += trainer.last_loss_metrics["clip_fraction"] * 0.5
    local_grad_norm = grad_norm([parameter for _, parameter in trainable])
    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    dist.barrier()
    wall_seconds = time.perf_counter() - wall_started
    local_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peaks = _all_gather_object(local_peak)
    result = {
        **config,
        "prompts": TINY_PROMPTS,
        "wall_seconds": wall_seconds,
        "seconds_per_prompt": wall_seconds / TINY_PROMPTS,
        "generation_seconds": _all_reduce_scalar(generation_seconds, dist.ReduceOp.MAX),
        "reward_mask_seconds": _all_reduce_scalar(reward_seconds, dist.ReduceOp.MAX),
        "policy_forward_seconds": _all_reduce_scalar(forward_seconds, dist.ReduceOp.MAX),
        "backward_seconds": _all_reduce_scalar(backward_seconds, dist.ReduceOp.MAX),
        "optimizer_seconds": _all_reduce_scalar(optimizer_seconds, dist.ReduceOp.MAX),
        "generated_tokens": int(_all_reduce_scalar(local_generated)),
        "policy_tokens": int(_all_reduce_scalar(local_policy_tokens)),
        "loss": _all_reduce_scalar(local_loss) / dist.get_world_size(),
        "grad_norm": _all_reduce_scalar(local_grad_norm, dist.ReduceOp.MAX),
        "clip_fraction": _all_reduce_scalar(local_clip) / dist.get_world_size(),
        "peak_vram_mib": max(peaks),
        "per_gpu_peak_vram_mib": peaks,
    }
    return result if rank == 0 else None


def add_speedups(results, wall_field):
    baseline = results[0][wall_field]
    for item in results:
        item["speedup_percent_vs_p0"] = (baseline / item[wall_field] - 1.0) * 100


def estimate_wall_times(generation, cached):
    by_name = {}
    for generation_item, cached_item in zip(generation, cached):
        per_prompt = generation_item["seconds_per_prompt"] + cached_item["seconds_per_prompt"]
        by_name[generation_item["name"]] = {
            "generation_seconds_per_prompt": generation_item["seconds_per_prompt"],
            "cached_policy_backward_seconds_per_prompt": cached_item["seconds_per_prompt"],
            "combined_seconds_per_prompt": per_prompt,
            "wall_seconds": {str(count): per_prompt * count for count in (150, 300, 600, 3000)},
        }
    return by_name


def comparison_summary(generation, cached, estimates):
    p0_generation, p1_generation = generation
    p0_cached, p1_cached = cached
    return {
        "generation_speedup_percent": (
            p0_generation["generation_wall_seconds"] / p1_generation["generation_wall_seconds"] - 1.0
        ) * 100,
        "generation_padding_reduction_percentage_points": (
            p0_generation["prompt_padding_ratio"] - p1_generation["prompt_padding_ratio"]
        ) * 100,
        "generation_padding_relative_reduction_percent": (
            1.0 - p1_generation["prompt_padding_ratio"] / p0_generation["prompt_padding_ratio"]
        ) * 100,
        "cached_policy_speedup_percent": (
            p0_cached["seconds_per_step"] / p1_cached["seconds_per_step"] - 1.0
        ) * 100,
        "cached_policy_padding_reduction_percentage_points": (
            p0_cached["policy_padding_ratio"] - p1_cached["policy_padding_ratio"]
        ) * 100,
        "cached_policy_padding_relative_reduction_percent": (
            1.0 - p1_cached["policy_padding_ratio"] / p0_cached["policy_padding_ratio"]
        ) * 100,
        "combined_speedup_percent": (
            estimates["P0"]["combined_seconds_per_prompt"]
            / estimates["P1"]["combined_seconds_per_prompt"]
            - 1.0
        ) * 100,
    }


def preflight_four_gpus():
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    states = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory}
    if set(states) != {0, 1, 2, 3} or compute or any(value > 1024 for value in states.values()):
        raise RuntimeError("all four GPUs must be idle before benchmark launch")
    return states


def write_report(path, result):
    generation_rows = "\n".join(
        f'| {item["name"]} | {item["generation_wall_seconds"]:.2f} | {item["seconds_per_prompt"]:.3f} | '
        f'{item["prompts_per_hour"]:.1f} | {item["generated_tokens_per_second"]:.1f} | '
        f'{item["prompt_padding_ratio"]:.2%} | {item["peak_vram_mib"]:.0f} | {item["speedup_percent_vs_p0"]:.2f}% |'
        for item in result["generation"]
    )
    policy_rows = "\n".join(
        f'| {item["name"]} | {item["seconds_per_step"]:.3f} | {item["seconds_per_prompt"]:.3f} | '
        f'{item["policy_tokens_per_second"]:.1f} | {item["policy_padding_ratio"]:.2%} | '
        f'{item["grad_norm_mean"]:.4f} | {item["clip_fraction_mean"]:.4f} | '
        f'{item["peak_vram_mib"]:.0f} | {item["speedup_percent_vs_p0"]:.2f}% |'
        for item in result["cached_policy_backward"]
    )
    estimate_rows = "\n".join(
        f'| {name} | {values["generation_seconds_per_prompt"]:.3f} | '
        f'{values["cached_policy_backward_seconds_per_prompt"]:.3f} | '
        f'{values["wall_seconds"]["150"]:.1f} | {values["wall_seconds"]["300"]:.1f} | '
        f'{values["wall_seconds"]["600"]:.1f} | {values["wall_seconds"]["3000"]:.1f} |'
        for name, values in result["estimated_wall_seconds"].items()
    )
    text = f"""# GR_USER_v1 Phase 5A-R Runtime Benchmark

This benchmark separates generation, cached policy/backward, and a tiny
end-to-end check. It is not a formal Pilot or a full epoch. G=4,
temperature=0.9, top-p=0.95, max-new-tokens=512, sqrt lambda=0.5, reward,
sampling, epsilon, beta, and optimizer mathematics remain frozen.

## Generation

| Config | wall sec | sec/prompt | prompts/hour | generated tok/sec | padding ratio | peak VRAM MiB | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
{generation_rows}

One short 8-prompt warmup and one timed 40-prompt pass are used per config.

## Cached policy/backward

| Config | sec/step | sec/prompt | policy tok/sec | padding ratio | grad norm | clip fraction | peak VRAM MiB | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{policy_rows}

Both configs read identical cached completion IDs, rewards, sequence/token
advantages, and penalty masks. Two warmup steps are excluded and five small
global-batch steps are timed on real four-GPU DDP.

## Correctness

```json
{json.dumps(result['correctness_parity'], ensure_ascii=False, indent=2)}
```

## Tiny end-to-end

```json
{json.dumps(result['tiny_end_to_end'], ensure_ascii=False, indent=2)}
```

## Extrapolation

| Config | generation sec/prompt | cached policy sec/prompt | 150 | 300 | 600 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
{estimate_rows}

Recommended runtime config: **{result['recommended_config']}**.

P1 comparison versus P0: generation speedup
`{result['comparison']['generation_speedup_percent']:.2f}%`, cached policy speedup
`{result['comparison']['cached_policy_speedup_percent']:.2f}%`, and combined speedup
`{result['comparison']['combined_speedup_percent']:.2f}%`. Generation padding fell
by `{result['comparison']['generation_padding_relative_reduction_percent']:.2f}%`
relative; cached policy padding fell by
`{result['comparison']['cached_policy_padding_relative_reduction_percent']:.2f}%`.

Frozen data remained unchanged. No 150-prompt Pilot, formal Pilot, full epoch,
or P2 performance benchmark was run. GR_REC_v1 was not modified.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=Path("/data/GRPO_USER/results/runtime_benchmark_v1r.json"))
    parser.add_argument("--docs-output", type=Path, default=Path("/data/GRPO_USER/docs/runtime_benchmark_v1r.md"))
    parser.add_argument("--cache-output", type=Path, default=Path("/data/GRPO_USER/runs/runtime_benchmark_v1r_cached_rollout.json"))
    parser.add_argument("--resume-generation-results", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.get_world_size() != 4 or local_rank not in range(4):
        raise RuntimeError("runtime benchmark requires exactly four GPU ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    sha_before = dataset_sha(Path(PILOT).parent)
    if sha_before != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen GR_USER_v1 data SHA mismatch")
    selected = select_benchmark_rows(read_jsonl(PILOT))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER, is_trainable=True, local_files_only=True)
    for module in peft_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    peft_model.config.use_cache = False
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DistributedDataParallel(peft_model, device_ids=[local_rank], output_device=local_rank)
    trainable = _trainable(model)
    initial_lora = {name: parameter.detach().cpu().clone() for name, parameter in trainable}

    if args.resume_generation_results:
        if rank == 0:
            generation_results = json.loads(args.resume_generation_results.read_text(encoding="utf-8"))["generation"]
            if [item["name"] for item in generation_results] != ["P0", "P1"]:
                raise RuntimeError("generation resume file must contain P0 and P1")
        else:
            generation_results = []
        if not args.cache_output.is_file():
            raise RuntimeError("generation resume requires the fixed rollout cache")
    else:
        generation_results = []
        p0_completions = None
        for config in CONFIGS:
            _restore_lora(model, initial_lora)
            result, completions = benchmark_generation(model, tokenizer, selected, config, rank, device)
            if rank == 0:
                generation_results.append(result)
                if config["name"] == "P0":
                    p0_completions = completions
                print(json.dumps({"generation_complete": config["name"], "result": result}), flush=True)

        if rank == 0:
            cache = build_rollout_cache(selected, p0_completions, tokenizer)
            validate_rollout_cache(cache, selected)
            args.cache_output.parent.mkdir(parents=True, exist_ok=True)
            args.cache_output.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    dist.barrier()
    cache = json.loads(args.cache_output.read_text(encoding="utf-8"))
    validate_rollout_cache(cache, selected)

    correctness_runs = {}
    for config in CONFIGS:
        correctness_runs[config["name"]] = correctness_run(
            model, tokenizer, selected, cache, config, initial_lora, trainable,
            rank, device, args.learning_rate,
        )
    if rank == 0:
        correctness = {
            "P0": {
                "passed": True,
                "reward_exact": True,
                "sequence_advantage_exact": True,
                "token_advantage_exact": True,
                "penalty_mask_exact": True,
                "loss_abs_error": 0.0,
                "gradient": {"max_abs_error": 0.0, "relative_l2_error": 0.0, "cosine_similarity": 1.0},
                "one_step_update": {"max_abs_error": 0.0, "relative_l2_error": 0.0, "cosine_similarity": 1.0},
            },
            "P1": compare_correctness(correctness_runs["P0"], correctness_runs["P1"]),
        }
    else:
        correctness = None
    if rank == 0:
        parity_debug = args.result_output.with_name("runtime_benchmark_v1r_parity_debug.json")
        parity_debug.parent.mkdir(parents=True, exist_ok=True)
        parity_debug.write_text(json.dumps(correctness, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"correctness_parity": correctness}, ensure_ascii=False), flush=True)
    accepted = torch.tensor(int(correctness is None or correctness["P1"]["passed"]), device=device)
    dist.broadcast(accepted, 0)
    if not bool(accepted.item()):
        dist.destroy_process_group()
        raise RuntimeError("P1 correctness acceptance failed; performance benchmark aborted")

    cached_results = []
    for config in CONFIGS:
        result = benchmark_cached_policy(
            model, tokenizer, selected, cache, config, initial_lora, trainable,
            rank, device, args.learning_rate,
        )
        if rank == 0:
            cached_results.append(result)
            print(json.dumps({"cached_policy_complete": config["name"], "result": result}), flush=True)

    tiny_results = []
    for config in CONFIGS:
        result = benchmark_tiny_e2e(
            model, tokenizer, selected, config, initial_lora, trainable,
            rank, device, args.learning_rate,
        )
        if rank == 0:
            tiny_results.append(result)
            print(json.dumps({"tiny_e2e_complete": config["name"], "result": result}), flush=True)

    sha_after = dataset_sha(Path(PILOT).parent)
    if sha_after != sha_before:
        raise RuntimeError("frozen data changed during runtime benchmark")
    if rank == 0:
        add_speedups(generation_results, "generation_wall_seconds")
        add_speedups(cached_results, "seconds_per_step")
        add_speedups(tiny_results, "wall_seconds")
        estimates = estimate_wall_times(generation_results, cached_results)
        comparison = comparison_summary(generation_results, cached_results, estimates)
        recommended = min(
            (name for name in ("P0", "P1") if correctness[name]["passed"]),
            key=lambda name: estimates[name]["combined_seconds_per_prompt"],
        )
        output = {
            "contract_version": "gr_user_runtime_benchmark_v1r",
            "frozen_contract": {
                "G": G,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "penalty_strategy": "sqrt",
                "lambda": 0.5,
                "epsilon": 0.2,
                "beta": 0.0,
                "learning_rate": args.learning_rate,
                "generation_timed_repeats": GENERATION_TIMED_REPEATS,
                "cached_warmup_steps": CACHED_WARMUP_STEPS,
                "cached_timed_steps": CACHED_TIMED_STEPS,
            },
            "selection": {
                "seed": SEED,
                "action_count": 20,
                "chain_count": 20,
                "sample_ids": {route: [row["sample_id"] for row in selected[route]] for route in ROUTES},
            },
            "generation": generation_results,
            "cached_rollout": {
                "path": str(args.cache_output),
                "prompt_groups": len(cache["groups"]),
                "candidate_count": len(cache["groups"]) * G,
                "shared_by_p0_p1": True,
                "fields": ["completion_ids", "rewards", "sequence_advantages", "token_advantages", "penalty_masks"],
            },
            "correctness_parity": correctness,
            "cached_policy_backward": cached_results,
            "tiny_end_to_end": tiny_results,
            "estimated_wall_seconds": estimates,
            "comparison": comparison,
            "recommended_config": recommended,
            "integrity": {
                "data_sha_before": sha_before,
                "data_sha_after": sha_after,
                "frozen_data_unchanged": sha_before == sha_after == EXPECTED_DATA_SHA,
                "formal_pilot_run": False,
                "pilot_150_run": False,
                "full_training_run": False,
                "p2_performance_tested": False,
                "gr_rec_v1_modified": False,
            },
        }
        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        args.result_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(args.docs_output, output)
        print(json.dumps({"status": "PASS", "recommended": recommended, "result": str(args.result_output)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
