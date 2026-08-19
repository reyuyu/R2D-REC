#!/usr/bin/env python3
"""Resume Pilot300 and consume the remaining GR_USER_v1 train epoch exactly once."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monitor.writer import monitor_from_env
from preflight_user_full_epoch import build_formal_plan, distribute_formal_step_rows, load_resume, select_remaining
from run_user_grpo_smoke import (
    EXPECTED_DATA_SHA,
    G,
    dataset_sha,
    generate_route,
    grad_norm,
    lora_delta,
    lora_snapshot,
    make_policy_batch,
    parameter_sha256,
    read_jsonl,
)
from run_user_pilot150 import (
    _all_gather_object,
    _all_reduce_scalar,
    _append_jsonl,
    _mean,
    aggregate_step_metrics,
    group_records_from_rollout,
    probe_step_summary,
    read_jsonl_if_present,
    rolling_summaries,
    route_metric_summary,
    validate_restored_optimizer,
)
from user_fixed_probe import aggregate_probe_steps, evaluate_user_fixed_probe, validate_probe_rows
from user_grpo_trainer import UserGRPOTrainer, prepare_scored_rollout
from user_monitor_adapter import UserMonitorAdapter


ROUTES = ("action", "chain")
START_STEP = 40
FINAL_STEP = 378
START_PROMPTS = 300
NEW_PROMPTS = 2700
CHECKPOINT_STEPS = (80, 160, 240, 320)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_config(config: dict) -> None:
    expected = {
        "seed": 20260820,
        "starting_global_step": 40,
        "starting_processed_prompts": 300,
        "remaining_action_prompts": 1350,
        "remaining_chain_prompts": 1350,
        "G": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "learning_rate": 1e-6,
        "epsilon": 0.2,
        "beta": 0.0,
        "population_std_correction": 0,
        "advantage_epsilon": 1e-4,
        "penalty_strategy": "sqrt",
        "lambda": 0.5,
        "forward_batch_size": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "runtime": "P0",
        "length_bucketing": False,
        "expected_new_optimizer_steps": 338,
        "expected_final_step": 378,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    cadence = config.get("monitor_cadence", {})
    expected_cadence = {"metrics_every": 1, "rollout_every": 5, "trace_every": 10, "probe_every": 25, "checkpoint_every": 80}
    cadence_mismatches = {key: (cadence.get(key), value) for key, value in expected_cadence.items() if cadence.get(key) != value}
    if mismatches or cadence_mismatches:
        raise RuntimeError(f"formal config drift: values={mismatches}, cadence={cadence_mismatches}")


def _restore_rank_rng(checkpoint: Path, rank: int, device) -> dict:
    state = torch.load(checkpoint / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    if state.get("global_optimizer_step") != START_STEP or state.get("processed_unique_prompts") != START_PROMPTS:
        raise RuntimeError(f"rank {rank} RNG cursor mismatch")
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    return {"rank": rank, "step": START_STEP, "processed": START_PROMPTS}


def _probe_due(step: int, final_step: int, every: int) -> bool:
    return step == final_step or (step > START_STEP and (step - START_STEP) % every == 0)


def _checkpoint_name(step: int, final: bool = False, emergency: bool = False) -> str:
    if final:
        return "full-epoch-final"
    if emergency:
        return f"emergency-step{step}"
    return f"checkpoint-step{step}"


def _markdown(summary: dict) -> str:
    probes = summary["fixed_probe"]["steps"]
    probe_rows = []
    for step, row in sorted(probes.items(), key=lambda item: int(item[0])):
        probe_rows.append(
            f"| {step} | {row['action_f1']:.6f} | {row['action_precision']:.6f} | "
            f"{row['action_recall']:.6f} | {row['chain_reward']:.6f} | "
            f"{row['chain_action_alignment']:.6f} | {row['chain_logic_alignment']:.6f} |"
        )
    runtime = summary["runtime"]
    return f"""# GR_USER_v1 Full Epoch

- Run: `{summary['run_id']}`
- Resume: step 40 / 300 prompts
- Final: step 378 / 3000 unique prompts
- Final checkpoint: `{summary['checkpoint_path']}`
- Internal status: **{summary['internal_status']}**

## Runtime

- Training wall: {runtime['training_wall_seconds']:.2f}s
- Probe wall: {runtime['probe_wall_seconds']:.2f}s ({runtime['probe_overhead_fraction']:.2%})
- Checkpoint wall: {runtime['checkpoint_wall_seconds']:.2f}s
- Total runner wall: {runtime['total_wall_seconds']:.2f}s
- Seconds/new prompt: {runtime['seconds_per_new_prompt']:.4f}
- Peak VRAM: {runtime['gpu_peak_vram_mib']:.1f} MiB

## Fixed light probe

| Step | Action F1 | Precision | Recall | Chain Total | Chain Action | Chain Logic |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(probe_rows)}

Final 40-sample Chain Probe v2 is run after training and recorded separately before external evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--probe-backfill", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/data/GRPO_USER/runs"))
    parser.add_argument("--result-output", type=Path, default=Path("/data/GRPO_USER/results/full_epoch_v1_summary.json"))
    parser.add_argument("--docs-output", type=Path, default=Path("/data/GRPO_USER/docs/full_epoch_v1.md"))
    args = parser.parse_args()

    total_started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("expected_final_step") != FINAL_STEP:
        raise RuntimeError("formal CPU preflight did not pass")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4 or local_rank not in range(4):
        raise RuntimeError("formal User epoch requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    run_dir = args.run_root / args.run_id
    monitor_dir = Path(os.environ.get("GRPO_MONITOR_DIR", "/data/GRPO/runs")) / args.run_id

    sha_before = dataset_sha(Path(config["dataset"]).parent)
    if sha_before != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen data SHA mismatch")
    resume_checkpoint = Path(config["resume_checkpoint"])
    resume = load_resume(resume_checkpoint)
    rows = read_jsonl(config["dataset"])
    selected = select_remaining(rows, resume["consumed_ids"], int(config["seed"]))
    plan = build_formal_plan(selected, START_STEP)
    current_ids = {route: [row["sample_id"] for row in selected[route]] for route in ROUTES}
    current_flat_ids = [sample_id for route in ROUTES for sample_id in current_ids[route]]
    cumulative_ids = list(resume["consumed_ids"]) + current_flat_ids
    train_ids = {row["sample_id"] for row in rows}
    if len(current_flat_ids) != NEW_PROMPTS or len(set(cumulative_ids)) != 3000 or set(cumulative_ids) != train_ids:
        raise RuntimeError("formal sample coverage contract failed")

    probe_config = config["fixed_probe"]
    probe_rows = read_jsonl(probe_config["dataset"])
    validate_probe_rows(probe_rows)
    probe_backfill = read_jsonl_if_present(args.probe_backfill)
    if len(probe_backfill) != 6 or collections.Counter((int(row["step"]), row["route"]) for row in probe_backfill) != {(40, "action"): 3, (40, "chain"): 3}:
        raise RuntimeError("light-probe Step40 backfill contract failed")

    manifest = {
        "run_id": args.run_id,
        "run_kind": "user_grpo",
        "experiment": config["experiment"],
        "status": "running",
        "start_time": utc_now(),
        "git_commit": args.git_commit,
        "runner": config["runner"],
        "world_size": world,
        "dataset": config["dataset"],
        "dataset_sha": sha_before,
        "parent_checkpoint": resume["metadata"]["parent_checkpoint"],
        "resume_from_run": config["resume_run_id"],
        "resume_checkpoint": str(resume_checkpoint),
        "cumulative_start_prompts": START_PROMPTS,
        "cumulative_start_step": START_STEP,
        "remaining_sample_ids": current_ids,
        "cumulative_sample_ids": cumulative_ids,
        "expected_final_step": FINAL_STEP,
        "expected_total_unique": 3000,
        "fixed_probe": probe_config,
        "checkpoint_order": "optimizer -> scalar -> probe -> checkpoint -> barrier",
        "frozen_contract": config,
        "output_dir": str(run_dir),
    }
    if is_main:
        if run_dir.exists() or monitor_dir.exists():
            raise FileExistsError("formal RUN_ID already exists")
        run_dir.mkdir(parents=True)
    dist.barrier()
    monitor_writer = monitor_from_env(args.run_id, rank)
    monitor = UserMonitorAdapter(monitor_writer)
    if is_main:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not monitor.write_manifest(**manifest):
            raise RuntimeError("failed to write monitor manifest")
        for event in probe_backfill:
            _append_jsonl(run_dir / "probes.jsonl", event)
            if not monitor.write_probe(event):
                raise RuntimeError("failed to write Step40 light probe")
    dist.barrier()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], dtype=torch.bfloat16, device_map={"": device}, local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(base_model, resume_checkpoint, is_trainable=True, local_files_only=True)
    for module in peft_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in peft_model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    model = DistributedDataParallel(peft_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name.lower() for name, _ in trainable):
        raise RuntimeError("only LoRA parameters may be trainable")
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=config["learning_rate"], weight_decay=0.0)
    optimizer_state = torch.load(resume_checkpoint / "optimizer.pt", map_location=device, weights_only=True)
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state
    optimizer_resume_audit = validate_restored_optimizer(optimizer, START_STEP)
    if any(group["lr"] != config["learning_rate"] for group in optimizer.param_groups):
        raise RuntimeError("restored optimizer LR drift")
    rng_resume_audit = _restore_rank_rng(resume_checkpoint, rank, device)
    rng_resume_audits = _all_gather_object(rng_resume_audit)
    model.module.config.use_cache = False
    model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if is_main:
        base_sha_before, base_tensor_count = parameter_sha256(model, include_lora=False)
        lora_sha_before, lora_tensor_count = parameter_sha256(model, include_lora=True)
        lora_before = lora_snapshot(model)
    else:
        base_sha_before = lora_sha_before = None
        base_tensor_count = lora_tensor_count = 0
        lora_before = None
    dist.barrier()

    route_positions = {route: {row["sample_id"]: index for index, row in enumerate(selected[route])} for route in ROUTES}
    all_group_records = []
    all_step_records = []
    processed = START_PROMPTS
    completed_plan_items = 0
    live_probe_wall_seconds = []
    checkpoint_wall_seconds = []
    checkpoint_records = []
    consecutive_grad_explosion = 0
    consecutive_high_clip = 0
    safety_stop_reasons = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    def run_fixed_probe(step: int, reason: str) -> None:
        nonlocal processed
        python_before = random.getstate()
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        training_before = model.module.training
        processed_before = processed
        optimizer_steps_before = {int(state["step"]) for state in optimizer.state.values() if "step" in state}
        events = evaluate_user_fixed_probe(
            model.module, tokenizer, probe_rows, device, rank=rank, world_size=world,
            step=step, reason=reason, seed=int(probe_config["seed"]), generate_fn=generate_route,
        )
        if random.getstate() != python_before or not torch.equal(torch.get_rng_state(), cpu_before) or not torch.equal(torch.cuda.get_rng_state(device), cuda_before):
            raise RuntimeError("fixed probe changed training RNG")
        if model.module.training != training_before or processed != processed_before:
            raise RuntimeError("fixed probe changed model mode or sampler cursor")
        optimizer_steps_after = {int(state["step"]) for state in optimizer.state.values() if "step" in state}
        if optimizer_steps_after != optimizer_steps_before:
            raise RuntimeError("fixed probe changed optimizer state")
        if is_main:
            live_probe_wall_seconds.append(float(events[0]["probe_wall_sec"]))
            for event in events:
                _append_jsonl(run_dir / "probes.jsonl", event)
                if not monitor.write_probe(event):
                    raise RuntimeError("failed to write fixed probe")
            print(json.dumps({"probe_step": step, **aggregate_probe_steps(events)[0]}, ensure_ascii=False), flush=True)
        dist.barrier()

    def save_checkpoint(step: int, cursor: int, *, final: bool = False, emergency: bool = False) -> tuple[Path, float]:
        started = time.perf_counter()
        directory = run_dir / _checkpoint_name(step, final=final, emergency=emergency)
        consumed_now = list(resume["consumed_ids"]) + [
            row["sample_id"] for item in plan[:cursor] for row in item["rows"]
        ]
        if is_main:
            directory.mkdir()
        dist.barrier()
        torch.save({
            "rank": rank,
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
            "numpy_rng_state": None,
            "global_optimizer_step": step,
            "processed_unique_prompts": processed,
            "training_plan_cursor": cursor,
        }, directory / f"rng_rank{rank}.pt")
        dist.barrier()
        if is_main:
            model.module.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            torch.save(optimizer.state_dict(), directory / "optimizer.pt")
            optimizer_audit = validate_restored_optimizer(optimizer, step)
            metadata = {
                "run_id": args.run_id,
                "processed_unique_prompts": processed,
                "global_optimizer_step": step,
                "optimizer_steps": step,
                "git_commit": args.git_commit,
                "parent_checkpoint": manifest["parent_checkpoint"],
                "resume_from_run": config["resume_run_id"],
                "resume_checkpoint": str(resume_checkpoint),
                "dataset_sha": sha_before,
                "training_config": config,
                "probe_light_sha": probe_config["sha256"],
                "consumed_sample_ids": consumed_now,
                "sampler_cursor": {"completed_plan_items": cursor, "next_global_step": step + 1},
                "rng_state_files": [f"rng_rank{value}.pt" for value in range(world)],
                "optimizer_resume_audit": optimizer_resume_audit,
                "optimizer_current_audit": optimizer_audit,
                "emergency": emergency,
                "created_at": utc_now(),
            }
            (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            required = [directory / "adapter_model.safetensors", directory / "optimizer.pt", directory / "metadata.json"]
            required += [directory / f"rng_rank{value}.pt" for value in range(world)]
            if not all(path.is_file() for path in required) or len(consumed_now) != len(set(consumed_now)):
                raise RuntimeError("checkpoint lightweight integrity failed")
        dist.barrier()
        elapsed = _all_reduce_scalar(time.perf_counter() - started, dist.ReduceOp.MAX)
        if is_main:
            checkpoint_records.append({"step": step, "path": str(directory), "wall_seconds": elapsed, "final": final, "emergency": emergency})
            checkpoint_wall_seconds.append(elapsed)
        return directory, elapsed

    for plan_index, plan_item in enumerate(plan):
        step = plan_item["step"]
        route = plan_item["route"]
        real_rows, dummy, loss_scale = distribute_formal_step_rows(plan_item["rows"], rank, world)
        generation_started = time.perf_counter()
        random.seed(config["seed"] + step * 100 + rank)
        torch.manual_seed(config["seed"] + step * 100 + rank)
        torch.cuda.manual_seed_all(config["seed"] + step * 100 + rank)
        model.eval()
        completions = generate_route(model.module, tokenizer, real_rows, device)
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started

        scoring_started = time.perf_counter()
        rollout = prepare_scored_rollout(real_rows, completions, tokenizer)
        batch = make_policy_batch(tokenizer, rollout, device)
        torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - scoring_started
        route_start = route_positions[route][real_rows[0]["sample_id"]]
        local_groups, local_traces = group_records_from_rollout(rollout, real_rows, route_start, step)
        objective_payload = {
            "per_kind_masked_token_count": rollout["metrics"]["per_kind_masked_token_count"],
            "per_kind_incremental_negative_mass": rollout["metrics"]["per_kind_incremental_negative_mass"],
        }

        model.eval()
        combined = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        with torch.no_grad():
            batch["old_per_token_logps"] = trainer._get_user_per_token_logps(model, combined, attention, batch["completion_ids"].size(1)).detach()
        optimizer.zero_grad(set_to_none=True)
        model.train()
        policy_started = time.perf_counter()
        local_loss = trainer._compute_loss(model, batch) * loss_scale
        if not torch.isfinite(local_loss):
            raise RuntimeError("formal loss is NaN or Inf")
        local_loss.backward()
        torch.cuda.synchronize(device)
        policy_seconds = time.perf_counter() - policy_started
        raw_grad_norm = grad_norm([parameter for _, parameter in trainable])
        if not math.isfinite(raw_grad_norm) or raw_grad_norm <= 0.0:
            raise RuntimeError("formal LoRA gradient norm is invalid")
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)

        global_prompt_count = len(plan_item["rows"])
        global_loss = _all_reduce_scalar(float(local_loss.detach())) / world
        ratio_mean = _all_reduce_scalar(trainer.last_loss_metrics["ratio_mean"] * len(real_rows)) / global_prompt_count
        clip_fraction = _all_reduce_scalar(trainer.last_loss_metrics["clip_fraction"] * len(real_rows)) / global_prompt_count
        global_grad_norm = _all_reduce_scalar(raw_grad_norm, dist.ReduceOp.MAX)
        generation_seconds = _all_reduce_scalar(generation_seconds, dist.ReduceOp.MAX)
        scoring_seconds = _all_reduce_scalar(scoring_seconds, dist.ReduceOp.MAX)
        policy_seconds = _all_reduce_scalar(policy_seconds, dist.ReduceOp.MAX)
        gathered = _all_gather_object({"groups": local_groups, "traces": local_traces, "objective": objective_payload})
        processed += global_prompt_count
        completed_plan_items = plan_index + 1
        monitor_writer.write_rank({
            "rollout_id": step, "step": step, "route": route, "real_prompt_count": len(real_rows),
            "dummy": dummy, "loss_scale": loss_scale, "generation_wall_sec": generation_seconds,
            "policy_wall_sec": policy_seconds,
        })

        if is_main:
            groups = sorted([record for payload in gathered for record in payload["groups"]], key=lambda record: record["route_index"])
            traces = [record for payload in gathered for record in payload["traces"]]
            rollout_metrics = aggregate_step_metrics(groups, [payload["objective"] for payload in gathered])
            policy_metrics = {
                "loss": global_loss, "grad_norm": global_grad_norm, "learning_rate": config["learning_rate"],
                "ratio_mean": ratio_mean, "clip_fraction": clip_fraction, "policy_wall_sec": policy_seconds,
            }
            step_event = {
                "step": step, "route": route, "processed_unique_prompts": processed,
                "step_unique_prompts": global_prompt_count, "generation_wall_sec": generation_seconds,
                "reward_mask_wall_sec": scoring_seconds, **policy_metrics, **rollout_metrics,
            }
            all_group_records.extend(groups)
            all_step_records.append(step_event)
            for record in groups:
                _append_jsonl(run_dir / "groups.jsonl", record)
            _append_jsonl(run_dir / "steps.jsonl", step_event)
            monitor.write_step(
                step=step, route=route, rollout_metrics=rollout_metrics, policy_metrics=policy_metrics,
                processed_unique_prompts=processed, step_unique_prompts=global_prompt_count,
                generation_wall_sec=generation_seconds, reward_mask_wall_sec=scoring_seconds,
            )
            monitor.write_rollout({
                "rollout_id": step, "step": step, "route": route, "g": G,
                "group_ids": [record["sample_id"] for record in groups],
                "reward_mean": rollout_metrics["task_reward_mean"], "reward_std": rollout_metrics["task_reward_std"],
                "zero_std_ratio": rollout_metrics["zero_std_ratio"],
                "completion_length_mean": _mean([record["completion_token_mean"] for record in groups]),
                "masked_candidate_rate": rollout_metrics["masked_candidate_rate"],
                "masked_token_rate": rollout_metrics["masked_token_rate"],
            })
            selected_trace = min(traces, key=lambda record: record["route_index"])
            monitor.write_trace({
                "rollout_id": step, "step": step, "route": route,
                "group_id": selected_trace["sample_id"], "candidates": selected_trace["candidates"],
            })
            safety = config["safety"]
            consecutive_grad_explosion = consecutive_grad_explosion + 1 if global_grad_norm > safety["grad_explosion_threshold"] else 0
            consecutive_high_clip = consecutive_high_clip + 1 if clip_fraction > safety["clip_fraction_threshold"] else 0
            if consecutive_grad_explosion >= safety["grad_explosion_consecutive_steps"]:
                safety_stop_reasons.append("consecutive_grad_explosion")
            if consecutive_high_clip >= safety["clip_fraction_consecutive_steps"]:
                safety_stop_reasons.append("sustained_high_clip_fraction")
            if not all(math.isfinite(value) for value in (global_loss, global_grad_norm, ratio_mean, clip_fraction, rollout_metrics["task_reward_std"], rollout_metrics["masked_token_rate"])):
                safety_stop_reasons.append("nan_or_inf_metric")
            if monitor_writer.errors:
                safety_stop_reasons.append("monitor_write_error")
            print(json.dumps({
                "step": step, "route": route, "processed": processed, "loss": global_loss,
                "reward": rollout_metrics["task_reward_mean"], "grad_norm": global_grad_norm,
                "clip_fraction": clip_fraction,
            }, ensure_ascii=False), flush=True)

        stop = torch.tensor(int(bool(safety_stop_reasons) if is_main else 0), device=device)
        dist.broadcast(stop, 0)
        if bool(stop.item()):
            save_checkpoint(step, completed_plan_items, emergency=True)
            if is_main:
                (run_dir / "STOPPED.json").write_text(json.dumps({"step": step, "reasons": safety_stop_reasons}, indent=2), encoding="utf-8")
            raise RuntimeError(f"formal safety stop: {safety_stop_reasons}")

        if _probe_due(step, FINAL_STEP, int(probe_config["every_steps"])):
            run_fixed_probe(step, "final" if step == FINAL_STEP else "periodic")
        if step in CHECKPOINT_STEPS:
            save_checkpoint(step, completed_plan_items)

    final_checkpoint, _ = save_checkpoint(FINAL_STEP, completed_plan_items, final=True)
    dist.barrier()
    training_elapsed_inclusive = time.perf_counter() - training_started
    live_probe_wall = _all_reduce_scalar(sum(live_probe_wall_seconds) if is_main else 0.0, dist.ReduceOp.MAX)
    checkpoint_wall = _all_reduce_scalar(sum(checkpoint_wall_seconds) if is_main else 0.0, dist.ReduceOp.MAX)
    training_wall = training_elapsed_inclusive - live_probe_wall - checkpoint_wall
    step40_probe_wall = max(float(row["probe_wall_sec"]) for row in probe_backfill)
    total_probe_wall = step40_probe_wall + live_probe_wall
    local_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peaks = _all_gather_object(local_peak)

    if is_main:
        base_sha_after, _ = parameter_sha256(model, include_lora=False)
        lora_sha_after, _ = parameter_sha256(model, include_lora=True)
        lora_delta_l2, lora_delta_max = lora_delta(model, lora_before)
    else:
        base_sha_after = lora_sha_after = None
        lora_delta_l2 = lora_delta_max = 0.0
    dist.barrier()
    sha_after = dataset_sha(Path(config["dataset"]).parent)
    if sha_after != sha_before:
        raise RuntimeError("frozen data changed during formal epoch")

    if is_main:
        if base_sha_before != base_sha_after:
            raise RuntimeError("base model changed during formal epoch")
        if lora_sha_before == lora_sha_after or lora_delta_l2 <= 0.0:
            raise RuntimeError("LoRA did not update during formal epoch")
        final_metadata = json.loads((final_checkpoint / "metadata.json").read_text(encoding="utf-8"))
        final_consumed = final_metadata["consumed_sample_ids"]
        duplicate = len(final_consumed) - len(set(final_consumed))
        missing = len(train_ids - set(final_consumed))
        if processed != 3000 or duplicate or missing or set(final_consumed) != train_ids:
            raise RuntimeError("final sample coverage audit failed")
        if len(all_group_records) != NEW_PROMPTS or len(all_step_records) != 338:
            raise RuntimeError("formal metric coverage failed")
        probes = probe_step_summary(read_jsonl_if_present(run_dir / "probes.jsonl"))
        expected_probe_steps = {str(value) for value in list(range(40, 366, 25)) + [378]}
        if set(probes) != expected_probe_steps:
            raise RuntimeError(f"light probe timeline incomplete: {sorted(probes)}")
        total_masked = sum(record["masked_token_count"] for record in all_group_records)
        total_valid = sum(record["valid_token_count"] for record in all_group_records)
        global_metrics = {
            "reward_std": _mean([record["reward_std"] for record in all_group_records]),
            "zero_std_ratio": _mean([float(record["zero_std"]) for record in all_group_records]),
            "grad_norm_mean": _mean([record["grad_norm"] for record in all_step_records]),
            "grad_norm_max": max(record["grad_norm"] for record in all_step_records),
            "clip_fraction_mean": _mean([record["clip_fraction"] for record in all_step_records]),
            "clip_fraction_max": max(record["clip_fraction"] for record in all_step_records),
            "masked_token_rate": total_masked / total_valid,
            "base_delta": 0.0,
            "base_sha_before": base_sha_before,
            "base_sha_after": base_sha_after,
            "base_tensor_count": base_tensor_count,
            "lora_delta_l2": lora_delta_l2,
            "lora_delta_max_abs": lora_delta_max,
            "lora_sha_before": lora_sha_before,
            "lora_sha_after": lora_sha_after,
            "lora_tensor_count": lora_tensor_count,
            "nan_or_inf": False,
        }
        summary = {
            "contract_version": config["contract_version"],
            "run_id": args.run_id,
            "git_commit": args.git_commit,
            "checkpoint_path": str(final_checkpoint),
            "resume": {
                "checkpoint": str(resume_checkpoint), "starting_step": START_STEP,
                "ending_step": FINAL_STEP, "starting_processed_prompts": START_PROMPTS,
                "optimizer_state_restored": True, "optimizer_resume_audit": optimizer_resume_audit,
                "rng_state_restored": True, "rng_resume_audit": rng_resume_audits,
            },
            "samples": {
                "starting_consumed": START_PROMPTS, "new_consumed": NEW_PROMPTS,
                "total_unique": len(set(final_consumed)), "action": 1500, "chain": 1500,
                "duplicate": duplicate, "missing": missing,
            },
            "runtime": {
                "training_wall_seconds": training_wall,
                "probe_wall_seconds": total_probe_wall,
                "live_probe_wall_seconds": live_probe_wall,
                "step40_backfill_probe_wall_seconds": step40_probe_wall,
                "probe_overhead_fraction": total_probe_wall / (training_wall + total_probe_wall),
                "checkpoint_wall_seconds": checkpoint_wall,
                "total_wall_seconds": time.perf_counter() - total_started,
                "seconds_per_new_prompt": training_wall / NEW_PROMPTS,
                "gpu_peak_vram_mib": max(peaks), "per_gpu_peak_vram_mib": peaks,
            },
            "cadence": config["monitor_cadence"],
            "checkpoints": checkpoint_records,
            "fixed_probe": {
                "dataset": probe_config["dataset"], "sha256": probe_config["sha256"],
                "steps": probes,
                "step40_to_final": {
                    key: probes["378"][key] - probes["40"][key]
                    for key in ("action_f1", "action_precision", "action_recall", "chain_reward", "chain_action_alignment", "chain_logic_alignment")
                },
                "total_wall_seconds": total_probe_wall,
            },
            "training_metrics": {
                "remainder2700": route_metric_summary(all_group_records),
                "rolling_means": rolling_summaries(all_group_records),
            },
            "global": global_metrics,
            "integrity": {
                "base_frozen": base_sha_before == base_sha_after,
                "lora_updated": lora_sha_before != lora_sha_after and lora_delta_l2 > 0,
                "nan_or_inf": False,
                "dataset_sha_before": sha_before, "dataset_sha_after": sha_after,
                "dataset_sha_unchanged": sha_before == sha_after == EXPECTED_DATA_SHA,
                "optimizer_final_audit": final_metadata["optimizer_current_audit"],
                "rng_state_files": final_metadata["rng_state_files"],
                "sample_coverage_exact": duplicate == 0 and missing == 0 and set(final_consumed) == train_ids,
                "external_evaluation_run": False,
            },
            "internal_status": "PENDING_FINAL_CHAIN_PROBE_V2",
        }
        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        args.docs_output.parent.mkdir(parents=True, exist_ok=True)
        args.result_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        args.docs_output.write_text(_markdown(summary), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        completed_manifest = {
            **manifest, "status": "completed_training_pending_final_probe_v2", "completed_at": utc_now(),
            "actual_unique_prompts": processed, "actual_optimizer_steps": FINAL_STEP,
            "checkpoint_path": str(final_checkpoint), "internal_status": summary["internal_status"],
        }
        (run_dir / "manifest.json").write_text(json.dumps(completed_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        monitor.write_manifest(**completed_manifest)
        print(json.dumps({"status": "PASS", "run_id": args.run_id, "checkpoint": str(final_checkpoint), "next": "FINAL_CHAIN_PROBE_V2"}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
