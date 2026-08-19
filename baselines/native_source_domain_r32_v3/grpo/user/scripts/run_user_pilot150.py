#!/usr/bin/env python3
"""Run the frozen GR_USER_v1 150-prompt directional Pilot on four GPUs."""

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
from user_grpo_trainer import UserGRPOTrainer, prepare_scored_rollout
from user_monitor_adapter import UserMonitorAdapter


ROUTES = ("action", "chain")
ACTION_BUCKETS = (
    ("1-5", 1, 5, 13),
    ("6-10", 6, 10, 13),
    ("11-20", 11, 20, 13),
    ("21-30", 21, 30, 12),
    ("31-40", 31, 40, 12),
    ("41+", 41, 10**9, 12),
)
CHAIN_BUCKETS = (("2", 2, 20), ("3", 3, 20), ("4", 4, 20), ("5", 5, 15))
SEGMENTS = {"early": (0, 19), "middle": (19, 56), "late": (56, 75)}
ACTION_FIELDS = (
    "f1",
    "precision",
    "recall",
    "wrong_selection",
    "hallucination",
    "duplicate",
)
CHAIN_FIELDS = (
    "reward",
    "action_alignment",
    "logic_alignment",
    "date_mismatch",
    "action_mismatch",
)
HISTORICAL_BASELINE = {
    "action": {
        "f1": 0.625160,
        "hallucination": 0.065,
        "duplicate": 0.040,
        "wrong_selection": 0.735,
    },
    "chain": {
        "reward": 0.457633,
        "action_alignment": 0.669349,
        "logic_alignment": 0.245917,
        "date_mismatch": 0.305,
        "action_mismatch": 0.140,
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _spread(rows, count):
    ordered = sorted(rows, key=lambda row: (row["prompt_token_count"], row["sample_id"]))
    if len(ordered) < count:
        raise ValueError(f"selection bucket has {len(ordered)} rows but needs {count}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    selected = [ordered[index] for index in indices]
    if len({row["sample_id"] for row in selected}) != count:
        raise AssertionError("spread selection repeated a sample")
    return selected


def select_pilot_rows(rows, seed):
    action_pool = [row for row in rows if row["route"] == "action"]
    chain_pool = [row for row in rows if row["route"] == "chain"]
    selected = {"action": [], "chain": []}
    bucket_counts = {"action": {}, "chain": {}}
    for label, low, high, count in ACTION_BUCKETS:
        chosen = _spread(
            [row for row in action_pool if low <= int(row["gold_sid_count"]) <= high],
            count,
        )
        selected["action"].extend(chosen)
        bucket_counts["action"][label] = len(chosen)
    for label, event_count, count in CHAIN_BUCKETS:
        chosen = _spread(
            [row for row in chain_pool if int(row["gold_event_count"]) == event_count],
            count,
        )
        selected["chain"].extend(chosen)
        bucket_counts["chain"][label] = len(chosen)
    random.Random(seed + 1).shuffle(selected["action"])
    random.Random(seed + 2).shuffle(selected["chain"])
    all_ids = [row["sample_id"] for route in ROUTES for row in selected[route]]
    if len(selected["action"]) != 75 or len(selected["chain"]) != 75:
        raise AssertionError("Pilot selection must be 75 Action + 75 Chain")
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Pilot selection contains duplicate sample IDs")
    length_ranges = {}
    for route in ROUTES:
        lengths = sorted(int(row["prompt_token_count"]) for row in selected[route])
        length_ranges[route] = {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "max": lengths[-1],
        }
    return selected, {"bucket_counts": bucket_counts, "prompt_length_coverage": length_ranges}


def build_training_plan(selected):
    chunks = {
        route: [selected[route][start : start + 8] for start in range(0, 75, 8)]
        for route in ROUTES
    }
    plan = []
    route_offsets = {route: 0 for route in ROUTES}
    for index in range(10):
        for route in ROUTES:
            rows = chunks[route][index]
            plan.append(
                {
                    "step": len(plan) + 1,
                    "route": route,
                    "rows": rows,
                    "route_start": route_offsets[route],
                }
            )
            route_offsets[route] += len(rows)
    if [len(item["rows"]) for item in plan] != [value for _ in range(9) for value in (8, 8)] + [3, 3]:
        raise AssertionError("Pilot plan must use 18 full steps and two 3-prompt tail steps")
    if route_offsets != {"action": 75, "chain": 75} or len(plan) != 20:
        raise AssertionError("Pilot plan coverage drift")
    return plan


def distribute_step_rows(rows, rank, world_size=4):
    if world_size != 4 or len(rows) not in (3, 8):
        raise ValueError("Pilot DDP steps support global prompt counts 8 or 3")
    if len(rows) == 8:
        real_rows = rows[rank * 2 : (rank + 1) * 2]
        return real_rows, False, 1.0
    if rank < 3:
        return [rows[rank]], False, world_size / len(rows)
    return [], True, 0.0


def _all_reduce_scalar(value, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(float(value), device=torch.cuda.current_device(), dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor)


def _all_gather_object(value):
    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return values


def _append_jsonl(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _pstdev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def candidate_grounding(score):
    statuses = [item["status"] for item in score.grounding_status]
    if statuses and all(status == "grounded" for status in statuses):
        return "grounded"
    if not statuses or all(status == "ungrounded" for status in statuses):
        return "ungrounded"
    return "partially_grounded"


def _masked_spans(compiled):
    spans = []
    seen = set()
    for record in compiled["records"]:
        if not record.get("included"):
            continue
        for span in record.get("token_spans", []):
            start = span.get("char_start")
            end = span.get("char_end")
            key = (start, end, record["kind"])
            if isinstance(start, int) and isinstance(end, int) and end > start and key not in seen:
                spans.append({"start": start, "end": end, "kind": record["kind"]})
                seen.add(key)
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["kind"]))


def _candidate_trace(rollout, candidate_index, route):
    score = rollout["scores"][candidate_index]
    compiled = rollout["compiled_penalties"][candidate_index]
    violation_kinds = [violation.kind for violation in score.violations]
    record = {
        "candidate_id": candidate_index % G,
        "route": route,
        "completion": rollout["completions"][candidate_index],
        "completion_length": rollout["completion_lengths"][candidate_index],
        "reward": float(rollout["rewards"][candidate_index]),
        "sequence_advantage": float(rollout["sequence_advantages"][candidate_index]),
        "violations": violation_kinds,
        "masked_spans": _masked_spans(compiled),
        "masked_token_count": int(rollout["local_penalty_mask"][candidate_index].sum()),
    }
    if route == "action":
        record.update(
            f1=score.f1,
            precision=score.precision,
            recall=score.recall,
            gold_sids=score.gold_sids,
            pred_sids=score.pred_sids_unique,
        )
    else:
        record.update(action_alignment=score.action_f1, logic_alignment=score.logic_f1)
    return record


def group_records_from_rollout(rollout, rows, route_start, step):
    records = []
    traces = []
    for row_index, row in enumerate(rows):
        start = row_index * G
        stop = start + G
        scores = rollout["scores"][start:stop]
        rewards = [float(value) for value in rollout["rewards"][start:stop]]
        sequence = [float(value) for value in rollout["sequence_advantages"][start:stop]]
        token_values = []
        masked_tokens = valid_tokens = masked_candidates = positive_flips = 0
        violation_counts = collections.Counter()
        for candidate, score in enumerate(scores, start=start):
            length = rollout["completion_lengths"][candidate]
            values = rollout["token_advantages"][candidate, :length]
            mask = rollout["local_penalty_mask"][candidate, :length]
            token_values.extend(float(value) for value in values)
            masked_tokens += int(mask.sum())
            valid_tokens += length
            masked_candidates += int(bool(mask.any()))
            positive_flips += int(((values < 0) & mask & (rollout["sequence_advantages"][candidate] > 0)).sum())
            violation_counts.update(violation.kind for violation in score.violations)
        base = {
            "sample_id": row["sample_id"],
            "route": rollout["route"],
            "route_index": route_start + row_index,
            "step": step,
            "reward_values": rewards,
            "reward": _mean(rewards),
            "reward_std": _pstdev(rewards),
            "zero_std": _pstdev(rewards) == 0.0,
            "sequence_advantage_mean": _mean(sequence),
            "sequence_advantage_std": _pstdev(sequence),
            "token_advantage_sum": sum(token_values),
            "token_advantage_squared_sum": sum(value * value for value in token_values),
            "token_count": len(token_values),
            "masked_candidate_count": masked_candidates,
            "masked_token_count": masked_tokens,
            "valid_token_count": valid_tokens,
            "positive_sequence_masked_token_flip_count": positive_flips,
            "violation_counts": dict(violation_counts),
            "completion_token_mean": _mean(rollout["completion_lengths"][start:stop]),
        }
        if rollout["route"] == "action":
            base.update(
                f1=_mean([score.f1 for score in scores]),
                precision=_mean([score.precision for score in scores]),
                recall=_mean([score.recall for score in scores]),
                exact_match=_mean([float(score.exact_set_match) for score in scores]),
                wrong_selection=_mean([
                    float(any(item.kind == "wrong_selection_sid" for item in score.violations))
                    for score in scores
                ]),
                hallucination=_mean([
                    float(any(item.kind == "hallucinated_sid" for item in score.violations))
                    for score in scores
                ]),
                duplicate=_mean([
                    float(any(item.kind == "duplicate_sid" for item in score.violations))
                    for score in scores
                ]),
            )
        else:
            statuses = [candidate_grounding(score) for score in scores]
            base.update(
                action_alignment=_mean([score.action_f1 for score in scores]),
                logic_alignment=_mean([score.logic_f1 for score in scores]),
                date_mismatch=_mean([
                    float(any(item.kind == "date_mismatch" for item in score.violations))
                    for score in scores
                ]),
                action_mismatch=_mean([
                    float(any(item.kind == "action_mismatch" for item in score.violations))
                    for score in scores
                ]),
                hallucination=_mean([
                    float(any(item.kind == "hallucinated_sid" for item in score.violations))
                    for score in scores
                ]),
                duplicate=_mean([
                    float(any(item.kind == "duplicate_event" for item in score.violations))
                    for score in scores
                ]),
                chronology=_mean([
                    float(any(item.kind == "chronology_violation" for item in score.violations))
                    for score in scores
                ]),
                grounded=_mean([float(status == "grounded") for status in statuses]),
                partially_grounded=_mean([float(status == "partially_grounded") for status in statuses]),
                ungrounded=_mean([float(status == "ungrounded") for status in statuses]),
            )
        records.append(base)
        traces.append(
            {
                "sample_id": row["sample_id"],
                "route_index": route_start + row_index,
                "candidates": [
                    _candidate_trace(rollout, candidate, rollout["route"])
                    for candidate in range(start, stop)
                ],
            }
        )
    return records, traces


def aggregate_step_metrics(group_records, objective_payloads):
    if not group_records:
        raise RuntimeError("step contains no real prompt groups")
    route = group_records[0]["route"]
    if any(record["route"] != route for record in group_records):
        raise RuntimeError("step mixed Action and Chain groups")
    rewards = [value for record in group_records for value in record["reward_values"]]
    token_sum = sum(record["token_advantage_sum"] for record in group_records)
    token_squared = sum(record["token_advantage_squared_sum"] for record in group_records)
    token_count = sum(record["token_count"] for record in group_records)
    token_mean = token_sum / token_count
    token_std = math.sqrt(max(0.0, token_squared / token_count - token_mean**2))
    violation_counts = collections.Counter()
    kind_counts = collections.Counter()
    kind_mass = collections.Counter()
    positive_flips = 0
    for record in group_records:
        violation_counts.update(record["violation_counts"])
        positive_flips += record["positive_sequence_masked_token_flip_count"]
    for payload in objective_payloads:
        kind_counts.update(payload["per_kind_masked_token_count"])
        kind_mass.update(payload["per_kind_incremental_negative_mass"])
    candidates = len(group_records) * G
    valid_tokens = sum(record["valid_token_count"] for record in group_records)
    metrics = {
        "task_reward_mean": _mean(rewards),
        "task_reward_std": _pstdev(rewards),
        "zero_std_ratio": _mean([float(record["zero_std"]) for record in group_records]),
        "sequence_advantage_mean": _mean([
            record["sequence_advantage_mean"] for record in group_records
        ]),
        "sequence_advantage_std": _mean([
            record["sequence_advantage_std"] for record in group_records
        ]),
        "token_advantage_mean": token_mean,
        "token_advantage_std": token_std,
        "masked_candidate_rate": sum(record["masked_candidate_count"] for record in group_records) / candidates,
        "masked_token_rate": sum(record["masked_token_count"] for record in group_records) / valid_tokens,
        "positive_sequence_masked_token_flip_count": positive_flips,
        "per_kind_masked_token_count": dict(kind_counts),
        "per_kind_incremental_negative_mass": dict(kind_mass),
        "violation_counts": dict(violation_counts),
    }
    if route == "action":
        metrics.update(
            f1_mean=_mean([record["f1"] for record in group_records]),
            precision_mean=_mean([record["precision"] for record in group_records]),
            recall_mean=_mean([record["recall"] for record in group_records]),
            exact_match_rate=_mean([record["exact_match"] for record in group_records]),
            wrong_selection_candidate_rate=_mean([record["wrong_selection"] for record in group_records]),
            hallucination_candidate_rate=_mean([record["hallucination"] for record in group_records]),
            duplicate_candidate_rate=_mean([record["duplicate"] for record in group_records]),
        )
    else:
        metrics.update(
            total_reward_mean=metrics["task_reward_mean"],
            action_alignment_mean=_mean([record["action_alignment"] for record in group_records]),
            logic_alignment_mean=_mean([record["logic_alignment"] for record in group_records]),
            date_mismatch_candidate_rate=_mean([record["date_mismatch"] for record in group_records]),
            action_mismatch_candidate_rate=_mean([record["action_mismatch"] for record in group_records]),
            grounded_rate=_mean([record["grounded"] for record in group_records]),
            partially_grounded_rate=_mean([
                record["partially_grounded"] for record in group_records
            ]),
            ungrounded_rate=_mean([record["ungrounded"] for record in group_records]),
        )
    return metrics


def segment_summaries(group_records):
    output = {}
    for route, fields in (("action", ACTION_FIELDS), ("chain", CHAIN_FIELDS)):
        route_records = sorted(
            [record for record in group_records if record["route"] == route],
            key=lambda record: record["route_index"],
        )
        if len(route_records) != 75:
            raise RuntimeError(f"{route} summary expected 75 groups")
        output[route] = {}
        for name, (start, stop) in SEGMENTS.items():
            records = route_records[start:stop]
            output[route][name] = {
                "prompt_count": len(records),
                **{field: _mean([record[field] for record in records]) for field in fields},
            }
    return output


def rolling_summaries(group_records, window=10):
    output = {}
    for route, fields in (("action", ACTION_FIELDS), ("chain", CHAIN_FIELDS)):
        records = sorted(
            [record for record in group_records if record["route"] == route],
            key=lambda record: record["route_index"],
        )
        points = []
        for stop in range(window, len(records) + 1):
            current = records[stop - window : stop]
            points.append(
                {
                    "through_prompt": stop,
                    **{field: _mean([record[field] for record in current]) for field in fields},
                }
            )
        output[route] = {"window_prompts": window, "points": points, "last": points[-1]}
    return output


def direction_decision(segments, global_metrics, config):
    action_early, action_late = segments["action"]["early"], segments["action"]["late"]
    chain_early, chain_late = segments["chain"]["early"], segments["chain"]["late"]
    gate = config["direction_gate"]
    checks = {
        "action_f1": action_late["f1"] - action_early["f1"] >= gate["action_f1_late_minus_early_min"],
        "action_precision": action_late["precision"] - action_early["precision"] >= gate["action_precision_late_minus_early_min"],
        "action_wrong_selection": action_late["wrong_selection"] - action_early["wrong_selection"] <= gate["action_wrong_selection_late_minus_early_max"],
        "chain_reward": chain_late["reward"] - chain_early["reward"] >= gate["chain_reward_late_minus_early_min"],
        "chain_logic": chain_late["logic_alignment"] - chain_early["logic_alignment"] >= gate["chain_logic_late_minus_early_min"],
        "chain_date_mismatch": chain_late["date_mismatch"] - chain_early["date_mismatch"] <= gate["chain_date_mismatch_late_minus_early_max"],
        "finite": not global_metrics["nan_or_inf"],
        "base_frozen": global_metrics["base_delta"] == 0.0,
        "lora_updated": global_metrics["lora_delta_l2"] > 0.0,
        "safety_stop_not_triggered": not global_metrics["safety_stop_triggered"],
    }
    return {
        "decision": "HEALTHY_TO_300" if all(checks.values()) else "STOP_AND_ANALYZE",
        "checks": checks,
    }


def markdown_summary(summary):
    def route_table(route, fields):
        rows = []
        for segment in ("early", "middle", "late"):
            values = summary["segments"][route][segment]
            rows.append(
                "| " + segment + " | " + " | ".join(f"{values[field]:.6f}" for field in fields) + " |"
            )
        return "\n".join(rows)

    action_rows = route_table("action", ACTION_FIELDS)
    chain_rows = route_table("chain", CHAIN_FIELDS)
    global_metrics = summary["global"]
    return f"""# GR_USER_v1 150-Prompt Directional Pilot

Run: `{summary['run_id']}`

Decision: **{summary['decision']['decision']}**

## Runtime

- Prompts: {summary['actual']['prompts']}
- Optimizer steps: {summary['actual']['optimizer_steps']}
- Training wall: {summary['actual']['training_wall_seconds']:.2f} seconds
- Seconds per prompt: {summary['actual']['seconds_per_prompt']:.3f}
- Peak VRAM: {summary['actual']['gpu_peak_vram_mib']:.0f} MiB

## Action

| Segment | F1 | Precision | Recall | Wrong selection | Hallucination | Duplicate |
|---|---:|---:|---:|---:|---:|---:|
{action_rows}

## Chain

| Segment | Reward | Action Alignment | Logic Alignment | Date mismatch | Action mismatch |
|---|---:|---:|---:|---:|---:|
{chain_rows}

## Global

- Mean group reward std: {global_metrics['reward_std']:.6f}
- Zero-std ratio: {global_metrics['zero_std_ratio']:.4%}
- Grad norm mean/max: {global_metrics['grad_norm_mean']:.6f} / {global_metrics['grad_norm_max']:.6f}
- Clip fraction mean/max: {global_metrics['clip_fraction_mean']:.4%} / {global_metrics['clip_fraction_max']:.4%}
- Masked-token rate: {global_metrics['masked_token_rate']:.4%}
- Base delta: {global_metrics['base_delta']}
- LoRA delta L2: {global_metrics['lora_delta_l2']:.9f}
- NaN/Inf: {global_metrics['nan_or_inf']}

Ten-prompt rolling means are stored in the JSON summary. Historical Phase 3B
baseline is recorded for context only; it is not treated as an exact matched
comparison because the prompt set differs.

No 300-prompt continuation, full epoch, or external evaluation was run.
"""


def _validate_config(config):
    expected = {
        "seed": 20260820,
        "action_prompts": 75,
        "chain_prompts": 75,
        "G": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "learning_rate": 1e-6,
        "epsilon": 0.2,
        "beta": 0.0,
        "advantage_epsilon": 1e-4,
        "penalty_strategy": "sqrt",
        "lambda": 0.5,
        "forward_batch_size": 1,
        "runtime": "P0",
        "length_bucketing": False,
        "expected_optimizer_steps": 20,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Pilot config drift: {mismatches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--config", type=Path, default=Path("/data/GRPO_USER/config/pilot150_v1.json"))
    parser.add_argument("--run-root", type=Path, default=Path("/data/GRPO_USER/runs"))
    parser.add_argument("--result-output", type=Path, default=Path("/data/GRPO_USER/results/pilot150_v1_summary.json"))
    parser.add_argument("--docs-output", type=Path, default=Path("/data/GRPO_USER/docs/pilot150_v1.md"))
    args = parser.parse_args()

    total_started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4 or local_rank not in range(4):
        raise RuntimeError("Pilot150 requires exactly four GPU ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    run_dir = args.run_root / args.run_id
    monitor_dir = Path(os.environ.get("GRPO_MONITOR_DIR", "/data/GRPO/runs")) / args.run_id

    sha_before = dataset_sha(Path(config["dataset"]).parent)
    if sha_before != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen GR_USER_v1 data SHA mismatch")
    selected, selection_audit = select_pilot_rows(read_jsonl(config["dataset"]), config["seed"])
    plan = build_training_plan(selected)
    manifest = {
        "run_id": args.run_id,
        "run_kind": "user_grpo",
        "experiment": config["experiment"],
        "status": "running",
        "demo": False,
        "start_time": utc_now(),
        "git_commit": args.git_commit,
        "runner": "run_user_pilot150.py",
        "world_size": world,
        "dataset": config["dataset"],
        "dataset_sha": sha_before,
        "parent_checkpoint": config["adapter"],
        "base_model": config["base_model"],
        "selected_sample_ids": {
            route: [row["sample_id"] for row in selected[route]] for route in ROUTES
        },
        "selection_audit": selection_audit,
        "effective_max_steps": len(plan),
        "expected_unique_prompts": 150,
        "frozen_contract": config,
        "historical_phase3b_baseline": HISTORICAL_BASELINE,
        "output_dir": str(run_dir),
    }
    if is_main:
        if run_dir.exists() or monitor_dir.exists():
            raise FileExistsError("run-id already has training or monitor data")
        run_dir.mkdir(parents=True)
    dist.barrier()
    monitor_writer = monitor_from_env(args.run_id, rank)
    monitor = UserMonitorAdapter(monitor_writer)
    if is_main:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not monitor.write_manifest(**manifest):
            raise RuntimeError("failed to write User monitor manifest")
    dist.barrier()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"] + rank)
    torch.manual_seed(config["seed"] + rank)
    torch.cuda.manual_seed_all(config["seed"] + rank)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(
        base_model, config["adapter"], is_trainable=True, local_files_only=True
    )
    for module in peft_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in peft_model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    model = DistributedDataParallel(
        peft_model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name.lower() for name, _ in trainable):
        raise RuntimeError("only LoRA parameters may be trainable")
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=config["learning_rate"],
        weight_decay=0.0,
    )
    model.module.config.use_cache = False
    model.module.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if is_main:
        base_sha_before, base_tensor_count = parameter_sha256(model, include_lora=False)
        lora_sha_before, lora_tensor_count = parameter_sha256(model, include_lora=True)
        lora_before = lora_snapshot(model)
    else:
        base_sha_before = lora_sha_before = None
        base_tensor_count = lora_tensor_count = 0
        lora_before = None
    dist.barrier()

    route_positions = {
        route: {row["sample_id"]: index for index, row in enumerate(selected[route])}
        for route in ROUTES
    }
    dummy_cache = {}
    all_group_records = []
    all_step_records = []
    processed = 0
    consecutive_grad_explosion = 0
    consecutive_high_clip = 0
    safety_stop_reasons = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    for plan_item in plan:
        step = plan_item["step"]
        route = plan_item["route"]
        real_rows, dummy, loss_scale = distribute_step_rows(plan_item["rows"], rank, world)
        generation_started = time.perf_counter()
        if real_rows:
            random.seed(config["seed"] + step * 100 + rank)
            torch.manual_seed(config["seed"] + step * 100 + rank)
            torch.cuda.manual_seed_all(config["seed"] + step * 100 + rank)
            model.eval()
            completions = generate_route(model.module, tokenizer, real_rows, device)
            if route not in dummy_cache:
                dummy_cache[route] = (real_rows[0], completions[:G])
            batch_rows = real_rows
        else:
            if route not in dummy_cache:
                raise RuntimeError("tail-step dummy cache is unavailable")
            cached_row, cached_completions = dummy_cache[route]
            batch_rows = [cached_row]
            completions = cached_completions
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started

        scoring_started = time.perf_counter()
        rollout = prepare_scored_rollout(batch_rows, completions, tokenizer)
        batch = make_policy_batch(tokenizer, rollout, device)
        torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - scoring_started

        if real_rows:
            route_start = route_positions[route][real_rows[0]["sample_id"]]
            local_groups, local_traces = group_records_from_rollout(
                rollout, real_rows, route_start, step
            )
            objective_payload = {
                "per_kind_masked_token_count": rollout["metrics"]["per_kind_masked_token_count"],
                "per_kind_incremental_negative_mass": rollout["metrics"]["per_kind_incremental_negative_mass"],
            }
        else:
            local_groups, local_traces = [], []
            objective_payload = {
                "per_kind_masked_token_count": {},
                "per_kind_incremental_negative_mass": {},
            }

        model.eval()
        combined = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        with torch.no_grad():
            batch["old_per_token_logps"] = trainer._get_user_per_token_logps(
                model, combined, attention, batch["completion_ids"].size(1)
            ).detach()

        optimizer.zero_grad(set_to_none=True)
        model.train()
        policy_started = time.perf_counter()
        local_loss = trainer._compute_loss(model, batch) * loss_scale
        if not torch.isfinite(local_loss):
            raise RuntimeError("Pilot loss is NaN or Inf")
        local_loss.backward()
        torch.cuda.synchronize(device)
        policy_seconds = time.perf_counter() - policy_started
        raw_grad_norm = grad_norm([parameter for _, parameter in trainable])
        if not math.isfinite(raw_grad_norm) or raw_grad_norm <= 0.0:
            raise RuntimeError("Pilot LoRA gradient norm is invalid")
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)

        global_prompt_count = len(plan_item["rows"])
        global_loss = _all_reduce_scalar(float(local_loss.detach())) / world
        ratio_mean = _all_reduce_scalar(
            trainer.last_loss_metrics["ratio_mean"] * len(real_rows)
        ) / global_prompt_count
        clip_fraction = _all_reduce_scalar(
            trainer.last_loss_metrics["clip_fraction"] * len(real_rows)
        ) / global_prompt_count
        global_grad_norm = _all_reduce_scalar(raw_grad_norm, dist.ReduceOp.MAX)
        generation_seconds = _all_reduce_scalar(generation_seconds, dist.ReduceOp.MAX)
        scoring_seconds = _all_reduce_scalar(scoring_seconds, dist.ReduceOp.MAX)
        policy_seconds = _all_reduce_scalar(policy_seconds, dist.ReduceOp.MAX)

        gathered = _all_gather_object(
            {
                "groups": local_groups,
                "traces": local_traces,
                "objective": objective_payload,
            }
        )
        processed += global_prompt_count
        rank_event = {
            "step": step,
            "route": route,
            "real_prompt_count": len(real_rows),
            "dummy": dummy,
            "loss_scale": loss_scale,
            "generation_wall_sec": generation_seconds,
            "policy_wall_sec": policy_seconds,
        }
        monitor_writer.write_rank(rank_event)

        if is_main:
            groups = sorted(
                [record for payload in gathered for record in payload["groups"]],
                key=lambda record: record["route_index"],
            )
            traces = [record for payload in gathered for record in payload["traces"]]
            objectives = [payload["objective"] for payload in gathered]
            rollout_metrics = aggregate_step_metrics(groups, objectives)
            policy_metrics = {
                "loss": global_loss,
                "grad_norm": global_grad_norm,
                "learning_rate": config["learning_rate"],
                "ratio_mean": ratio_mean,
                "clip_fraction": clip_fraction,
                "policy_wall_sec": policy_seconds,
            }
            step_event = {
                "step": step,
                "route": route,
                "processed_unique_prompts": processed,
                "step_unique_prompts": global_prompt_count,
                "generation_wall_sec": generation_seconds,
                "reward_mask_wall_sec": scoring_seconds,
                **policy_metrics,
                **rollout_metrics,
            }
            all_group_records.extend(groups)
            all_step_records.append(step_event)
            for record in groups:
                _append_jsonl(run_dir / "groups.jsonl", record)
            _append_jsonl(run_dir / "steps.jsonl", step_event)
            monitor.write_step(
                step=step,
                route=route,
                rollout_metrics=rollout_metrics,
                policy_metrics=policy_metrics,
                processed_unique_prompts=processed,
                step_unique_prompts=global_prompt_count,
                generation_wall_sec=generation_seconds,
                reward_mask_wall_sec=scoring_seconds,
            )
            monitor.write_rollout(
                {
                    "rollout_id": step,
                    "step": step,
                    "route": route,
                    "g": G,
                    "group_ids": [record["sample_id"] for record in groups],
                    "reward_mean": rollout_metrics["task_reward_mean"],
                    "reward_std": rollout_metrics["task_reward_std"],
                    "zero_std_ratio": rollout_metrics["zero_std_ratio"],
                    "completion_length_mean": _mean([
                        record["completion_token_mean"] for record in groups
                    ]),
                    "masked_candidate_rate": rollout_metrics["masked_candidate_rate"],
                    "masked_token_rate": rollout_metrics["masked_token_rate"],
                }
            )
            selected_trace = min(traces, key=lambda record: record["route_index"])
            monitor.write_trace(
                {
                    "rollout_id": step,
                    "step": step,
                    "route": route,
                    "group_id": selected_trace["sample_id"],
                    "candidates": selected_trace["candidates"],
                }
            )

            safety = config["safety"]
            consecutive_grad_explosion = (
                consecutive_grad_explosion + 1
                if global_grad_norm > safety["grad_explosion_threshold"] else 0
            )
            consecutive_high_clip = (
                consecutive_high_clip + 1
                if clip_fraction > safety["clip_fraction_threshold"] else 0
            )
            if consecutive_grad_explosion >= safety["grad_explosion_consecutive_steps"]:
                safety_stop_reasons.append("consecutive_grad_explosion")
            if consecutive_high_clip >= safety["clip_fraction_consecutive_steps"]:
                safety_stop_reasons.append("sustained_high_clip_fraction")
            if not all(math.isfinite(value) for value in (
                global_loss,
                global_grad_norm,
                ratio_mean,
                clip_fraction,
                rollout_metrics["task_reward_std"],
                rollout_metrics["masked_token_rate"],
            )):
                safety_stop_reasons.append("nan_or_inf_metric")
            if monitor_writer.errors:
                safety_stop_reasons.append("monitor_write_error")
            print(
                json.dumps(
                    {
                        "step": step,
                        "route": route,
                        "processed": processed,
                        "loss": global_loss,
                        "reward": rollout_metrics["task_reward_mean"],
                        "grad_norm": global_grad_norm,
                        "clip_fraction": clip_fraction,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        stop = torch.tensor(int(bool(safety_stop_reasons) if is_main else 0), device=device)
        dist.broadcast(stop, 0)
        if bool(stop.item()):
            if is_main:
                (run_dir / "STOPPED.json").write_text(
                    json.dumps({"step": step, "reasons": safety_stop_reasons}, indent=2),
                    encoding="utf-8",
                )
            raise RuntimeError(f"Pilot safety stop: {safety_stop_reasons}")

    dist.barrier()
    training_wall = time.perf_counter() - training_started
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
        raise RuntimeError("frozen data changed during Pilot150")

    checkpoint_dir = run_dir / "pilot150-final"
    if is_main:
        if base_sha_before != base_sha_after:
            raise RuntimeError("base model changed during Pilot150")
        if lora_sha_before == lora_sha_after or lora_delta_l2 <= 0.0:
            raise RuntimeError("LoRA did not update during Pilot150")
        checkpoint_dir.mkdir()
        model.module.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        checkpoint_metadata = {
            "run_id": args.run_id,
            "processed_unique_prompts": processed,
            "optimizer_steps": len(plan),
            "git_commit": args.git_commit,
            "parent_checkpoint": config["adapter"],
            "dataset_sha": sha_after,
            "training_config": config,
            "created_at": utc_now(),
        }
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(checkpoint_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        segments = segment_summaries(all_group_records)
        rolling = rolling_summaries(all_group_records)
        reward_stds = [record["reward_std"] for record in all_group_records]
        total_masked = sum(record["masked_token_count"] for record in all_group_records)
        total_valid = sum(record["valid_token_count"] for record in all_group_records)
        global_metrics = {
            "reward_std": _mean(reward_stds),
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
            "safety_stop_triggered": False,
        }
        decision = direction_decision(segments, global_metrics, config)
        summary = {
            "contract_version": "gr_user_pilot150_v1",
            "run_id": args.run_id,
            "git_commit": args.git_commit,
            "checkpoint_path": str(checkpoint_dir),
            "historical_phase3b_baseline": HISTORICAL_BASELINE,
            "actual": {
                "prompts": processed,
                "action_prompts": 75,
                "chain_prompts": 75,
                "optimizer_steps": len(plan),
                "training_wall_seconds": training_wall,
                "total_run_wall_seconds": time.perf_counter() - total_started,
                "seconds_per_prompt": training_wall / processed,
                "gpu_peak_vram_mib": max(peaks),
                "per_gpu_peak_vram_mib": peaks,
            },
            "segments": segments,
            "rolling_means": rolling,
            "global": global_metrics,
            "decision": decision,
            "integrity": {
                "data_sha_before": sha_before,
                "data_sha_after": sha_after,
                "frozen_data_unchanged": sha_before == sha_after == EXPECTED_DATA_SHA,
                "base_frozen": base_sha_before == base_sha_after,
                "lora_updated": lora_sha_before != lora_sha_after and lora_delta_l2 > 0,
                "nan_or_inf": False,
                "formal_pilot150_completed": True,
                "continued_to_300": False,
                "full_epoch_run": False,
                "external_evaluation_run": False,
            },
            "selection_audit": selection_audit,
            "selected_sample_ids": manifest["selected_sample_ids"],
        }
        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        args.docs_output.parent.mkdir(parents=True, exist_ok=True)
        args.result_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        args.docs_output.write_text(markdown_summary(summary), encoding="utf-8")
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        completed_manifest = {
            **manifest,
            "status": "completed",
            "completed_at": utc_now(),
            "actual_unique_prompts": processed,
            "actual_optimizer_steps": len(plan),
            "checkpoint_path": str(checkpoint_dir),
            "decision": decision["decision"],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(completed_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        monitor.write_manifest(**completed_manifest)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "run_id": args.run_id,
                    "checkpoint": str(checkpoint_dir),
                    "decision": decision["decision"],
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
