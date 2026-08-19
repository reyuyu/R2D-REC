"""Generate and audit GR_USER_v1 G=4 rollouts without training."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_SHA = {
    "train_3000.jsonl": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot_600.jsonl": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe_v1.jsonl": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}
ACTION_ALLOCATION = {"1-5": 8, "6-10": 10, "11-20": 15, "21-30": 10, "31-40": 5, "41+": 2}
CHAIN_ALLOCATION = {"2": 8, "3": 27, "4": 12, "5": 3}
G = 4
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 20260819
MAX_NEW_TOKENS = 512


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def numeric_summary(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def evenly_spaced(rows, count):
    ordered = sorted(rows, key=lambda row: (row["prompt_token_count"], row["sample_id"]))
    if count > len(ordered):
        raise ValueError(f"requested {count} rows from a stratum of {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("evenly-spaced selection produced duplicate indices")
    return [ordered[index] for index in indices]


def select_formal(rows):
    selected = []
    for route, allocation in (("action", ACTION_ALLOCATION), ("chain", CHAIN_ALLOCATION)):
        route_rows = [row for row in rows if row["route"] == route]
        for bucket, count in allocation.items():
            stratum = [row for row in route_rows if str(row["bucket"]) == bucket]
            chosen = evenly_spaced(stratum, count)
            for row in chosen:
                row = dict(row)
                row["audit_stratum"] = bucket
                selected.append(row)
    if collections.Counter(row["route"] for row in selected) != {"action": 50, "chain": 50}:
        raise AssertionError("formal selection is not 50 Action + 50 Chain")
    if len({row["sample_id"] for row in selected}) != 100:
        raise AssertionError("formal selection contains duplicate prompts")
    return selected


def select_sanity(rows):
    selected = []
    for route in ("action", "chain"):
        route_rows = sorted(
            [row for row in rows if row["route"] == route],
            key=lambda row: (row["prompt_token_count"], row["sample_id"]),
        )
        for label, row in (("short", route_rows[0]), ("long", route_rows[-1])):
            item = dict(row)
            item["audit_stratum"] = label
            selected.append(item)
    return selected


def gpu_preflight(physical_gpu_ids):
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().splitlines()
    states = {}
    for row in gpu_rows:
        index, uuid, used, free, utilization = [part.strip() for part in row.split(",")]
        states[index] = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    compute_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    compute = []
    for row in compute_output.splitlines() if compute_output else []:
        uuid, pid, process_name, used = [part.strip() for part in row.split(",", 3)]
        compute.append({"uuid": uuid, "pid": int(pid), "process_name": process_name, "used_memory_mib": int(used)})
    selected = []
    for gpu_id in physical_gpu_ids:
        if str(gpu_id) not in states:
            raise RuntimeError(f"GPU {gpu_id} does not exist")
        state = states[str(gpu_id)]
        processes = [item for item in compute if item["uuid"] == state["uuid"]]
        if processes or state["memory_used_mib"] > 1024:
            raise RuntimeError(f"GPU {gpu_id} is not idle: state={state}, compute={processes}")
        selected.append({**state, "compute_processes": processes})
    return selected


def lora_checksums(model, limit=16):
    import torch

    tensors = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name.lower()]
    tensors.sort(key=lambda item: item[0])
    if len(tensors) < 10:
        raise RuntimeError(f"found only {len(tensors)} LoRA tensors")
    checksums = {}
    for name, parameter in tensors[:limit]:
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        checksums[name] = hashlib.sha256(raw).hexdigest()
    return checksums


def render_prompts(tokenizer, rows):
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count != row["prompt_token_count"]:
            raise AssertionError(
                f"prompt rendering drift for {row['sample_id']}: {token_count} != {row['prompt_token_count']}"
            )
        rendered.append(text)
    return rendered


def trim_generated_ids(token_ids, stop_ids):
    ids = list(token_ids)
    for index, token_id in enumerate(ids):
        if token_id in stop_ids:
            return ids[:index], True, token_id
    return ids, False, None


def candidate_status(grounding_status):
    statuses = [item["status"] for item in grounding_status]
    if statuses and all(status == "grounded" for status in statuses):
        return "grounded"
    if not statuses or all(status == "ungrounded" for status in statuses):
        return "ungrounded"
    return "partially_grounded"


def compact_candidate(candidate):
    return {
        "sample_id": candidate["sample_id"],
        "route": candidate["route"],
        "bucket": candidate["bucket"],
        "candidate_index": candidate["candidate_index"],
        "reward": candidate["reward"],
        "format_valid": candidate["format_valid"],
        "violation_kinds": candidate["violation_kinds"],
        "candidate_grounding": candidate.get("candidate_grounding"),
        "completion_token_count": candidate["completion_token_count"],
        "completion_excerpt": candidate["completion"][:600],
    }


def cooccurrence(candidates):
    counts = collections.Counter()
    for candidate in candidates:
        kinds = sorted(set(candidate["violation_kinds"]))
        for left_index, left in enumerate(kinds):
            for right in kinds[left_index + 1 :]:
                counts[f"{left}+{right}"] += 1
    return dict(counts.most_common())


def group_variance(groups, route):
    selected = [group for group in groups if group["route"] == route]
    stds = [group["reward_population_std"] for group in selected]
    zero = [group for group in selected if group["zero_std"]]
    bucket_output = {}
    for bucket in sorted({str(group["bucket"]) for group in selected}):
        bucket_groups = [group for group in selected if str(group["bucket"]) == bucket]
        bucket_stds = [group["reward_population_std"] for group in bucket_groups]
        bucket_output[bucket] = {
            "prompt_count": len(bucket_groups),
            "zero_std_count": sum(group["zero_std"] for group in bucket_groups),
            "zero_std_rate": sum(group["zero_std"] for group in bucket_groups) / len(bucket_groups),
            "mean_group_std": statistics.fmean(bucket_stds),
            "median_group_std": statistics.median(bucket_stds),
        }
    return {
        "prompt_count": len(selected),
        "zero_std_count": len(zero),
        "zero_std_rate": len(zero) / len(selected),
        "mean_group_std": statistics.fmean(stds),
        "median_group_std": statistics.median(stds),
        "by_bucket": bucket_output,
    }


def action_metrics(candidates):
    selected = [item for item in candidates if item["route"] == "action"]
    rate = lambda predicate: sum(predicate(item) for item in selected) / len(selected)
    return {
        "candidate_count": len(selected),
        "f1": numeric_summary([item["reward"] for item in selected]),
        "precision_mean": statistics.fmean(item["score"]["precision"] for item in selected),
        "recall_mean": statistics.fmean(item["score"]["recall"] for item in selected),
        "json_valid_rate": rate(lambda item: item["format_valid"]),
        "exact_set_rate": rate(lambda item: item["score"]["exact_set_match"]),
        "hallucination_candidate_rate": rate(lambda item: "hallucinated_sid" in item["violation_kinds"]),
        "duplicate_candidate_rate": rate(lambda item: "duplicate_sid" in item["violation_kinds"]),
        "wrong_selection_candidate_rate": rate(lambda item: "wrong_selection_sid" in item["violation_kinds"]),
        "completion_tokens": numeric_summary([item["completion_token_count"] for item in selected]),
        "max_new_tokens_hit_rate": rate(lambda item: item["hit_max_new_tokens"]),
        "penalty_mask_fraction": numeric_summary([item["penalty_mask_fraction"] for item in selected]),
        "violation_counts": dict(collections.Counter(kind for item in selected for kind in item["violation_kinds"])),
        "violation_cooccurrence": cooccurrence(selected),
    }


def chain_metrics(candidates):
    selected = [item for item in candidates if item["route"] == "chain"]
    rate = lambda predicate: sum(predicate(item) for item in selected) / len(selected)
    event_counts = collections.Counter(
        status["status"] for item in selected for status in item["score"]["grounding_status"]
    )
    event_total = sum(event_counts.values())
    candidate_counts = collections.Counter(item["candidate_grounding"] for item in selected)
    constraint_kinds = (
        "hallucinated_sid",
        "date_mismatch",
        "action_mismatch",
        "duplicate_event",
        "chronology_violation",
        "excess_event",
    )
    return {
        "candidate_count": len(selected),
        "total_reward": numeric_summary([item["reward"] for item in selected]),
        "action_alignment_f1_mean": statistics.fmean(item["score"]["action_f1"] for item in selected),
        "logic_alignment_f1_mean": statistics.fmean(item["score"]["logic_f1"] for item in selected),
        "json_valid_rate": rate(lambda item: item["format_valid"]),
        "event_grounding": {
            status: {"count": event_counts[status], "rate": event_counts[status] / event_total if event_total else 0.0}
            for status in ("grounded", "partially_grounded", "ungrounded")
        },
        "candidate_grounding": {
            status: {"count": candidate_counts[status], "rate": candidate_counts[status] / len(selected)}
            for status in ("grounded", "partially_grounded", "ungrounded")
        },
        "constraint_candidate_rates": {
            kind: rate(lambda item, target=kind: target in item["violation_kinds"]) for kind in constraint_kinds
        },
        "constraint_violation_counts": {
            kind: sum(item["violation_kinds"].count(kind) for item in selected) for kind in constraint_kinds
        },
        "completion_tokens": numeric_summary([item["completion_token_count"] for item in selected]),
        "max_new_tokens_hit_rate": rate(lambda item: item["hit_max_new_tokens"]),
        "penalty_mask_fraction": numeric_summary([item["penalty_mask_fraction"] for item in selected]),
        "violation_cooccurrence": cooccurrence(selected),
    }


def representative_examples(candidates):
    action = [item for item in candidates if item["route"] == "action"]
    chain = [item for item in candidates if item["route"] == "chain"]
    highest = lambda rows, predicate: [compact_candidate(item) for item in sorted(
        [row for row in rows if predicate(row)], key=lambda row: (-row["reward"], row["sample_id"], row["candidate_index"])
    )[:5]]
    lowest = lambda rows, predicate: [compact_candidate(item) for item in sorted(
        [row for row in rows if predicate(row)], key=lambda row: (row["reward"], row["sample_id"], row["candidate_index"])
    )[:5]]
    return {
        "action_high_reward_hallucination": highest(action, lambda item: "hallucinated_sid" in item["violation_kinds"]),
        "action_high_reward_duplicate": highest(action, lambda item: "duplicate_sid" in item["violation_kinds"]),
        "chain_high_reward_date_or_action_mismatch": highest(
            chain,
            lambda item: bool({"date_mismatch", "action_mismatch"} & set(item["violation_kinds"])),
        ),
        "action_low_reward_valid_grounded_no_duplicate": lowest(
            action,
            lambda item: item["format_valid"]
            and "hallucinated_sid" not in item["violation_kinds"]
            and "duplicate_sid" not in item["violation_kinds"],
        ),
        "chain_low_reward_valid_grounded_no_duplicate": lowest(
            chain,
            lambda item: item["format_valid"]
            and item["candidate_grounding"] == "grounded"
            and "duplicate_event" not in item["violation_kinds"],
        ),
    }


def markdown_summary(summary):
    action = summary["action"]
    chain = summary["chain"]
    variance = summary["group_variance"]
    lines = [f"""# GR_USER_v1 G=4 rollout-only audit

Run: `{summary['run_id']}`

## Configuration

- Parent: Beta baseline Epoch 2 checkpoint-1106
- GPU IDs: {summary['gpu']['physical_ids']}
- G: {G}
- Sampling: temperature={TEMPERATURE}, top_p={TOP_P}, seed={SEED}
- max_new_tokens: {MAX_NEW_TOKENS}
- Training: disabled

## Group variance

| Route | Zero std | Mean group std | Median group std |
|---|---:|---:|---:|
| Action | {variance['action']['zero_std_rate']:.2%} | {variance['action']['mean_group_std']:.6f} | {variance['action']['median_group_std']:.6f} |
| Chain | {variance['chain']['zero_std_rate']:.2%} | {variance['chain']['mean_group_std']:.6f} | {variance['chain']['median_group_std']:.6f} |

Assessment: **{summary['g4_assessment']}**

### Per-bucket variance

| Route | Bucket | Prompts | Zero std | Mean std | Median std |
|---|---|---:|---:|---:|---:|"""]
    for route in ("action", "chain"):
        for bucket, values in variance[route]["by_bucket"].items():
            lines.append(
                f"| {route.title()} | {bucket} | {values['prompt_count']} | "
                f"{values['zero_std_rate']:.2%} | {values['mean_group_std']:.6f} | "
                f"{values['median_group_std']:.6f} |"
            )
    lines.append(f"""

## Candidate metrics

- Action F1 mean/p50/p90: {action['f1']['mean']:.6f} / {action['f1']['p50']:.6f} / {action['f1']['p90']:.6f}
- Action precision/recall: {action['precision_mean']:.6f} / {action['recall_mean']:.6f}; JSON valid: {action['json_valid_rate']:.2%}; exact set: {action['exact_set_rate']:.2%}
- Action hallucination/duplicate/wrong selection: {action['hallucination_candidate_rate']:.2%} / {action['duplicate_candidate_rate']:.2%} / {action['wrong_selection_candidate_rate']:.2%}
- Chain reward mean/p50/p90: {chain['total_reward']['mean']:.6f} / {chain['total_reward']['p50']:.6f} / {chain['total_reward']['p90']:.6f}
- Chain Action/Logic Alignment F1: {chain['action_alignment_f1_mean']:.6f} / {chain['logic_alignment_f1_mean']:.6f}; JSON valid: {chain['json_valid_rate']:.2%}
- Chain candidate grounding (grounded/partial/ungrounded): {chain['candidate_grounding']['grounded']['rate']:.2%} / {chain['candidate_grounding']['partially_grounded']['rate']:.2%} / {chain['candidate_grounding']['ungrounded']['rate']:.2%}
- Action penalty-mask fraction mean: {action['penalty_mask_fraction']['mean']:.4%}
- Chain penalty-mask fraction mean: {chain['penalty_mask_fraction']['mean']:.4%}
- Action truncation: {action['max_new_tokens_hit_rate']:.2%}; Chain truncation: {chain['max_new_tokens_hit_rate']:.2%}

### Chain constraint candidate rates

| Constraint | Rate | Violation records |
|---|---:|---:|""")
    for kind, rate in chain["constraint_candidate_rates"].items():
        lines.append(f"| `{kind}` | {rate:.2%} | {chain['constraint_violation_counts'][kind]} |")
    lines.append("\n### Violation cooccurrence\n")
    lines.append(f"- Action: `{json.dumps(action['violation_cooccurrence'], sort_keys=True)}`")
    lines.append(f"- Chain: `{json.dumps(chain['violation_cooccurrence'], sort_keys=True)}`")
    lines.append("\n## Representative examples\n")
    for label, examples in summary["representative_examples"].items():
        lines.append(f"### {label.replace('_', ' ').title()}")
        if not examples:
            lines.append("\nNone found.\n")
            continue
        lines.append("")
        for example in examples:
            excerpt = example["completion_excerpt"].replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"- `{example['sample_id']}` candidate {example['candidate_index']}, "
                f"reward={example['reward']:.6f}, violations={example['violation_kinds']}: {excerpt}"
            )
        lines.append("")
    lines.append(f"""## Integrity

- LoRA checksum unchanged: {summary['integrity']['lora_checksum_unchanged']}
- Frozen data SHA unchanged: {summary['integrity']['frozen_sha_unchanged']}
- `requires_grad` parameters after load: {summary['integrity']['requires_grad_parameter_count']}
""")
    return "\n".join(lines)


def aggregate_shards(args):
    shard_dirs = [Path(value) for value in args.aggregate_run_dirs]
    if len(shard_dirs) != args.shard_count:
        raise SystemExit(f"expected {args.shard_count} shard directories, got {len(shard_dirs)}")
    shard_summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in shard_dirs]
    candidates = [item for path in shard_dirs for item in read_jsonl(path / "candidates.jsonl")]
    groups = [item for path in shard_dirs for item in read_jsonl(path / "groups.jsonl")]
    selected = [
        item
        for path in shard_dirs
        for item in json.loads((path / "selected_samples.json").read_text(encoding="utf-8"))
    ]
    candidate_keys = {(item["sample_id"], item["candidate_index"]) for item in candidates}
    if len(selected) != 100 or len({item["sample_id"] for item in selected}) != 100:
        raise RuntimeError("aggregate does not contain 100 unique prompts")
    if len(candidates) != 400 or len(candidate_keys) != 400 or len(groups) != 100:
        raise RuntimeError("aggregate does not contain 400 unique candidates and 100 groups")
    if collections.Counter(item["route"] for item in selected) != {"action": 50, "chain": 50}:
        raise RuntimeError("aggregate route balance differs from 50 Action + 50 Chain")
    if not all(summary["integrity"]["lora_checksum_unchanged"] for summary in shard_summaries):
        raise RuntimeError("one or more shard LoRA checksums changed")
    if not all(summary["integrity"]["frozen_sha_unchanged"] for summary in shard_summaries):
        raise RuntimeError("one or more shard frozen-data checks failed")

    action_variance = group_variance(groups, "action")
    chain_variance = group_variance(groups, "chain")
    run_id = args.run_id or f"GR-USER-G4-AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    integrity = {
        "worker_count": len(shard_summaries),
        "lora_tensor_count_checked_per_worker": [
            summary["integrity"]["lora_tensor_count_checked"] for summary in shard_summaries
        ],
        "lora_checksum_unchanged": True,
        "requires_grad_parameter_count": sum(
            summary["integrity"]["requires_grad_parameter_count"] for summary in shard_summaries
        ),
        "frozen_sha_before": EXPECTED_SHA,
        "frozen_sha_after": EXPECTED_SHA,
        "frozen_sha_unchanged": True,
        "sid_special_token_decode_preserved": all(
            summary["integrity"]["sid_special_token_decode_preserved"] for summary in shard_summaries
        ),
        "workers": [summary["integrity"] for summary in shard_summaries],
    }
    summary = {
        "contract_version": "gr_user_rollout_audit_g4_v1",
        "run_id": run_id,
        "mode": "audit",
        "run_dir": str(run_dir),
        "shard_run_dirs": [str(path) for path in shard_dirs],
        "gpu": {
            "physical_ids": sorted(
                gpu_id for summary in shard_summaries for gpu_id in summary["gpu"]["physical_ids"]
            ),
            "preflight": [state for summary in shard_summaries for state in summary["gpu"]["preflight"]],
        },
        "generation": {
            **shard_summaries[0]["generation"],
            "prompt_count": len(selected),
            "completion_count": len(candidates),
            "worker_count": len(shard_summaries),
        },
        "selection": {
            "counts": dict(collections.Counter(item["route"] for item in selected)),
            "strata": dict(
                collections.Counter(f"{item['route']}:{item['audit_stratum']}" for item in selected)
            ),
            "prompt_tokens": {
                route: numeric_summary([item["prompt_token_count"] for item in selected if item["route"] == route])
                for route in ("action", "chain")
            },
        },
        "timing": {"workers": [summary["timing"] for summary in shard_summaries]},
        "integrity": integrity,
        "action": action_metrics(candidates),
        "chain": chain_metrics(candidates),
        "group_variance": {"action": action_variance, "chain": chain_variance},
        "g4_assessment": (
            "G4 healthy"
            if action_variance["zero_std_rate"] < 0.05 and chain_variance["zero_std_rate"] < 0.05
            else "G4 potentially insufficient"
        ),
        "representative_examples": representative_examples(candidates),
    }
    with (run_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (run_dir / "groups.jsonl").open("w", encoding="utf-8") as handle:
        for item in groups:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (run_dir / "selected_samples.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "rollout_audit_g4_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.docs_dir / "rollout_audit_g4_v1.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "assessment": summary["g4_assessment"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sanity", "audit", "audit-shard", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/data/GRPO_USER/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("/data/GRPO_USER/results"))
    parser.add_argument("--docs-dir", type=Path, default=Path("/data/GRPO_USER/docs"))
    parser.add_argument("--physical-gpu-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--aggregate-run-dirs", nargs="*", default=[])
    args = parser.parse_args()

    if args.mode == "aggregate":
        aggregate_shards(args)
        return

    physical_gpu_ids = [int(value) for value in args.physical_gpu_ids.split(",") if value.strip()]
    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if visible != [str(value) for value in physical_gpu_ids]:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={visible} does not match --physical-gpu-ids={physical_gpu_ids}")
    gpu_state = gpu_preflight(physical_gpu_ids)
    if len(physical_gpu_ids) != 1:
        raise SystemExit("v1 rollout runner intentionally uses exactly one free GPU")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in os.sys.path:
        os.sys.path.insert(0, str(scripts_dir))
    from user_action_reward import score_action
    from user_chain_reward import score_chain
    from user_penalty_mask import compile_penalty_mask
    from user_generated_token_projection import project_compiled_mask_to_generated

    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    random.seed(SEED)

    sha_before = {name: sha256_file(args.data_dir / name) for name in EXPECTED_SHA}
    if sha_before != EXPECTED_SHA:
        raise SystemExit(f"frozen data SHA mismatch: {sha_before}")
    rows = read_jsonl(args.data_dir / "pilot_600.jsonl")
    selected = select_sanity(rows) if args.mode == "sanity" else select_formal(rows)
    if args.mode == "audit-shard":
        if not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("invalid shard index/count")
        selected = selected[args.shard_index :: args.shard_count]
    run_id = args.run_id or f"GR-USER-G4-{args.mode.upper()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    if None in stop_ids:
        stop_ids.remove(None)
    if tokenizer.eos_token_id != tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise RuntimeError("standard EOS and <|im_end|> differ")
    sid_probe = next(sid for row in rows for sid in row["history_sids"])
    if tokenizer.decode(tokenizer.encode(sid_probe, add_special_tokens=False), skip_special_tokens=False) != sid_probe:
        raise RuntimeError("SID-preserving decode audit failed")
    rendered_prompts = render_prompts(tokenizer, selected)

    configuration = {
        "run_id": run_id,
        "mode": args.mode,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "data": str(args.data_dir / "pilot_600.jsonl"),
        "physical_gpu_ids": physical_gpu_ids,
        "gpu_preflight": gpu_state,
        "G": G,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": args.batch_size,
        "stop_token_ids": sorted(stop_ids),
        "decode_skip_special_tokens": False,
        "prompt_count": len(selected),
        "expected_completion_count": len(selected) * G,
        "selection": collections.Counter(f"{row['route']}:{row['audit_stratum']}" for row in selected),
    }
    (run_dir / "config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "selected_samples.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": row["sample_id"],
                    "route": row["route"],
                    "bucket": row["bucket"],
                    "audit_stratum": row["audit_stratum"],
                    "prompt_token_count": row["prompt_token_count"],
                }
                for row in selected
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    load_started = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    requires_grad_count = sum(parameter.requires_grad for parameter in model.parameters())
    if requires_grad_count:
        raise RuntimeError(f"{requires_grad_count} parameters still require gradients")
    checksum_before = lora_checksums(model)
    load_seconds = time.perf_counter() - load_started

    candidates = []
    generated_started = time.perf_counter()
    candidate_path = run_dir / "candidates.jsonl"
    processing_order = sorted(range(len(selected)), key=lambda index: selected[index]["prompt_token_count"])
    with candidate_path.open("w", encoding="utf-8") as candidate_handle, torch.inference_mode():
        for offset in range(0, len(processing_order), args.batch_size):
            batch_indices = processing_order[offset : offset + args.batch_size]
            batch_prompts = [rendered_prompts[index] for index in batch_indices]
            encoded = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=G,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=sorted(stop_ids),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            generated = output_ids[:, encoded["input_ids"].shape[1] :].detach().cpu().tolist()
            for local_index, selected_index in enumerate(batch_indices):
                row = selected[selected_index]
                for candidate_index in range(G):
                    raw_ids = generated[local_index * G + candidate_index]
                    completion_ids, stopped, stop_token_id = trim_generated_ids(raw_ids, stop_ids)
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
                    scorer = score_action if row["route"] == "action" else score_chain
                    score = scorer(completion, row, tokenizer)
                    compiled = compile_penalty_mask(completion, score.violations, tokenizer, row["route"])
                    compiled = project_compiled_mask_to_generated(compiled, completion_ids)
                    if compiled["token_count"] != len(completion_ids):
                        raise RuntimeError("projected compiler token count differs from generation")
                    violation_kinds = [item.kind for item in score.violations]
                    candidate = {
                        "run_id": run_id,
                        "sample_id": row["sample_id"],
                        "route": row["route"],
                        "bucket": row["bucket"],
                        "audit_stratum": row["audit_stratum"],
                        "prompt_token_count": row["prompt_token_count"],
                        "candidate_index": candidate_index,
                        "completion": completion,
                        "completion_ids": completion_ids,
                        "completion_token_count": len(completion_ids),
                        "stopped_on_eos": stopped,
                        "stop_token_id": stop_token_id,
                        "hit_max_new_tokens": not stopped and len(completion_ids) >= MAX_NEW_TOKENS,
                        "reward": score.reward,
                        "format_valid": score.format_valid,
                        "violation_kinds": violation_kinds,
                        "score": score.to_dict(),
                        "penalty": compiled,
                        "penalty_mask_fraction": compiled["masked_token_fraction"],
                    }
                    if row["route"] == "chain":
                        candidate["candidate_grounding"] = candidate_status(score.grounding_status)
                    candidates.append(candidate)
                    candidate_handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    candidate_handle.flush()
    generation_seconds = time.perf_counter() - generated_started

    checksum_after = lora_checksums(model)
    checksum_unchanged = checksum_before == checksum_after
    if not checksum_unchanged:
        raise RuntimeError("LoRA checksum changed during rollout-only audit")
    del model, base_model
    torch.cuda.empty_cache()

    expected_count = len(selected) * G
    if len(candidates) != expected_count:
        raise RuntimeError(f"generated {len(candidates)} candidates, expected {expected_count}")
    if any(not item["completion"] for item in candidates):
        raise RuntimeError("empty completion found")
    if any(not math.isfinite(item["reward"]) for item in candidates):
        raise RuntimeError("non-finite reward found")

    by_sample = collections.defaultdict(list)
    for candidate in candidates:
        by_sample[candidate["sample_id"]].append(candidate)
    groups = []
    with (run_dir / "groups.jsonl").open("w", encoding="utf-8") as group_handle:
        for row in selected:
            group_candidates = sorted(by_sample[row["sample_id"]], key=lambda item: item["candidate_index"])
            rewards = [item["reward"] for item in group_candidates]
            std = statistics.pstdev(rewards)
            group = {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "route": row["route"],
                "bucket": row["bucket"],
                "audit_stratum": row["audit_stratum"],
                "prompt_token_count": row["prompt_token_count"],
                "rewards": rewards,
                "reward_mean": statistics.fmean(rewards),
                "reward_population_std": std,
                "zero_std": std <= 1e-12,
                "candidate_indices": [item["candidate_index"] for item in group_candidates],
            }
            groups.append(group)
            group_handle.write(json.dumps(group, ensure_ascii=False) + "\n")

    sha_after = {name: sha256_file(args.data_dir / name) for name in EXPECTED_SHA}
    integrity = {
        "lora_tensor_count_checked": len(checksum_before),
        "lora_checksums_before": checksum_before,
        "lora_checksums_after": checksum_after,
        "lora_checksum_unchanged": checksum_unchanged,
        "requires_grad_parameter_count": requires_grad_count,
        "frozen_sha_before": sha_before,
        "frozen_sha_after": sha_after,
        "frozen_sha_unchanged": sha_before == sha_after == EXPECTED_SHA,
        "sid_special_token_decode_preserved": True,
    }
    if not integrity["frozen_sha_unchanged"]:
        raise RuntimeError("frozen data changed during rollout audit")

    action_variance = group_variance(groups, "action")
    chain_variance = group_variance(groups, "chain")
    summary = {
        "contract_version": "gr_user_rollout_audit_g4_v1",
        "run_id": run_id,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "gpu": {"physical_ids": physical_gpu_ids, "preflight": gpu_state},
        "generation": {
            "G": G,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_token_ids": sorted(stop_ids),
            "skip_special_tokens": False,
            "prompt_count": len(selected),
            "completion_count": len(candidates),
            "batch_size": args.batch_size,
        },
        "selection": {
            "counts": dict(collections.Counter(row["route"] for row in selected)),
            "strata": dict(collections.Counter(f"{row['route']}:{row['audit_stratum']}" for row in selected)),
            "prompt_tokens": {
                route: numeric_summary([row["prompt_token_count"] for row in selected if row["route"] == route])
                for route in ("action", "chain")
            },
        },
        "timing": {"model_load_seconds": load_seconds, "generation_and_scoring_seconds": generation_seconds},
        "integrity": integrity,
        "action": action_metrics(candidates),
        "chain": chain_metrics(candidates),
        "group_variance": {
            "action": action_variance,
            "chain": chain_variance,
        },
        "g4_assessment": (
            "G4 healthy"
            if action_variance["zero_std_rate"] < 0.05 and chain_variance["zero_std_rate"] < 0.05
            else "G4 potentially insufficient"
        ),
        "representative_examples": representative_examples(candidates),
    }
    if args.mode == "sanity":
        summary["sanity"] = {
            "passed": len(candidates) == 16
            and integrity["lora_checksum_unchanged"]
            and integrity["frozen_sha_unchanged"]
            and all(len(item["penalty"]["penalty_mask"]) == item["completion_token_count"] for item in candidates),
            "parser_executed_count": len(candidates),
            "nonempty_completion_count": sum(bool(item["completion"]) for item in candidates),
        }
        if not summary["sanity"]["passed"]:
            raise RuntimeError("sanity validation failed")

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    if args.mode == "audit":
        args.results_dir.mkdir(parents=True, exist_ok=True)
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / "rollout_audit_g4_v1_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.docs_dir / "rollout_audit_g4_v1.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
