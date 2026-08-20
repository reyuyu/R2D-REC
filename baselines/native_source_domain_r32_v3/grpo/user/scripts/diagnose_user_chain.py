#!/usr/bin/env python3
"""Matched, inference-only diagnosis of GR_USER_v1 Chain probe changes."""

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
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260820
G = 4
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 512
UNCHANGED_EPSILON = 0.01
EXPECTED_FROZEN_SHA = {
    "train_3000.jsonl": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot_600.jsonl": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe_v1.jsonl": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}
CHECKPOINT_LABELS = ("C0", "C20", "C40")
EVALUATION_LABELS = (*CHECKPOINT_LABELS, "Final")
METRICS = ("total_reward", "action_alignment", "logic_alignment")
CHAIN_KINDS = (
    "hallucinated_sid",
    "date_mismatch",
    "action_mismatch",
    "duplicate_event",
    "chronology_violation",
    "excess_event",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{value}".encode()).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def numeric_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate an integer total proportionally with deterministic tie-breaking."""
    population = sum(counts.values())
    if population <= 0 or total < 0:
        raise ValueError("invalid proportional allocation")
    raw = {key: total * value / population for key, value in counts.items()}
    output = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(output.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - output[key]), key))
    for key in order[:remaining]:
        output[key] += 1
    return output


def quartile_boundaries(values: list[int]) -> list[float]:
    if not values:
        raise ValueError("cannot compute quartiles from an empty population")
    return [float(percentile([float(value) for value in values], q)) for q in (0.25, 0.50, 0.75)]


def quartile_for(value: int, boundaries: list[float]) -> str:
    return f"Q{1 + sum(value > boundary for boundary in boundaries)}"


def _flatten_ids(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for nested in value for item in _flatten_ids(nested)]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _flatten_ids(nested)]
    return []


def frozen_hashes(data_dir: Path) -> dict[str, str]:
    return {name: sha256_file(data_dir / name) for name in EXPECTED_FROZEN_SHA}


def validate_frozen_data(data_dir: Path) -> dict[str, str]:
    actual = frozen_hashes(data_dir)
    if actual != EXPECTED_FROZEN_SHA:
        raise RuntimeError(f"frozen GR_USER_v1 data SHA mismatch: {actual}")
    return actual


def _allocate_probe_targets(train_chain: list[dict], total: int) -> dict[tuple[str, str, str], int]:
    boundaries = quartile_boundaries([int(row["prompt_token_count"]) for row in train_chain])
    event_counts = collections.Counter(str(row["gold_event_count"]) for row in train_chain)
    event_targets = largest_remainder(dict(event_counts), total)
    global_source_counts = collections.Counter(
        "converted_cot" if row.get("converted_from_cot") else "native_nocot" for row in train_chain
    )
    global_source_targets = largest_remainder(dict(global_source_counts), total)
    if set(global_source_targets) != {"native_nocot", "converted_cot"}:
        raise ValueError("Chain source allocation requires native_nocot and converted_cot rows")

    # Preserve both event-count and source margins. Allocate converted rows by
    # fractional remainder, then use native rows as each event's complement.
    converted_raw = {}
    converted_targets = {}
    for event, event_target in sorted(event_targets.items()):
        event_rows = [row for row in train_chain if str(row["gold_event_count"]) == event]
        converted_rate = sum(row.get("converted_from_cot", False) for row in event_rows) / len(event_rows)
        converted_raw[event] = event_target * converted_rate
        converted_targets[event] = math.floor(converted_raw[event])
    remaining_converted = global_source_targets["converted_cot"] - sum(converted_targets.values())
    converted_order = sorted(
        converted_targets,
        key=lambda event: (-(converted_raw[event] - converted_targets[event]), -int(event)),
    )
    for event in converted_order[:remaining_converted]:
        converted_targets[event] += 1

    targets: dict[tuple[str, str, str], int] = {}
    for event, event_target in sorted(event_targets.items()):
        event_rows = [row for row in train_chain if str(row["gold_event_count"]) == event]
        source_targets = {
            "converted_cot": converted_targets[event],
            "native_nocot": event_target - converted_targets[event],
        }
        for source, source_target in sorted(source_targets.items()):
            source_rows = [
                row for row in event_rows
                if ("converted_cot" if row.get("converted_from_cot") else "native_nocot") == source
            ]
            quartile_counts = collections.Counter(
                quartile_for(int(row["prompt_token_count"]), boundaries) for row in source_rows
            )
            quartile_targets = largest_remainder(dict(quartile_counts), source_target)
            for quartile, target in quartile_targets.items():
                targets[(event, source, quartile)] = target
    if sum(targets.values()) != total:
        raise AssertionError("Probe v2 allocation does not sum to requested total")
    source_margin = collections.Counter()
    for (_event, source, _quartile), target in targets.items():
        source_margin[source] += target
    if dict(source_margin) != global_source_targets:
        raise AssertionError(f"Probe v2 source margin drift: {dict(source_margin)} != {global_source_targets}")
    return targets


def _select_targets(
    eligible: list[dict], targets: dict[tuple[str, str, str], int], boundaries: list[float]
) -> tuple[list[dict], list[dict]]:
    selected = []
    redistributions = []
    used = set()
    for (event, source, quartile), target in sorted(targets.items()):
        cell = [
            row for row in eligible
            if str(row["gold_event_count"]) == event
            and row["diagnosis_source"] == source
            and quartile_for(int(row["prompt_token_count"]), boundaries) == quartile
        ]
        cell.sort(key=lambda row: stable_key(row["sample_id"], f"probe-v2:{event}:{source}:{quartile}"))
        take = min(target, len(cell))
        selected.extend(cell[:take])
        used.update(row["sample_id"] for row in cell[:take])
        deficit = target - take
        if deficit:
            fallback = [
                row for row in eligible
                if str(row["gold_event_count"]) == event
                and row["diagnosis_source"] == source
                and row["sample_id"] not in used
            ]
            fallback.sort(key=lambda row: stable_key(row["sample_id"], f"probe-v2:fallback:{event}:{source}"))
            if len(fallback) < deficit:
                raise RuntimeError(
                    f"held-out source pool lacks {deficit} rows for event={event}, source={source}"
                )
            chosen = fallback[:deficit]
            selected.extend(chosen)
            used.update(row["sample_id"] for row in chosen)
            redistributions.append({
                "event_count": int(event),
                "source": source,
                "from_quartile": quartile,
                "count": deficit,
                "reason": "quartile_cell_exhausted",
            })
    if len(selected) != sum(targets.values()) or len(used) != len(selected):
        raise AssertionError("Probe v2 selection is incomplete or duplicated")
    return selected, redistributions


def build_probe_v2(args) -> None:
    from transformers import AutoTokenizer
    from build_gr_user_v1 import build_candidate

    data_dir = args.data_dir
    sha_before = validate_frozen_data(data_dir)
    train = read_jsonl(data_dir / "train_3000.jsonl")
    pilot = read_jsonl(data_dir / "pilot_600.jsonl")
    probe_v1 = read_jsonl(data_dir / "probe_v1.jsonl")
    pilot300_manifest = json.loads(args.pilot300_manifest.read_text(encoding="utf-8"))
    cumulative_ids = set(_flatten_ids(pilot300_manifest.get("cumulative300_sample_ids", [])))
    excluded_sets = {
        "train_3000": {row["sample_id"] for row in train},
        "pilot_600": {row["sample_id"] for row in pilot},
        "probe_v1": {row["sample_id"] for row in probe_v1},
        "cumulative_pilot300": cumulative_ids,
    }
    excluded = set().union(*excluded_sets.values())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    pools = []
    source_audit = {}
    specs = (("native_nocot", "user_chain_nocot.jsonl", False), ("converted_cot", "user_chain_cot.jsonl", True))
    for source_label, filename, converted in specs:
        path = args.source_dir / filename
        accepted = []
        rejected = collections.Counter()
        with path.open(encoding="utf-8") as handle:
            for source_line, line in enumerate(handle):
                row, reason = build_candidate(
                    tokenizer, path, source_line, json.loads(line), "chain", converted
                )
                if row is None:
                    rejected[reason] += 1
                    continue
                row["diagnosis_source"] = source_label
                accepted.append(row)
        eligible = [row for row in accepted if row["sample_id"] not in excluded]
        pools.extend(eligible)
        source_audit[source_label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "accepted": len(accepted),
            "eligible_after_exclusion": len(eligible),
            "rejections": dict(rejected),
        }

    train_chain = [row for row in train if row["route"] == "chain"]
    boundaries = quartile_boundaries([int(row["prompt_token_count"]) for row in train_chain])
    targets = _allocate_probe_targets(train_chain, args.sample_count)
    selected, redistributions = _select_targets(pools, targets, boundaries)
    selected.sort(key=lambda row: stable_key(row["sample_id"], "probe-v2:shuffle"))
    for row in selected:
        row["diagnosis_prompt_quartile"] = quartile_for(int(row["prompt_token_count"]), boundaries)
    selected_ids = {row["sample_id"] for row in selected}
    overlaps = {label: len(selected_ids & ids) for label, ids in excluded_sets.items()}
    if any(overlaps.values()):
        raise RuntimeError(f"Probe v2 held-out overlap contract failed: {overlaps}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)
    manifest = {
        "contract_version": "gr_user_chain_probe_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "sample_count": len(selected),
        "source": "real user_chain_nocot/user_chain_cot source pools; no synthetic rows",
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "frozen_sha_before": sha_before,
        "frozen_sha_after": frozen_hashes(data_dir),
        "held_out_overlap": overlaps,
        "source_audit": source_audit,
        "train_prompt_token_quartile_boundaries": boundaries,
        "target_cells": {"|".join(key): value for key, value in sorted(targets.items())},
        "actual_distribution": {
            "event_count": dict(sorted(collections.Counter(str(row["gold_event_count"]) for row in selected).items())),
            "source": dict(sorted(collections.Counter(row["diagnosis_source"] for row in selected).items())),
            "prompt_quartile": dict(sorted(collections.Counter(row["diagnosis_prompt_quartile"] for row in selected).items())),
        },
        "quartile_redistributions": redistributions,
        "sample_ids": [row["sample_id"] for row in selected],
    }
    if manifest["frozen_sha_after"] != sha_before:
        raise RuntimeError("frozen data changed while building Probe v2")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "probe": str(args.output), "manifest": str(args.manifest), **manifest["actual_distribution"]}, ensure_ascii=False))


def _gpu_preflight(physical_gpu_id: int) -> dict:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    state = None
    for row in rows:
        index, uuid, used, free, utilization = [part.strip() for part in row.split(",")]
        if int(index) == physical_gpu_id:
            state = {"index": int(index), "uuid": uuid, "memory_used_mib": int(used), "memory_free_mib": int(free), "utilization_percent": int(utilization)}
            break
    if state is None:
        raise RuntimeError(f"GPU {physical_gpu_id} does not exist")
    processes_raw = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    processes = []
    for row in processes_raw.splitlines() if processes_raw else []:
        uuid, pid, name, memory = [part.strip() for part in row.split(",", 3)]
        if uuid == state["uuid"]:
            processes.append({"pid": int(pid), "process_name": name, "used_memory_mib": int(memory)})
    if processes or state["memory_used_mib"] > 1024:
        raise RuntimeError(f"GPU {physical_gpu_id} is not idle: state={state}, processes={processes}")
    return {**state, "compute_processes": processes}


def _lora_checksums(model, limit: int = 16) -> dict[str, str]:
    import torch

    tensors = sorted((name, parameter) for name, parameter in model.named_parameters() if "lora_" in name.lower())
    if len(tensors) < 10:
        raise RuntimeError(f"found only {len(tensors)} LoRA tensors")
    output = {}
    for name, parameter in tensors[:limit]:
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        output[name] = hashlib.sha256(raw).hexdigest()
    return output


def _trim_generated(token_ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[:index], True
    return token_ids, False


def _grounding_label(statuses: list[dict]) -> str:
    values = [item.get("status") for item in statuses]
    if values and all(value == "grounded" for value in values):
        return "grounded"
    if not values or all(value == "ungrounded" for value in values):
        return "ungrounded"
    return "partially_grounded"


def evaluate_checkpoint(args) -> None:
    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if visible != [str(args.physical_gpu_id)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible} does not match physical GPU {args.physical_gpu_id}")
    gpu_before = _gpu_preflight(args.physical_gpu_id)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from user_chain_reward import score_chain
    from user_penalty_mask import compile_penalty_mask

    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnosis evaluator requires exactly one visible GPU")
    rows = read_jsonl(args.probe)
    if len(rows) != args.expected_samples or any(row.get("route") != "chain" for row in rows):
        raise RuntimeError("Probe v2 must contain the expected number of Chain rows")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    stop_ids.discard(None)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) for row in rows
    ]
    for row, text in zip(rows, rendered):
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual != row["prompt_token_count"]:
            raise RuntimeError(f"prompt token drift for {row['sample_id']}: {actual} != {row['prompt_token_count']}")

    python_state = random.getstate()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state("cuda:0").clone()
    output_rows = []
    model = base_model = None
    started = time.perf_counter()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True
        )
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False, local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("inference model has trainable parameters")
        checksum_before = _lora_checksums(model)
        with torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch_rows = rows[offset: offset + args.batch_size]
                batch_prompts = rendered[offset: offset + args.batch_size]
                batch_seed = SEED + offset
                random.seed(batch_seed)
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                encoded = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
                encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
                generated_all = model.generate(
                    **encoded, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                    num_return_sequences=G, max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=sorted(stop_ids), pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
                generated = generated_all[:, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
                for local_index, sample in enumerate(batch_rows):
                    for candidate_index in range(G):
                        ids, stopped = _trim_generated(generated[local_index * G + candidate_index], stop_ids)
                        completion = tokenizer.decode(ids, skip_special_tokens=False)
                        score = score_chain(completion, sample, tokenizer)
                        compiled = compile_penalty_mask(completion, score.violations, tokenizer, "chain")
                        records = []
                        for record in compiled["records"]:
                            records.append({
                                "kind": record["kind"],
                                "included": record["included"],
                                "masked_token_indices": record["masked_token_indices"],
                                "token_spans": record["token_spans"],
                            })
                        output_rows.append({
                            "checkpoint": args.label,
                            "sample_id": sample["sample_id"],
                            "event_count": int(sample["gold_event_count"]),
                            "source": sample["diagnosis_source"],
                            "prompt_quartile": sample["diagnosis_prompt_quartile"],
                            "prompt_token_count": int(sample["prompt_token_count"]),
                            "candidate_index": candidate_index,
                            "completion": completion,
                            "completion_token_count": len(ids),
                            "hit_max_new_tokens": not stopped and len(ids) >= MAX_NEW_TOKENS,
                            "total_reward": float(score.total_reward),
                            "action_alignment": float(score.action_f1),
                            "logic_alignment": float(score.logic_f1),
                            "format_valid": bool(score.format_valid),
                            "predicted_event_count": len(score.predicted_events),
                            "matched_event_count": len(score.matches),
                            "predicted_events": score.predicted_events,
                            "grounding": _grounding_label(score.grounding_status),
                            "violations": [item.kind for item in score.violations],
                            "penalty": {
                                "masked_token_count": int(sum(compiled["penalty_mask"])),
                                "records": records,
                            },
                        })
        torch.cuda.synchronize()
        checksum_after = _lora_checksums(model)
        if checksum_before != checksum_after:
            raise RuntimeError("LoRA checksum changed during inference-only diagnosis")
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, "cuda:0")
        if model is not None:
            del model
        if base_model is not None:
            del base_model
        torch.cuda.empty_cache()
    if len(output_rows) != len(rows) * G:
        raise RuntimeError("diagnosis candidate count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    integrity = {
        "label": args.label,
        "adapter": str(args.adapter),
        "probe_sha256": sha256_file(args.probe),
        "candidate_count": len(output_rows),
        "GPU": gpu_before,
        "G": G,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "requires_grad_parameter_count": 0,
        "lora_checksums_before": checksum_before,
        "lora_checksums_after": checksum_after,
        "lora_checksum_unchanged": checksum_before == checksum_after,
        "rng_restored": random.getstate() == python_state and torch.equal(torch.get_rng_state(), cpu_state),
        "wall_seconds": time.perf_counter() - started,
    }
    args.integrity.write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **integrity}, ensure_ascii=False))


def _candidate_violation_rate(rows: list[dict], kind: str) -> float:
    return sum(kind in row["violations"] for row in rows) / len(rows) if rows else 0.0


def _checkpoint_summary(rows: list[dict]) -> dict:
    return {
        "candidate_count": len(rows),
        "Total": statistics.fmean(row["total_reward"] for row in rows),
        "Action": statistics.fmean(row["action_alignment"] for row in rows),
        "Logic": statistics.fmean(row["logic_alignment"] for row in rows),
        "format_valid_rate": statistics.fmean(float(row["format_valid"]) for row in rows),
        "violations": {kind: _candidate_violation_rate(rows, kind) for kind in CHAIN_KINDS},
        "grounding": {
            label: sum(row["grounding"] == label for row in rows) / len(rows)
            for label in ("grounded", "partially_grounded", "ungrounded")
        },
        "completion_tokens": numeric_summary([row["completion_token_count"] for row in rows]),
        "truncation_rate": sum(row["hit_max_new_tokens"] for row in rows) / len(rows),
    }


def _group_means(rows: list[dict]) -> dict[str, dict]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    output = {}
    for sample_id, candidates in grouped.items():
        if len(candidates) != G:
            raise RuntimeError(f"sample {sample_id} does not have G={G} candidates")
        first = candidates[0]
        output[sample_id] = {
            "event_count": first["event_count"],
            "source": first.get("source"),
            "prompt_quartile": first.get("prompt_quartile"),
            "prompt_token_count": first["prompt_token_count"],
            **{metric: statistics.fmean(row[metric] for row in candidates) for metric in METRICS},
        }
    return output


def paired_deltas(groups: dict[str, dict[str, dict]], left: str, right: str) -> dict:
    ids = sorted(set(groups[left]) & set(groups[right]))
    if len(ids) != len(groups[left]) or len(ids) != len(groups[right]):
        raise RuntimeError(f"{left}/{right} are not matched on identical samples")
    output = {}
    for metric in METRICS:
        values = [groups[right][sample_id][metric] - groups[left][sample_id][metric] for sample_id in ids]
        output[metric] = {
            **numeric_summary(values),
            "improved": sum(value > UNCHANGED_EPSILON for value in values),
            "unchanged": sum(abs(value) <= UNCHANGED_EPSILON for value in values),
            "degraded": sum(value < -UNCHANGED_EPSILON for value in values),
        }
    return output


def bootstrap_mean_ci(values: list[float], seed: int = SEED, samples: int = 20000) -> list[float]:
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def analyze_existing_probe(path: Path, probe_v1_path: Path) -> dict:
    from user_chain_reward import ordered_action_matching

    rows = [row for row in read_jsonl(path) if row["route"] == "chain" and int(row["step"]) in (0, 20, 40)]
    samples = {row["sample_id"]: row for row in read_jsonl(probe_v1_path) if row["route"] == "chain"}
    by_checkpoint = {label: {} for label in CHECKPOINT_LABELS}
    events_by_checkpoint = {label: {} for label in CHECKPOINT_LABELS}
    step_label = {0: "C0", 20: "C20", 40: "C40"}
    for row in rows:
        label = step_label[int(row["step"])]
        sample = samples[row["group_id"]]
        candidates = row["candidates"]
        by_checkpoint[label][row["group_id"]] = {
            "event_count": int(sample["gold_event_count"]),
            "prompt_token_count": int(sample["prompt_token_count"]),
            "total_reward": statistics.fmean(candidate["total_reward"] for candidate in candidates),
            "action_alignment": statistics.fmean(candidate["action_alignment"] for candidate in candidates),
            "logic_alignment": statistics.fmean(candidate["logic_alignment"] for candidate in candidates),
        }
        events_by_checkpoint[label][row["group_id"]] = row
    comparisons = {
        "C20-C0": paired_deltas(by_checkpoint, "C0", "C20"),
        "C40-C0": paired_deltas(by_checkpoint, "C0", "C40"),
        "C40-C20": paired_deltas(by_checkpoint, "C20", "C40"),
    }
    sample_rows = []
    for sample_id in sorted(by_checkpoint["C0"]):
        item = {"sample_id": sample_id, **{key: by_checkpoint["C0"][sample_id][key] for key in ("event_count", "prompt_token_count")}}
        item["checkpoints"] = {label: by_checkpoint[label][sample_id] for label in CHECKPOINT_LABELS}
        item["deltas"] = {
            "C20-C0": {metric: by_checkpoint["C20"][sample_id][metric] - by_checkpoint["C0"][sample_id][metric] for metric in METRICS},
            "C40-C0": {metric: by_checkpoint["C40"][sample_id][metric] - by_checkpoint["C0"][sample_id][metric] for metric in METRICS},
            "C40-C20": {metric: by_checkpoint["C40"][sample_id][metric] - by_checkpoint["C20"][sample_id][metric] for metric in METRICS},
        }
        sample_rows.append(item)
    ranked = sorted(sample_rows, key=lambda item: item["deltas"]["C40-C0"]["total_reward"])

    def detailed(sample_id: str) -> dict:
        sample = samples[sample_id]
        checkpoints = {}
        for label in CHECKPOINT_LABELS:
            event = events_by_checkpoint[label][sample_id]
            checkpoints[label] = {
                "means": by_checkpoint[label][sample_id],
                "candidates": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "total_reward": candidate["total_reward"],
                        "action_alignment": candidate["action_alignment"],
                        "logic_alignment": candidate["logic_alignment"],
                        "predicted_event_count": len(candidate.get("predicted_events", [])),
                        "matched_event_count": len(ordered_action_matching(
                            candidate.get("predicted_events", []), sample["gold_events"]
                        )),
                        "violations": candidate.get("violations", []),
                        "grounding_penalty_kinds": candidate.get("penalty_kinds", []),
                        "completion_token_count": candidate.get("completion_length", 0),
                        "completion": _compact_completion(candidate["completion"], limit=1000),
                    } for candidate in event["candidates"]
                ],
            }
        signals = []
        delta = by_checkpoint["C40"][sample_id]
        base = by_checkpoint["C0"][sample_id]
        if abs(delta["action_alignment"] - base["action_alignment"]) > UNCHANGED_EPSILON:
            signals.append("event selection/action matching changed")
        if abs(delta["logic_alignment"] - base["logic_alignment"]) > UNCHANGED_EPSILON:
            signals.append("logic-text alignment changed")
        for kind, description in (
            ("hallucinated_sid", "SID grounding"),
            ("date_mismatch", "date grounding"),
            ("action_mismatch", "action grounding"),
            ("chronology_violation", "event order"),
        ):
            c0_rate = statistics.fmean(kind in candidate.get("violations", []) for candidate in events_by_checkpoint["C0"][sample_id]["candidates"])
            c40_rate = statistics.fmean(kind in candidate.get("violations", []) for candidate in events_by_checkpoint["C40"][sample_id]["candidates"])
            if not math.isclose(c0_rate, c40_rate):
                signals.append(f"{description} violation rate changed {c0_rate:.0%}->{c40_rate:.0%}")
        c0_length = statistics.fmean(candidate.get("completion_length", 0) for candidate in events_by_checkpoint["C0"][sample_id]["candidates"])
        c40_length = statistics.fmean(candidate.get("completion_length", 0) for candidate in events_by_checkpoint["C40"][sample_id]["candidates"])
        if abs(c40_length - c0_length) >= 20:
            signals.append(f"completion length changed {c0_length:.1f}->{c40_length:.1f} tokens")
        return {
            "sample_id": sample_id,
            "event_count": sample["gold_event_count"],
            "prompt_token_count": sample["prompt_token_count"],
            "gold_events": sample["gold_events"],
            "observed_attribution_signals": signals or ["no single large observed component change"],
            "checkpoints": checkpoints,
        }

    return {
        "sample_count": len(sample_rows),
        "unchanged_epsilon": UNCHANGED_EPSILON,
        "comparisons": comparisons,
        "per_sample": sample_rows,
        "top_degraded": [item["sample_id"] for item in ranked[:3]],
        "top_improved": [item["sample_id"] for item in ranked[-3:][::-1]],
        "top_degraded_details": [detailed(item["sample_id"]) for item in ranked[:3]],
        "top_improved_details": [detailed(item["sample_id"]) for item in ranked[-3:][::-1]],
    }


def _distribution(rows: list[dict]) -> dict:
    chain = [row for row in rows if row.get("route") == "chain"]
    count = len(chain)
    events = collections.Counter(str(row["gold_event_count"]) for row in chain)
    source = collections.Counter("converted_cot" if row.get("converted_from_cot") else "native_nocot" for row in chain)
    domains = collections.Counter(str(row.get("source_segment")) for row in chain if row.get("source_segment") is not None)
    return {
        "count": count,
        "event_count": {key: {"count": events[key], "proportion": events[key] / count} for key in ("2", "3", "4", "5")},
        "prompt_tokens": numeric_summary([row["prompt_token_count"] for row in chain]),
        "source": {key: {"count": value, "proportion": value / count} for key, value in sorted(source.items())},
        "domain_or_source_segment": {key: {"count": value, "proportion": value / count} for key, value in sorted(domains.items())},
    }


def distribution_audit(data_dir: Path, first_manifest: Path, second_manifest: Path) -> dict:
    train = read_jsonl(data_dir / "train_3000.jsonl")
    pilot = read_jsonl(data_dir / "pilot_600.jsonl")
    first = json.loads(first_manifest.read_text(encoding="utf-8"))
    second = json.loads(second_manifest.read_text(encoding="utf-8"))
    first_ids = set(_flatten_ids(first.get("selected_sample_ids", [])))
    second_ids = set(_flatten_ids(second.get("current_batch_sample_ids", second.get("selected_sample_ids", []))))
    by_id = {row["sample_id"]: row for row in pilot}
    sets = {
        "train_3000": train,
        "pilot_600": pilot,
        "first150": [by_id[sample_id] for sample_id in first_ids if sample_id in by_id],
        "second150": [by_id[sample_id] for sample_id in second_ids if sample_id in by_id],
        "cumulative300": [by_id[sample_id] for sample_id in first_ids | second_ids if sample_id in by_id],
    }
    output = {name: _distribution(rows) for name, rows in sets.items()}
    train_props = {key: value["proportion"] for key, value in output["train_3000"]["event_count"].items()}
    for name in ("pilot_600", "first150", "second150", "cumulative300"):
        output[name]["event_count_shift_vs_train"] = {
            key: {
                "pilot_proportion": output[name]["event_count"][key]["proportion"],
                "train_proportion": train_props[key],
                "ratio_pilot_to_train": output[name]["event_count"][key]["proportion"] / train_props[key],
            } for key in ("2", "3", "4", "5")
        }
    return output


def reward_constraint_conflict(rows_by_label: dict[str, list[dict]]) -> dict:
    output = {}
    for label, rows in rows_by_label.items():
        thresholds = {}
        for threshold in (0.6, 0.7):
            selected = [row for row in rows if row["total_reward"] >= threshold]
            thresholds[str(threshold)] = {
                "candidate_count": len(selected),
                "date_mismatch_rate": _candidate_violation_rate(selected, "date_mismatch"),
                "action_mismatch_rate": _candidate_violation_rate(selected, "action_mismatch"),
                "any_local_penalty_rate": sum(bool(row["penalty"]["masked_token_count"]) for row in selected) / len(selected) if selected else 0.0,
            }
        high_action = [row for row in rows if row["action_alignment"] >= 0.7]
        output[label] = {
            "reward_thresholds": thresholds,
            "action_alignment_ge_0.7_count": len(high_action),
            "action_alignment_ge_0.7_with_local_penalty_rate": sum(bool(row["penalty"]["masked_token_count"]) for row in high_action) / len(high_action) if high_action else 0.0,
        }
    return output


def token_attribution(rows_by_label: dict[str, list[dict]], base_lambda: float = 0.5) -> dict:
    from simulate_user_token_advantage import compile_effective_penalties, population_advantages, simulate_token_advantages

    output = {}
    for label, rows in rows_by_label.items():
        by_sample = collections.defaultdict(list)
        for row in rows:
            by_sample[row["sample_id"]].append(row)
        kinds = collections.defaultdict(lambda: {"candidate_ids": set(), "violations": 0, "masked_tokens": 0, "span_lengths": [], "incremental_negative_mass_proxy": 0.0})
        for sample_id, candidates in by_sample.items():
            candidates.sort(key=lambda row: row["candidate_index"])
            advantages, _mean, _std = population_advantages([row["total_reward"] for row in candidates])
            for candidate, advantage in zip(candidates, advantages):
                proxy = {
                    "route": "chain",
                    "completion_token_count": candidate["completion_token_count"],
                    "penalty": candidate["penalty"],
                }
                penalties, winners = compile_effective_penalties(
                    proxy, "sqrt", {kind: base_lambda for kind in CHAIN_KINDS}
                )
                token_advantages = simulate_token_advantages(advantage, penalties)
                baseline_negative = max(-advantage, 0.0)
                for index, winner_kinds in enumerate(winners):
                    if not winner_kinds:
                        continue
                    incremental = max(0.0, max(-token_advantages[index], 0.0) - baseline_negative)
                    for kind in winner_kinds:
                        kinds[kind]["incremental_negative_mass_proxy"] += incremental / len(winner_kinds)
                for record in candidate["penalty"]["records"]:
                    if not record["included"]:
                        continue
                    kind = record["kind"]
                    indices = sorted(set(record["masked_token_indices"]))
                    kinds[kind]["candidate_ids"].add((sample_id, candidate["candidate_index"]))
                    kinds[kind]["violations"] += 1
                    kinds[kind]["masked_tokens"] += len(indices)
                    kinds[kind]["span_lengths"].append(len(indices))
        output[label] = {
            kind: {
                "candidate_count": len(values["candidate_ids"]),
                "violation_count": values["violations"],
                "masked_token_count": values["masked_tokens"],
                "average_span_length": statistics.fmean(values["span_lengths"]) if values["span_lengths"] else 0.0,
                "incremental_negative_mass_proxy": values["incremental_negative_mass_proxy"],
            } for kind, values in sorted(kinds.items())
        }
    return output


def _compact_completion(text: str, limit: int = 2000) -> dict:
    return {"text": text[:limit], "truncated": len(text) > limit, "original_chars": len(text)}


def _per_sample_rows(probe_rows: list[dict], rows_by_label: dict[str, list[dict]]) -> list[dict]:
    probe = {row["sample_id"]: row for row in probe_rows}
    grouped = {label: collections.defaultdict(list) for label in CHECKPOINT_LABELS}
    for label, rows in rows_by_label.items():
        for row in rows:
            grouped[label][row["sample_id"]].append(row)
    means = {label: _group_means(rows_by_label[label]) for label in CHECKPOINT_LABELS}
    output = []
    for sample_id in sorted(probe):
        item = {
            "sample_id": sample_id,
            "event_count": probe[sample_id]["gold_event_count"],
            "prompt_token_count": probe[sample_id]["prompt_token_count"],
            "source": probe[sample_id]["diagnosis_source"],
            "prompt_quartile": probe[sample_id]["diagnosis_prompt_quartile"],
            "gold_events": probe[sample_id]["gold_events"],
            "checkpoints": {},
        }
        for label in CHECKPOINT_LABELS:
            candidates = sorted(grouped[label][sample_id], key=lambda row: row["candidate_index"])
            item["checkpoints"][label] = {
                "means": {metric: means[label][sample_id][metric] for metric in METRICS},
                "candidates": [
                    {
                        "candidate_index": row["candidate_index"],
                        "total_reward": row["total_reward"],
                        "action_alignment": row["action_alignment"],
                        "logic_alignment": row["logic_alignment"],
                        "predicted_event_count": row["predicted_event_count"],
                        "matched_event_count": row["matched_event_count"],
                        "grounding": row["grounding"],
                        "violations": row["violations"],
                        "completion": _compact_completion(row["completion"]),
                    } for row in candidates
                ],
            }
        item["deltas"] = {
            pair: {
                metric: means[right][sample_id][metric] - means[left][sample_id][metric]
                for metric in METRICS
            }
            for pair, left, right in (("C20-C0", "C0", "C20"), ("C40-C0", "C0", "C40"), ("C40-C20", "C20", "C40"))
        }
        output.append(item)
    return output


def _classification(c40_c0: dict) -> str:
    total = c40_c0["total_reward"]
    ci = total["bootstrap_95pct_ci"]
    degraded_rate = total["degraded"] / total["count"]
    if ci[1] < 0.0 and degraded_rate > 0.5:
        return "CONFIRMED_CHAIN_DEGRADATION"
    if ci[0] <= 0.0 <= ci[1] and abs(total["mean"]) < UNCHANGED_EPSILON and degraded_rate <= 0.55:
        return "SMALL_PROBE_NOISE"
    return "INCONCLUSIVE"


def render_markdown(summary: dict) -> str:
    existing = summary["existing_probe_8"]["comparisons"]["C40-C0"]
    v2 = summary["probe_v2"]
    lines = [
        "# GR_USER_v1 Chain Degradation Diagnosis",
        "",
        f"Final classification: **{summary['classification']}**",
        "",
        "This report is inference-only. No optimizer, backward pass, parameter update, training data mutation, or external benchmark was used.",
        "",
        "## Existing 8-sample fixed probe",
        "",
        f"Approximately unchanged means `abs(delta) <= {UNCHANGED_EPSILON}`.",
        "",
        "| Metric | Improved | Unchanged | Degraded | Mean delta | Median delta | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, label in (("total_reward", "Total"), ("action_alignment", "Action"), ("logic_alignment", "Logic")):
        row = existing[metric]
        lines.append(f"| {label} | {row['improved']} | {row['unchanged']} | {row['degraded']} | {row['mean']:.6f} | {row['p50']:.6f} | {row['min']:.6f} | {row['max']:.6f} |")
    lines.extend([
        "",
        f"Top degraded: {', '.join(summary['existing_probe_8']['top_degraded'])}",
        "",
        f"Top improved: {', '.join(summary['existing_probe_8']['top_improved'])}",
        "",
        "### Human-readable top sample comparisons",
        "",
        "The signals below are descriptive associations from candidate outputs, not training-causal claims.",
        "",
    ])
    for heading, key in (("Degraded", "top_degraded_details"), ("Improved", "top_improved_details")):
        lines.append(f"#### {heading}")
        lines.append("")
        for item in summary["existing_probe_8"][key]:
            lines.extend([
                f"##### `{item['sample_id']}`",
                "",
                f"Events/prompt tokens: {item['event_count']} / {item['prompt_token_count']}",
                "",
                f"Observed signals: {'; '.join(item['observed_attribution_signals'])}.",
                "",
                "Gold events:",
                "",
                "```json",
                json.dumps(item["gold_events"], ensure_ascii=False, indent=2),
                "```",
                "",
            ])
            for label in CHECKPOINT_LABELS:
                checkpoint = item["checkpoints"][label]
                means = checkpoint["means"]
                lines.extend([
                    f"{label} means: Total {means['total_reward']:.6f}, Action {means['action_alignment']:.6f}, Logic {means['logic_alignment']:.6f}.",
                    "",
                ])
                for candidate in checkpoint["candidates"]:
                    lines.append(
                        f"- Candidate {candidate['candidate_id']}: T/A/L "
                        f"{candidate['total_reward']:.4f}/{candidate['action_alignment']:.4f}/{candidate['logic_alignment']:.4f}; "
                        f"events={candidate['predicted_event_count']}; violations={candidate['violations']}; "
                        f"local={candidate['grounding_penalty_kinds']}; text={json.dumps(candidate['completion']['text'], ensure_ascii=False)}"
                    )
                lines.append("")
    lines.extend([
        "## Held-out Chain Probe v2",
        "",
        f"Samples: {v2['sample_count']}; SHA256: `{v2['sha256']}`.",
        "",
        "| Checkpoint | Total | Action | Logic | Grounded | Partial | Ungrounded |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label in CHECKPOINT_LABELS:
        row = v2["checkpoints"][label]
        lines.append(f"| {label} | {row['Total']:.6f} | {row['Action']:.6f} | {row['Logic']:.6f} | {row['grounding']['grounded']:.2%} | {row['grounding']['partially_grounded']:.2%} | {row['grounding']['ungrounded']:.2%} |")
    delta = v2["comparisons"]["C40-C0"]["total_reward"]
    lines.extend([
        "",
        f"C40-C0 Total mean/median: {delta['mean']:.6f} / {delta['p50']:.6f}; bootstrap 95% CI [{delta['bootstrap_95pct_ci'][0]:.6f}, {delta['bootstrap_95pct_ci'][1]:.6f}].",
        "",
        f"Degraded sample rate: {delta['degraded'] / delta['count']:.2%}.",
        "",
        f"Primary degradation component: **{v2['primary_degradation_component']}**.",
        "",
        "## Distribution audit",
        "",
        "Event-count ratios below are dataset proportion divided by train_3000 proportion.",
        "",
        "| Set | 2-event | 3-event | 4-event | 5-event |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("pilot_600", "first150", "second150", "cumulative300"):
        shifts = summary["pilot_distribution_audit"][name]["event_count_shift_vs_train"]
        lines.append(f"| {name} | {shifts['2']['ratio_pilot_to_train']:.2f}x | {shifts['3']['ratio_pilot_to_train']:.2f}x | {shifts['4']['ratio_pilot_to_train']:.2f}x | {shifts['5']['ratio_pilot_to_train']:.2f}x |")
    lines.extend([
        "",
        "## Reward / local-penalty conflict",
        "",
        "See the JSON result for checkpoint-specific >=0.6 and >=0.7 reward buckets, Action>=0.7 conflict rates, all violation rates, grounding, and sqrt-lambda=0.5 token attribution.",
        "",
        "## Decision rule",
        "",
        "CONFIRMED requires the C40-C0 Total bootstrap CI to be entirely below zero and a majority of samples to degrade by more than 0.01. SMALL_PROBE_NOISE requires a CI spanning zero, absolute mean below 0.01, and no more than 55% degraded. Other outcomes are INCONCLUSIVE.",
        "",
    ])
    return "\n".join(lines)


def generate_charts(summary: dict, per_sample: list[dict], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for item in per_sample:
        values = [item["checkpoints"][label]["means"]["total_reward"] for label in CHECKPOINT_LABELS]
        ax.plot(CHECKPOINT_LABELS, values, color="#7a8794", alpha=0.35, linewidth=1)
    means = [summary["probe_v2"]["checkpoints"][label]["Total"] for label in CHECKPOINT_LABELS]
    ax.plot(CHECKPOINT_LABELS, means, color="#c43d3d", linewidth=3, marker="o", label="mean")
    ax.set_ylabel("Chain Total")
    ax.set_title("Matched per-sample Chain Total")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "paired_slope_total.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    for metric, title, filename in (
        ("action_alignment", "C40-C0 Action Alignment", "action_delta_distribution.png"),
        ("logic_alignment", "C40-C0 Logic Alignment", "logic_delta_distribution.png"),
    ):
        values = [item["deltas"]["C40-C0"][metric] for item in per_sample]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=12, color="#4f7c6d", edgecolor="white")
        ax.axvline(0, color="#222222", linewidth=1)
        ax.axvline(statistics.fmean(values), color="#c43d3d", linewidth=2, label="mean")
        ax.set_title(title)
        ax.set_xlabel("Paired delta")
        ax.set_ylabel("Samples")
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    buckets = (2, 3, 4, 5)
    bucket_values = []
    for bucket in buckets:
        values = [item["deltas"]["C40-C0"]["total_reward"] for item in per_sample if item["event_count"] == bucket]
        bucket_values.append(statistics.fmean(values) if values else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(value) for value in buckets], bucket_values, color="#58799b")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xlabel("Gold event count")
    ax.set_ylabel("Mean C40-C0 Total")
    ax.set_title("Paired delta by event-count bucket")
    fig.tight_layout()
    path = output_dir / "event_bucket_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    distribution = summary["pilot_distribution_audit"]
    names = ("train_3000", "first150", "second150", "cumulative300")
    x = list(range(len(buckets)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(names):
        values = [distribution[name]["event_count"][str(bucket)]["proportion"] for bucket in buckets]
        ax.bar([value + (index - 1.5) * width for value in x], values, width=width, label=name)
    ax.set_xticks(x, [str(value) for value in buckets])
    ax.set_xlabel("Gold event count")
    ax.set_ylabel("Proportion")
    ax.set_title("Train vs Pilot Chain event-count distribution")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "train_vs_pilot_event_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def analyze(args) -> None:
    sha_before = validate_frozen_data(args.data_dir)
    manifest = json.loads(args.probe_manifest.read_text(encoding="utf-8"))
    probe_rows = read_jsonl(args.probe)
    rows_by_label = {label: read_jsonl(path) for label, path in zip(CHECKPOINT_LABELS, args.candidates)}
    evaluation_integrity = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in zip(CHECKPOINT_LABELS, args.integrity_files)
    }
    if any(
        not item.get("lora_checksum_unchanged") or item.get("requires_grad_parameter_count") != 0
        for item in evaluation_integrity.values()
    ):
        raise RuntimeError("checkpoint inference integrity contract failed")
    expected_ids = {row["sample_id"] for row in probe_rows}
    for label, rows in rows_by_label.items():
        if {row["sample_id"] for row in rows} != expected_ids or len(rows) != len(probe_rows) * G:
            raise RuntimeError(f"{label} evaluation is not matched to Probe v2")
    groups = {label: _group_means(rows) for label, rows in rows_by_label.items()}
    comparisons = {}
    for name, left, right in (("C20-C0", "C0", "C20"), ("C40-C0", "C0", "C40"), ("C40-C20", "C20", "C40")):
        comparison = paired_deltas(groups, left, right)
        for index, metric in enumerate(METRICS):
            values = [groups[right][sample_id][metric] - groups[left][sample_id][metric] for sample_id in sorted(expected_ids)]
            comparison[metric]["bootstrap_95pct_ci"] = bootstrap_mean_ci(values, seed=SEED + index)
        comparisons[name] = comparison
    action_delta = comparisons["C40-C0"]["action_alignment"]["mean"]
    logic_delta = comparisons["C40-C0"]["logic_alignment"]["mean"]
    total_negative = max(-action_delta, 0.0) + max(-logic_delta, 0.0)
    if total_negative == 0.0:
        component = "mixed"
    elif max(-action_delta, 0.0) / total_negative >= 0.65:
        component = "Action"
    elif max(-logic_delta, 0.0) / total_negative >= 0.65:
        component = "Logic"
    else:
        component = "mixed"
    per_sample = _per_sample_rows(probe_rows, rows_by_label)
    write_jsonl(args.per_sample_output, per_sample)
    summary = {
        "contract_version": "gr_user_chain_diagnosis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inference_contract": {"G": G, "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
        "integrity": {
            "frozen_sha_before": sha_before,
            "frozen_sha_after": frozen_hashes(args.data_dir),
            "frozen_sha_unchanged": frozen_hashes(args.data_dir) == sha_before,
            "training_performed": False,
            "optimizer_step": False,
            "backward": False,
            "checkpoint_modified": False,
            "external_benchmark_run": False,
            "checkpoint_evaluations": evaluation_integrity,
        },
        "existing_probe_8": analyze_existing_probe(args.existing_probes, args.data_dir / "probe_v1.jsonl"),
        "probe_v2": {
            "sample_count": len(probe_rows),
            "source": manifest["source"],
            "sha256": sha256_file(args.probe),
            "distribution": manifest["actual_distribution"],
            "checkpoints": {label: _checkpoint_summary(rows) for label, rows in rows_by_label.items()},
            "comparisons": comparisons,
            "primary_degradation_component": component,
            "per_sample_output": str(args.per_sample_output),
        },
        "pilot_distribution_audit": distribution_audit(args.data_dir, args.pilot150_manifest, args.pilot300_manifest),
        "reward_constraint_conflict": reward_constraint_conflict(rows_by_label),
        "token_local_penalty_attribution": {
            "strategy": "sqrt",
            "lambda": 0.5,
            "mass_definition": "incremental negative mass beyond unpenalized task advantage",
            "checkpoints": token_attribution(rows_by_label),
        },
    }
    if not summary["integrity"]["frozen_sha_unchanged"]:
        raise RuntimeError("frozen data changed during analysis")
    summary["classification"] = _classification(comparisons["C40-C0"])
    summary["figures"] = generate_charts(summary, per_sample, args.charts_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.docs_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.docs_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": "PASS", "classification": summary["classification"], "output": str(args.output), "docs": str(args.docs_output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build = subparsers.add_parser("build-probe")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--pilot300-manifest", type=Path, required=True)
    build.add_argument("--sample-count", type=int, default=40)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--label", choices=EVALUATION_LABELS, required=True)
    evaluate.add_argument("--base-model", type=Path, required=True)
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument("--probe", type=Path, required=True)
    evaluate.add_argument("--physical-gpu-id", type=int, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--expected-samples", type=int, default=40)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--integrity", type=Path, required=True)

    aggregate = subparsers.add_parser("analyze")
    aggregate.add_argument("--data-dir", type=Path, required=True)
    aggregate.add_argument("--probe", type=Path, required=True)
    aggregate.add_argument("--probe-manifest", type=Path, required=True)
    aggregate.add_argument("--candidates", type=Path, nargs=3, required=True)
    aggregate.add_argument("--integrity-files", type=Path, nargs=3, required=True)
    aggregate.add_argument("--existing-probes", type=Path, required=True)
    aggregate.add_argument("--pilot150-manifest", type=Path, required=True)
    aggregate.add_argument("--pilot300-manifest", type=Path, required=True)
    aggregate.add_argument("--per-sample-output", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--docs-output", type=Path, required=True)
    aggregate.add_argument("--charts-dir", type=Path, default=Path("/data/GRPO_USER/results/diagnosis"))
    args = parser.parse_args()
    if args.mode == "build-probe":
        build_probe_v2(args)
    elif args.mode == "evaluate":
        evaluate_checkpoint(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
