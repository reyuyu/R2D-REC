#!/usr/bin/env python3
"""CPU-only, exact native preprocessing audit for alpha_mini_v1.

Builds a fresh tokenized cache with the production tokenizer/template and
reports the packed dataset exposure. It never loads a model or touches CUDA.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_mini_v1_4gpu_gc04_2epoch.yaml"
DATASET = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/onereason_alpha_mini_v1.jsonl")
CACHE = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/tokenized_alpha_mini_v1_train_8k_sid8w8")
REPORT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/alpha_mini_v1_native_packing_audit.json")
MATERIAL_MANIFEST = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json")
CUTOFF = 8192
TASKS = ("material", "recommendation", "user_action", "user_chain")
IGNORE_INDEX = -100


def _load_native():
    os.chdir(ROOT)
    sys.argv = ["audit_alpha_mini_v1_cpu.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_alpha_mini_audit", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    native.install_native_patches()
    return native


def _row_to_messages(row: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompt: list[dict[str, str]] = []
    history = row.get("history", [])
    if isinstance(history, list):
        for pair in history:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("history must contain [prompt, response] pairs")
            prompt.append({"role": "user", "content": str(pair[0])})
            prompt.append({"role": "assistant", "content": str(pair[1])})
    query = [str(row[field]) for field in ("instruction", "input") if row.get(field)]
    prompt.append({"role": "user", "content": "\n".join(query)})
    return prompt, [{"role": "assistant", "content": str(row["output"])}]


def _untruncated_length(processor, prompt, response, system: str) -> int:
    """Exact qwen3_nothink message encoding before the normal cutoff policy."""
    messages = processor.template.mm_plugin.process_messages(prompt + response, [], [], [], processor.processor)
    input_ids, _ = processor.template.mm_plugin.process_token_ids(
        [], [], [], [], [], processor.tokenizer, processor.processor
    )
    pairs = processor.template.encode_multiturn(processor.tokenizer, messages, system, "", False)
    return len(input_ids) + sum(len(source) + len(target) for source, target in pairs) + int(processor.template.efficient_eos)


def _post_template_audit(native, data_args, tokenizer_module, template) -> dict:
    from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor

    processor = PackedSupervisedDatasetProcessor(template, tokenizer_module["tokenizer"], tokenizer_module.get("processor"), data_args)
    raw_total = 0
    raw_over = Counter()
    reached_cutoff = Counter()
    accepted = Counter()
    actual_max = 0
    raw_max = 0
    rows = 0
    with DATASET.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            segment = str(row["source_segment"])
            prompt, response = _row_to_messages(row)
            raw_length = _untruncated_length(processor, prompt, response, str(row.get("system", "")))
            input_ids, _ = processor._encode_data_example(prompt, response, str(row.get("system", "")), "", [], [], [])
            actual_length = len(input_ids)
            if actual_length > CUTOFF:
                raise RuntimeError(f"post-template length exceeds cutoff at row {line_no}: {actual_length}")
            rows += 1
            raw_total += raw_length
            raw_max = max(raw_max, raw_length)
            actual_max = max(actual_max, actual_length)
            accepted[segment] += 1
            if raw_length > CUTOFF:
                raw_over[segment] += 1
            if actual_length == CUTOFF:
                reached_cutoff[segment] += 1
    return {
        "input_rows": rows,
        "actual_encoder_rows_leq_8192": rows,
        "actual_encoder_rows_over_8192": 0,
        "actual_encoder_max_length": actual_max,
        "untruncated_template_max_length": raw_max,
        "untruncated_template_mean_length": raw_total / rows,
        "untruncated_template_over_8192": dict(sorted(raw_over.items())),
        "untruncated_template_over_8192_total": sum(raw_over.values()),
        "encoder_reached_8192_after_deterministic_cutoff": dict(sorted(reached_cutoff.items())),
        "encoder_reached_8192_after_deterministic_cutoff_total": sum(reached_cutoff.values()),
        "accepted_before_packing_by_source_segment": dict(sorted(accepted.items())),
        "filtered_by_native_length_check": 0,
    }


def _cache_stats(dataset) -> dict:
    segment_counts = Counter()
    supervised_tokens = Counter()
    pack_shares = Counter()
    metadata = {
        "packed_rows_with_rec_targets": 0,
        "rec_target_segments": 0,
        "rec_target_components": 0,
        "rec_target_sid8_weight_failures": 0,
        "rec_target_label_location_failures": 0,
        "rec_target_source_segment_failures": 0,
    }
    for row in dataset:
        labels = row["labels"]
        tasks = row["sample_task_ids"]
        sample_ids = row["sample_ids"]
        weights = row["loss_weights"]
        present: dict[int, int] = {}
        for position in range(1, len(labels)):
            sample_id = int(sample_ids[position])
            if labels[position] == IGNORE_INDEX or sample_id < 0:
                continue
            task = int(tasks[position])
            if task < 0 or task >= len(TASKS):
                raise RuntimeError("valid supervised token has an invalid task id")
            supervised_tokens[TASKS[task]] += 1
            previous = present.setdefault(sample_id, task)
            if previous != task:
                raise RuntimeError("a packed segment crossed task routes")
        counts = Counter(present.values())
        for task, count in counts.items():
            segment_counts[TASKS[task]] += count
        total_segments = sum(counts.values())
        if not total_segments:
            raise RuntimeError("packed row has no supervised segment")
        for task, count in counts.items():
            pack_shares[TASKS[task]] += count / total_segments

        targets = json.loads(row.get("rec_pu_targets_json", "[]"))
        if targets:
            metadata["packed_rows_with_rec_targets"] += 1
        for target in targets:
            metadata["rec_target_segments"] += 1
            if target.get("source_segment") not in {"recommendation_cot", "recommendation_nocot"}:
                metadata["rec_target_source_segment_failures"] += 1
            for level in ("a", "b", "c"):
                label_pos = int(target[f"{level}_label_position"])
                logit_pos = int(target[f"{level}_logit_position"])
                metadata["rec_target_components"] += 1
                if label_pos - 1 != logit_pos or labels[label_pos] == IGNORE_INDEX:
                    metadata["rec_target_label_location_failures"] += 1
                if float(weights[label_pos]) != 8.0:
                    metadata["rec_target_sid8_weight_failures"] += 1

    packed_rows = len(dataset)
    total_supervised = sum(supervised_tokens.values())
    if metadata["rec_target_segments"] != segment_counts["recommendation"]:
        raise RuntimeError("recommendation target metadata count does not equal packed recommendation segment count")
    if any(metadata[key] for key in (
        "rec_target_sid8_weight_failures", "rec_target_label_location_failures", "rec_target_source_segment_failures"
    )):
        raise RuntimeError(f"recommendation packed-target audit failed: {metadata}")
    return {
        "packed_rows": packed_rows,
        "accepted_segments": {task: int(segment_counts[task]) for task in TASKS},
        "accepted_segments_total": int(sum(segment_counts.values())),
        "supervised_tokens": {task: int(supervised_tokens[task]) for task in TASKS},
        "supervised_tokens_total": int(total_supervised),
        "supervised_token_share": {task: supervised_tokens[task] / total_supervised for task in TASKS},
        "e_share_segment_normalized_pack_mean": {task: pack_shares[task] / packed_rows for task in TASKS},
        "recommendation_metadata": metadata,
        "optimizer_steps_per_epoch_4gpu_batch1_ga16": (packed_rows + 63) // 64,
    }


def main() -> None:
    if CACHE.exists():
        raise RuntimeError(f"Refusing to reuse or overwrite existing Mini cache: {CACHE}")
    os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
    os.environ["MATERIAL_DOMAIN_MANIFEST"] = str(MATERIAL_MANIFEST)
    native = _load_native()
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer
    from datasets import load_from_disk

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # Alpha sidecar keys are deliberately consumed by the native launcher,
    # not by LLaMA-Factory's ordinary argument parser used for this CPU audit.
    for key in tuple(raw):
        if key.startswith("alpha_") or key in {
            "rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled",
        }:
            raw.pop(key)
    model_args, data_args, training_args, _, _ = _parse_train_args(raw)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    length_audit = _post_template_audit(native, data_args, tokenizer_module, template)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    cache_data = load_from_disk(str(CACHE))
    cache_stats = _cache_stats(cache_data["train"])
    expected = length_audit["actual_encoder_rows_leq_8192"]
    if cache_stats["accepted_segments_total"] != expected:
        raise RuntimeError(f"packed segments {cache_stats['accepted_segments_total']} != accepted rows {expected}")
    report = {
        "kind": "alpha_mini_v1_exact_native_cpu_audit",
        "config": str(CONFIG),
        "dataset": str(DATASET),
        "cache": str(CACHE),
        "model_loaded": False,
        "cuda_available": False,
        "template": "qwen3_nothink",
        "cutoff_len": CUTOFF,
        "packing": "neat",
        "loss_route": "native_sid8_ce_global_item_weight_8",
        "length_audit": length_audit,
        "packing": cache_stats,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
