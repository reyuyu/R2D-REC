#!/usr/bin/env python3
"""Build and fail-closed audit the Mini-Short-CoT production SID8 cache."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from datasets import load_from_disk


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_mini_short_cot_4gpu_gc04_2epoch.yaml"
DATA_ROOT = Path("/data/lf_data_versions/alltrain/mini_short_cot")
CACHE = DATA_ROOT / "tokenized_train_8k_sid8w8"
REPORT = DATA_ROOT / "cache_audit.json"
MATERIAL_MANIFEST = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json")
IGNORE_INDEX = -100
SID_RE = re.compile(r"^<s_[abc]_\d+>$")
DOMAIN_RE = re.compile(r"^<\|(?:ad|video|prod|living)_begin\|>$")


def load_native() -> None:
    os.chdir(ROOT)
    sys.argv = ["build_and_audit_mini_short_cot_cache.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_mini_short_cot_cache", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    native.install_native_patches()


def main() -> None:
    os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
    os.environ["MATERIAL_DOMAIN_MANIFEST"] = str(MATERIAL_MANIFEST)
    load_native()
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parser_raw = dict(raw)
    for key in tuple(parser_raw):
        if key.startswith("alpha_") or key in {"rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled"}:
            parser_raw.pop(key)
    model_args, data_args, training_args, _, _ = _parse_train_args(parser_raw)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    module = None
    if not CACHE.exists():
        module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    packed = load_from_disk(str(CACHE))["train"]
    if module is not None and len(module["train_dataset"]) != len(packed):
        raise RuntimeError("Fresh module/cache packed-row mismatch")

    added = tokenizer.get_added_vocab()
    item_ids = {int(token_id) for token, token_id in added.items() if SID_RE.match(token) or DOMAIN_RE.match(token)}
    item_lookup = np.zeros(max(item_ids) + 1, dtype=np.bool_)
    item_lookup[list(item_ids)] = True
    segments = Counter()
    weights = Counter()
    violations = Counter()
    total_segments = 0
    for row in packed:
        labels = np.asarray(row["labels"], dtype=np.int64)
        sample_ids = np.asarray(row["sample_ids"], dtype=np.int64)
        task_ids = np.asarray(row["sample_task_ids"], dtype=np.int64)
        loss_weights = np.asarray(row["loss_weights"], dtype=np.float64)
        for value, count in zip(*np.unique(loss_weights, return_counts=True)):
            weights[str(float(value))] += int(count)
        violations["ignore_nonzero_weight"] += int(np.count_nonzero(loss_weights[labels == IGNORE_INDEX]))
        positions = np.flatnonzero(sample_ids >= 0)
        splits = np.flatnonzero(np.diff(sample_ids[positions]) != 0) + 1 if positions.size else []
        for run in (part for part in np.split(positions, splits) if part.size):
            valid = run[labels[run] != IGNORE_INDEX]
            if not valid.size:
                continue
            task_id = int(task_ids[valid[0]])
            route = ("material", "recommendation", "user_action", "user_chain")[task_id]
            segments[route] += 1
            total_segments += 1
            segment_weights = loss_weights[valid]
            segment_labels = labels[valid]
            if route == "material" and np.all(segment_weights == 4.0):
                continue
            item_mask = item_lookup[segment_labels]
            violations["item_not_8"] += int(np.count_nonzero(segment_weights[item_mask] != 8.0))
            violations["ordinary_not_1"] += int(np.count_nonzero(segment_weights[~item_mask] != 1.0))

    expected = {"material": 36298, "recommendation": 11192, "user_action": 1200, "user_chain": 800}
    if dict(segments) != expected or total_segments != 49490:
        raise RuntimeError(f"Packed segment contract failed: {dict(segments)}")
    if any(violations.values()):
        raise RuntimeError(f"SID8 cache contract failed: {dict(violations)}")
    if weights.get("2.0", 0) or weights.get("3.0", 0):
        raise RuntimeError("Forbidden weight 2/3 found")
    steps_per_epoch = math.ceil(math.ceil(len(packed) / 4) / 16)
    total_steps = steps_per_epoch * 2
    report = {
        "pass": True,
        "config": str(CONFIG),
        "cache": str(CACHE),
        "packs": len(packed),
        "segments": dict(segments),
        "weight_counts": dict(weights),
        "weight_contract_violations": dict(violations),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_2epoch": total_steps,
        "warmup_steps": math.ceil(total_steps * float(raw["warmup_ratio"])),
        "mini_fix_parent_packs": 4365,
        "pack_delta_percent": 100.0 * (len(packed) / 4365 - 1.0),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
