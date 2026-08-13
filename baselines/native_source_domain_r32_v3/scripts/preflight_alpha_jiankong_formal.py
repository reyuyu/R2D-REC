#!/usr/bin/env python3
"""Fail-closed preflight for the formal alpha-jiankong monitor-only run.

This script deliberately performs no model loading and no mutation.  It checks
the frozen YAML, registered split, packed caches, split/probe isolation and the
source-aware native loss contract before the launcher is allowed to start
``torch.distributed.run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from datasets import load_from_disk

from alpha_sid8_cache_contract import AlphaSID8CacheContractError, scan_alpha_sid8_cache


EXPECTED = {
    "dataset": "onereason_alpha_jiankong_train98",
    "alpha_validation_dev_dataset": "onereason_alpha_jiankong_dev2",
    "dataset_dir": "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1",
    "tokenized_path": "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_train98_8k_sid8w8",
    "alpha_validation_dev_cache": "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev2_8k_sid8w8",
    "alpha_validation_probe_cache": "/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev_probe_v1_8k_sid8w8",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2.0e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "cutoff_len": 8192,
    "packing": True,
    "neat_packing": True,
    "flash_attn": "fa2",
    "enable_liger_kernel": True,
    "bf16": True,
    "pure_bf16": True,
    "gradient_checkpointing": True,
    "use_reentrant_gc": False,
    "seed": 20260806,
    "resume_from_checkpoint": None,
    "rec_pu_enabled": False,
    "multitask_pack_ratio_enabled": False,
    "rec_candidate_metrics_enabled": False,
    "alpha_monitor_enabled": True,
    "alpha_train_tf_enabled": True,
    "alpha_train_tf_interval": 50,
    "alpha_validation_enabled": True,
    "alpha_dev_probe_enabled": True,
    "alpha_dev_probe_interval": 100,
    "alpha_full_dev_enabled": True,
    "alpha_full_dev_at_epoch_end": True,
    "save_strategy": "epoch",
}

SPLIT_DIR = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")
PROBE_DIR = SPLIT_DIR / "probe_v1"
# This is recomputed from probe_v1/manifest.json raw_row_sha256.  A prior
# hand-written note had ``...8506c146...``; the immutable artifact is
# ``...8506a368...`` and is the only value accepted by the launcher.
EXPECTED_PROBE_SHA = "664098072283f45566b5d0c614eb8506a3683bede39786c1464aa437682aeee4"
EXPECTED_MATERIAL_WEIGHTS = {
    "video": 1.1791795483099141,
    "prod": 0.7738397930531297,
    "ad": 1.133452723686895,
    "living": 0.8980530210503628,
}
EXPECTED_MATERIAL_COUNTS = {"video": 30092, "prod": 29180, "ad": 22768, "living": 17960}


class PreflightError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise PreflightError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # concise fail-closed error below
        raise PreflightError(f"Cannot read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"Expected JSON object: {path}")
    return value


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _line_hashes(path: Path) -> set[str]:
    digest: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            digest.add(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return digest


def _cache_len(path: str) -> int:
    dataset = load_from_disk(path)
    if "train" not in dataset:
        _fail(f"Packed cache must contain train split: {path}")
    return len(dataset["train"])


def _validate_config(config: dict[str, Any]) -> None:
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            _fail(f"Frozen config mismatch for {key}: expected {expected!r}, got {config.get(key)!r}")
    if "max_steps" in config and config["max_steps"] not in (None, -1):
        _fail(f"Stale max_steps is forbidden, got {config['max_steps']!r}")
    if config.get("load_best_model_at_end"):
        _fail("load_best_model_at_end must remain disabled")
    if config.get("metric_for_best_model"):
        _fail("metric_for_best_model must not control training")
    if config.get("lr_scheduler_type") == "reduce_lr_on_plateau":
        _fail("ReduceLROnPlateau is forbidden for monitor-only validation")
    if config.get("model_name_or_path") != "/data/models/onereason-8b-pretrain-competition":
        _fail("Wrong base model")
    if not isinstance(config.get("alpha_validation_metrics_path"), str) or "SMOKE" in config["alpha_validation_metrics_path"]:
        _fail("Formal validation JSONL must use a non-smoke run directory")


def _validate_registry_and_split(config: dict[str, Any]) -> tuple[int, int, int, int]:
    registry = _load_json(SPLIT_DIR / "dataset_info.json")
    if registry.get("onereason_alpha_jiankong_train98", {}).get("file_name") != "train.jsonl":
        _fail("Train registry is wrong")
    if registry.get("onereason_alpha_jiankong_dev2", {}).get("file_name") != "dev.jsonl":
        _fail("Dev registry is wrong")
    train_rows = _count_lines(SPLIT_DIR / "train.jsonl")
    dev_rows = _count_lines(SPLIT_DIR / "dev.jsonl")
    if (train_rows, dev_rows) != (216732, 4520):
        _fail(f"Unexpected split row counts: train={train_rows}, dev={dev_rows}")
    audit = _load_json(SPLIT_DIR / "split_audit.json")
    leakage = audit.get("leakage", {})
    required_zero = (
        "recommendation_group_id_overlap", "recommendation_history_domain_overlap",
        "user_family_overlap", "exact_full_row_overlap", "canonical_prompt_overlap",
    )
    if not audit.get("overall", {}).get("train") == train_rows or not audit.get("overall", {}).get("dev") == dev_rows:
        _fail("Split audit row counts disagree with JSONL")
    if not audit.get("leakage", {}).get("row_conservation") or any(leakage.get(key) != 0 for key in required_zero):
        _fail(f"Split leakage audit failed: {leakage}")
    train_packs = _cache_len(config["tokenized_path"])
    dev_packs = _cache_len(config["alpha_validation_dev_cache"])
    if (train_packs, dev_packs) != (33810, 733):
        _fail(f"Unexpected packed cache sizes: train={train_packs}, dev={dev_packs}")
    return train_rows, dev_rows, train_packs, dev_packs


def _validate_probe(config: dict[str, Any]) -> tuple[int, int, int]:
    manifest = _load_json(PROBE_DIR / "manifest.json")
    if manifest.get("probe_sha256") != EXPECTED_PROBE_SHA:
        _fail("Probe SHA mismatch; refusing to rebuild or substitute a probe")
    if (manifest.get("raw_rows"), manifest.get("selected_groups")) != (176, 85):
        _fail("Probe row/group count mismatch")
    probe_rows = _count_lines(PROBE_DIR / "probe.jsonl")
    probe_packs = _cache_len(config["alpha_validation_probe_cache"])
    if (probe_rows, probe_packs) != (176, 42):
        _fail(f"Probe row/pack count mismatch: rows={probe_rows}, packs={probe_packs}")
    probe_hashes = list(manifest.get("raw_row_sha256", []))
    if len(probe_hashes) != probe_rows:
        _fail("Probe manifest hash count mismatch")
    computed_probe = hashlib.sha256("\n".join(probe_hashes).encode("ascii")).hexdigest()
    if computed_probe != EXPECTED_PROBE_SHA:
        _fail("Probe manifest hashes do not reproduce fixed probe SHA")
    dev_hashes = _line_hashes(SPLIT_DIR / "dev.jsonl")
    train_hashes = _line_hashes(SPLIT_DIR / "train.jsonl")
    probe_hash_set = set(probe_hashes)
    if not probe_hash_set <= dev_hashes or probe_hash_set & train_hashes:
        _fail("Probe isolation failed: probe must be dev-only and train-disjoint")
    group_ids: set[str] = set()
    routes: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    with (PROBE_DIR / "probe.jsonl").open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            metadata = json.loads(row["aux_metadata_json"])
            group_ids.add(str(metadata["recommendation_group_id"]))
            routes[str(row["source_segment"])] += 1
            gold = str(metadata["recommendation_current_gold_sid"])
            for domain in ("video", "prod", "ad", "living"):
                if gold.startswith(f"<|{domain}_begin|>"):
                    domains[domain] += 1
                    break
    if len(group_ids) != 85 or not routes["recommendation_cot"] or not routes["recommendation_nocot"] or any(not domains[key] for key in ("video", "prod", "ad", "living")):
        _fail("Probe coverage/group integrity failed")
    return probe_rows, len(group_ids), probe_packs


def _validate_native_contract() -> None:
    if os.getenv("GLOBAL_ITEM_WEIGHT") != "8":
        _fail("GLOBAL_ITEM_WEIGHT must be exactly 8")
    manifest_path = Path(os.getenv("MATERIAL_DOMAIN_MANIFEST", ""))
    if not manifest_path.is_file():
        _fail("MATERIAL_DOMAIN_MANIFEST must point to the frozen baseline manifest")
    material = _load_json(manifest_path)
    if material.get("material_domain_counts") != EXPECTED_MATERIAL_COUNTS:
        _fail("Material domain counts differ from frozen baseline")
    if material.get("material_domain_weights") != EXPECTED_MATERIAL_WEIGHTS:
        _fail("Material domain weights differ from frozen baseline")
    launcher = Path(__file__).with_name("train_native_source_domain_r32_v3.py").read_text(encoding="utf-8")
    required_fragments = (
        "CANONICAL_TEXT_WEIGHT = 4.0",
        "if source == CANONICAL_SOURCE:",
        "if source != \"material_sample\":",
        "return MATERIAL_DOMAIN_WEIGHTS[match.group(1)]",
        "compute_native_sid8_loss",
    )
    if any(fragment not in launcher for fragment in required_fragments):
        _fail("Native SID8/material loss route source audit failed")


def _validate_persisted_sid8_cache(config: dict[str, Any]) -> dict[str, object]:
    """Trust the cache contents, not merely the environment variable."""
    try:
        return scan_alpha_sid8_cache(config["tokenized_path"])
    except AlphaSID8CacheContractError as error:
        _fail(f"Persisted SID8 cache contract failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            _fail("Formal YAML must be a mapping")
        _validate_config(config)
        train_rows, dev_rows, train_packs, dev_packs = _validate_registry_and_split(config)
        probe_rows, probe_groups, probe_packs = _validate_probe(config)
        _validate_native_contract()
        sid8_cache = _validate_persisted_sid8_cache(config)
        per_update = args.world_size * int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])
        steps_per_epoch = math.ceil(train_packs / per_update)
        total_steps = math.ceil(steps_per_epoch * float(config["num_train_epochs"]))
        warmup_steps = math.ceil(total_steps * float(config["warmup_ratio"]))
        payload = {
            "status": "ALPHA_FORMAL_PREFLIGHT_PASS",
            "dataset_train": config["dataset"], "dataset_dev": config["alpha_validation_dev_dataset"],
            "train_raw_rows": train_rows, "dev_raw_rows": dev_rows,
            "train_packs": train_packs, "dev_packs": dev_packs,
            "world_size": args.world_size, "global_batch_packs": per_update,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
            "rec_pu": "OFF", "pack_ratio": "OFF", "alpha_monitor": "ON",
            "train_tf_interval": config["alpha_train_tf_interval"],
            "probe_rows": probe_rows, "probe_groups": probe_groups, "probe_packs": probe_packs,
            "probe_sha256": EXPECTED_PROBE_SHA, "probe_interval": config["alpha_dev_probe_interval"],
            "full_dev_epoch_end": "ON", "full_dev_packs": dev_packs,
            "sid_domain_weight": 8, "canonical_response_weight": 4,
            "sid8_cache_packs": sid8_cache["packs"],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print("ALPHA_FORMAL_PREFLIGHT")
            for key, value in payload.items():
                if key != "status":
                    print(f"{key}={value}")
    except PreflightError as error:
        print(f"ALPHA_FORMAL_PREFLIGHT_FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
