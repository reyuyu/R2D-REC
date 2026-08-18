#!/usr/bin/env python3
"""Fail-closed preflight for Mini-Short-CoT."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_mini_short_cot_4gpu_gc04_2epoch.yaml"
DATA_ROOT = Path("/data/lf_data_versions/alltrain/mini_short_cot")
MANIFEST = DATA_ROOT / "manifest.json"
AUDIT = DATA_ROOT / "cache_audit.json"
DOC = ROOT / "docs/experiment_MINI_SHORT_COT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (CONFIG, MANIFEST, AUDIT, DOC):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    dataset = DATA_ROOT / manifest["file"]
    expected_config = {
        "dataset": "onereason_mini_short_cot",
        "dataset_dir": str(DATA_ROOT),
        "tokenized_path": str(DATA_ROOT / "tokenized_train_8k_sid8w8"),
        "cutoff_len": 8192,
        "packing": True,
        "neat_packing": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "warmup_ratio": 0.03,
        "seed": 20260806,
        "rec_pu_enabled": False,
        "multitask_pack_ratio_enabled": False,
        "rec_candidate_metrics_enabled": False,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected_config.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Config mismatch: {mismatches}")
    if manifest["counts"]["total_rows"] != 49490:
        raise RuntimeError("Dataset total must be 49,490")
    if manifest["counts"]["recommendation_cot_transformed"] != 6235:
        raise RuntimeError("All 6,235 recommendation CoT rows must be transformed")
    if not all(manifest["parity"].values()):
        raise RuntimeError(f"Dataset parity failed: {manifest['parity']}")
    if manifest["cot_audit"]["later_section_markers_remaining"] != 0:
        raise RuntimeError("Later recommendation CoT sections remain")
    if sha256(dataset) != manifest["sha256"]:
        raise RuntimeError("Dataset SHA-256 mismatch")
    if not audit.get("pass") or audit["segments"] != {
        "material": 36298, "recommendation": 11192, "user_action": 1200, "user_chain": 800
    }:
        raise RuntimeError("Cache audit failed")
    if any(audit["weight_contract_violations"].values()):
        raise RuntimeError("SID8 weight contract failed")
    result = {
        "status": "PASS",
        "dataset_sha256": manifest["sha256"],
        "transformed_cot": 6235,
        "packs": audit["packs"],
        "steps_per_epoch": audit["steps_per_epoch"],
        "total_steps": audit["total_steps_2epoch"],
        "warmup_steps": audit["warmup_steps"],
    }
    print("MINI_SHORT_COT_PREFLIGHT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
