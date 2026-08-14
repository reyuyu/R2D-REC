#!/usr/bin/env python3
"""Fail-closed static preflight for mini-cot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from audit_mini_cot_repeat_cache import audit


ROOT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1")
SOURCE = ROOT / "onereason_alpha_mini_v1.jsonl"
BASELINE = ROOT / "tokenized_alpha_mini_v1_train_8k_sid8w8"
NEW = ROOT / "tokenized_alpha_mini_v1_train_8k_sid8w8_cot05n"
MANIFEST = ROOT / "mini_cot_repeat_count_manifest_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GLOBAL_ITEM_WEIGHT") != "8":
        raise SystemExit("MINI_COT_PREFLIGHT_FAIL: GLOBAL_ITEM_WEIGHT must be 8")
    for p in (SOURCE, BASELINE, NEW, MANIFEST):
        if not p.exists():
            raise SystemExit(f"MINI_COT_PREFLIGHT_FAIL: missing {p}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required = {
        "dataset": "onereason_alpha_mini_v1",
        "dataset_dir": str(ROOT),
        "tokenized_path": str(NEW),
        "alpha_cot_repeat_weighting": True,
        "alpha_cot_repeat_manifest": str(MANIFEST),
        "rec_pu_enabled": False,
        "multitask_pack_ratio_enabled": False,
        "resume_from_checkpoint": None,
        "cutoff_len": 8192,
        "packing": True,
        "neat_packing": True,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 2,
    }
    mismatches = {k: (v, config.get(k)) for k, v in required.items() if config.get(k) != v}
    if mismatches:
        raise SystemExit(f"MINI_COT_PREFLIGHT_FAIL: config mismatch {mismatches}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("kind") != "mini_cot_repeat_count_manifest_v1":
        raise SystemExit("MINI_COT_PREFLIGHT_FAIL: wrong manifest kind")
    if manifest.get("source_train_sha256") != sha256(SOURCE):
        raise SystemExit("MINI_COT_PREFLIGHT_FAIL: source SHA mismatch")
    if manifest.get("total_rows") != 49490 or manifest.get("recommendation_cot_rows") != 6235 or manifest.get("recommendation_nocot_rows") != 4957:
        raise SystemExit(f"MINI_COT_PREFLIGHT_FAIL: unexpected source counts {manifest}")
    group_counts = manifest.get("group_cot_counts", {})
    if not group_counts or any(not isinstance(k, str) or not isinstance(v, int) or v <= 0 for k, v in group_counts.items()):
        raise SystemExit("MINI_COT_PREFLIGHT_FAIL: invalid group counts")
    parity = audit(BASELINE, NEW)
    print(json.dumps({
        "status": "MINI_COT_PREFLIGHT_PASS",
        "source": str(SOURCE),
        "baseline_cache": str(BASELINE),
        "mini_cot_cache": str(NEW),
        "manifest": str(MANIFEST),
        "groups_with_cot": len(group_counts),
        "cot_rows": manifest["recommendation_cot_rows"],
        "packs": parity["packs"],
        "changed_loss_weight_positions": parity["changed_loss_weight_positions"],
        "parity": parity["status"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
