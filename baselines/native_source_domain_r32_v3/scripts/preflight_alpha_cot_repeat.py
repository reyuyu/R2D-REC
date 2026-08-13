#!/usr/bin/env python3
"""Fail-closed preflight for the isolated Alpha CoT-repeat cache experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from alpha_sid8_cache_contract import AlphaSID8CacheContractError, scan_alpha_sid8_cache
from audit_alpha_cot_repeat_cache import AuditError, audit


SPLIT = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")
BASELINE_CACHE = SPLIT / "tokenized_train98_8k_sid8w8"
BAD_CACHE = SPLIT / "tokenized_train98_8k"
NEW_CACHE = Path("/app/lf_tokenized/alpha-jiankong-split-v1/tokenized_train98_8k_sid8w8_cot05n")
MANIFEST = SPLIT / "alpha_cot_repeat_count_manifest_v1.json"


class PreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if os.environ.get("GLOBAL_ITEM_WEIGHT") != "8":
            raise PreflightError("GLOBAL_ITEM_WEIGHT must be exactly 8.")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise PreflightError("YAML must be a mapping.")
        required = {
            "dataset": "onereason_alpha_jiankong_train98",
            "dataset_dir": str(SPLIT),
            "tokenized_path": str(NEW_CACHE),
            "alpha_cot_repeat_weighting": True,
            "alpha_cot_repeat_manifest": str(MANIFEST),
            "rec_pu_enabled": False,
            "multitask_pack_ratio_enabled": False,
            "resume_from_checkpoint": None,
            "num_train_epochs": 2,
            "gradient_accumulation_steps": 16,
            "cutoff_len": 8192,
            "packing": True,
            "neat_packing": True,
        }
        bad = {key: (expected, config.get(key)) for key, expected in required.items() if config.get(key) != expected}
        if bad:
            raise PreflightError(f"Dedicated YAML contract mismatch: {bad}")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("kind") != "alpha_cot_repeat_count_manifest_v1" or manifest.get("groups_with_cot") != 13698 or manifest.get("recommendation_cot_rows") != 27186:
            raise PreflightError("Repeat-count manifest is not the audited train98 manifest.")
        if manifest.get("source_train_sha256") != _sha256(SPLIT / "train.jsonl"):
            raise PreflightError("train98 SHA differs from the repeat-count manifest.")
        baseline = scan_alpha_sid8_cache(BASELINE_CACHE)
        try:
            scan_alpha_sid8_cache(BAD_CACHE)
        except AlphaSID8CacheContractError:
            old_bad_cache_rejected = True
        else:
            raise PreflightError("The historical SID2/3 cache unexpectedly passed the strict baseline contract.")
        result = audit(BASELINE_CACHE, NEW_CACHE)
        payload = {
            "status": "ALPHA_COT_REPEAT_PREFLIGHT_PASS",
            "baseline_sid8_contract_packs": baseline["packs"],
            "old_sid23_cache_rejected": old_bad_cache_rejected,
            "experimental_cache": str(NEW_CACHE),
            "experiment_packs": result["packs"],
            "cot_body_ratio": result["cot_body_ratio"],
            "final_sid_weighted_mass": result["final_sid_new_weighted_mass"],
            "nothink_weighted_mass": result["nothink_new_weighted_mass"],
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True) if args.json else "ALPHA_COT_REPEAT_PREFLIGHT_PASS")
    except (OSError, ValueError, AuditError, AlphaSID8CacheContractError, PreflightError) as error:
        print(f"ALPHA_COT_REPEAT_PREFLIGHT_FAIL: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
