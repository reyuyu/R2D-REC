#!/usr/bin/env python3
"""Hard preflight for the clean bata_baseline dataset and training route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from build_bata_baseline_dataset import (
    EXPECTED_ARCHIVE_COUNTS,
    EXPECTED_BETA_USER_COUNT,
    EXPECTED_MATERIAL_DOMAINS,
    EXPECTED_OUTPUT_COUNTS,
    MATERIAL_DOMAIN_WEIGHTS,
    archive_rows,
    beta_user_rows,
    convert_archive,
    recommendation_identity,
    sha256_file,
)


EXPECTED_DATASET_NAME = "onereason_bata_baseline"
EXPECTED_DATASET_ROOT = Path("/data/lf_data_versions/alltrain/bata_baseline_v1")
LOCKED_CONFIG = {
    "model_name_or_path": "/data/models/onereason-8b-pretrain-competition",
    "trust_remote_code": True,
    "flash_attn": "fa2",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "enable_liger_kernel": True,
    "dataset": EXPECTED_DATASET_NAME,
    "dataset_dir": str(EXPECTED_DATASET_ROOT),
    "template": "qwen3_nothink",
    "cutoff_len": 8192,
    "packing": True,
    "neat_packing": True,
    "overwrite_cache": False,
    "preprocessing_num_workers": 16,
    "dataloader_num_workers": 8,
    "remove_unused_columns": False,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "save_only_model": False,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "report_to": "none",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2.0e-4,
    "num_train_epochs": 2,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "bf16": True,
    "pure_bf16": True,
    "gradient_checkpointing": True,
    "use_reentrant_gc": False,
    "ddp_timeout": 180000000,
    "seed": 20260806,
    "resume_from_checkpoint": None,
    "rec_pu_enabled": False,
    "multitask_pack_ratio_enabled": False,
    "rec_candidate_metrics_enabled": True,
    "rec_candidate_metrics_interval": 50,
}


def load_groups(archive: Path) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in archive_rows(archive):
        if row["data_source"] != "recommend":
            continue
        key, current = recommendation_identity(row)
        if current not in groups[key]:
            groups[key].append(current)
    return groups


def audit_dataset(manifest: dict, dataset: Path) -> None:
    archive = Path(manifest["archive_parquet"])
    beta = Path(manifest["beta_dataset"])
    if sha256_file(archive) != manifest["archive_parquet_sha256"]:
        raise AssertionError("Archive Parquet SHA256 differs from manifest")
    if sha256_file(beta) != manifest["beta_dataset_sha256"]:
        raise AssertionError("BETA source SHA256 differs from manifest")
    if sha256_file(dataset) != manifest["sha256"]:
        raise AssertionError("bata_baseline JSONL SHA256 differs from manifest")

    groups = load_groups(archive)
    users = iter(beta_user_rows(beta))
    counts: Counter[str] = Counter()
    segments: Counter[str] = Counter()
    recommendation_rows = 0
    replaced_users = 0
    appended_users = 0

    with dataset.open(encoding="utf-8") as output:
        for line_number, archive_row in enumerate(archive_rows(archive), 1):
            actual = json.loads(next(output))
            if archive_row["data_source"] == "understand_user":
                expected = next(users)
                if actual != expected:
                    raise AssertionError(f"BETA user replacement changed at output row {line_number}")
                replaced_users += 1
            else:
                expected = convert_archive(archive_row, groups)
                if actual != expected:
                    raise AssertionError(f"Archive row changed at output row {line_number}")
                if actual["data_source"] == "recommend":
                    metadata = json.loads(actual["aux_metadata_json"])
                    if metadata["recommendation_group_size"] != len(metadata["recommendation_all_gold_sids"]):
                        raise AssertionError(f"Recommendation metadata size differs at row {line_number}")
                    if metadata["recommendation_current_gold_sid"] not in metadata["recommendation_all_gold_sids"]:
                        raise AssertionError(f"Recommendation current gold is absent at row {line_number}")
                    recommendation_rows += 1
            counts[actual["data_source"]] += 1
            segments[actual["source_segment"]] += 1

        for expected in users:
            actual = json.loads(next(output))
            if actual != expected:
                raise AssertionError("Appended BETA user row changed")
            counts[actual["data_source"]] += 1
            segments[actual["source_segment"]] += 1
            appended_users += 1
        if next(output, None) is not None:
            raise AssertionError("Dataset contains rows beyond the declared sources")

    if dict(counts) != EXPECTED_OUTPUT_COUNTS or dict(counts) != manifest["source_counts"]:
        raise AssertionError(f"Output route counts differ: {dict(counts)}")
    if dict(segments) != manifest["source_segment_counts"]:
        raise AssertionError(f"Output segment counts differ: {dict(segments)}")
    if replaced_users != EXPECTED_ARCHIVE_COUNTS["understand_user"] or appended_users != 396:
        raise AssertionError("BETA user replacement layout differs")
    if recommendation_rows != EXPECTED_ARCHIVE_COUNTS["recommend"]:
        raise AssertionError("Recommendation metadata coverage differs")
    if len(groups) != manifest["recommendation_metadata"]["groups"]:
        raise AssertionError("Recommendation group count differs")


def audit_registry(dataset: Path) -> None:
    registry = json.loads(Path("/app/LLaMA-Factory/data/dataset_info.json").read_text(encoding="utf-8"))
    entry = registry.get(EXPECTED_DATASET_NAME)
    if entry is None or Path(entry["file_name"]).resolve() != dataset.resolve():
        raise AssertionError("Global LLaMA-Factory registry does not select bata_baseline")
    versions = json.loads(Path("/app/LLaMA-Factory/data/onereason_dataset_versions.json").read_text(encoding="utf-8"))
    version = versions.get("versions", {}).get("bata_baseline_v1")
    if version is None or version.get("reproduction_contract", {}).get("archive_non_user_projection_preserved") is not True:
        raise AssertionError("Version registry lacks the bata_baseline reproduction contract")


def audit_config(config: dict) -> None:
    differences = {key: (config.get(key), expected) for key, expected in LOCKED_CONFIG.items() if config.get(key) != expected}
    if differences:
        raise AssertionError(f"Training config differs from locked reproduction fields: {differences}")
    if config.get("rec_pu_enabled") is not False or config.get("multitask_pack_ratio_enabled") is not False:
        raise AssertionError("REC-PU and PackRatio must both be disabled")


def audit_runtime_loss_route(launcher: Path, config_path: Path, manifest_path: Path) -> None:
    original_argv = sys.argv[:]
    original_manifest = os.environ.get("MATERIAL_DOMAIN_MANIFEST")
    original_weight = os.environ.get("GLOBAL_ITEM_WEIGHT")
    try:
        sys.argv = [str(launcher), str(config_path)]
        os.environ["MATERIAL_DOMAIN_MANIFEST"] = str(manifest_path)
        os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
        spec = importlib.util.spec_from_file_location("bata_baseline_training_route", launcher)
        if spec is None or spec.loader is None:
            raise AssertionError("Cannot load the native training launcher")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        labels = [-100, 101, 202]
        item_ids = {101}
        expected = {
            "material_sample": [0.0, 8.0, 1.0],
            "sid_bucket_canonical_no_think": [0.0, 4.0, 4.0],
            "sid_bucket_reverse": [0.0, 8.0, 1.0],
            "understand_user": [0.0, 8.0, 1.0],
            "recommend": [0.0, 8.0, 1.0],
        }
        actual = {source: module._build_loss_weights(labels, source, item_ids) for source in expected}
        if actual != expected:
            raise AssertionError(f"Runtime loss routes differ: {actual}")
        for domain, weight in MATERIAL_DOMAIN_WEIGHTS.items():
            prompt = [{"content": f"<|{domain}_begin|>"}]
            if module._material_domain_weight("material_sample", prompt, []) != weight:
                raise AssertionError(f"Material domain weight differs for {domain}")
    finally:
        sys.argv = original_argv
        if original_manifest is None:
            os.environ.pop("MATERIAL_DOMAIN_MANIFEST", None)
        else:
            os.environ["MATERIAL_DOMAIN_MANIFEST"] = original_manifest
        if original_weight is None:
            os.environ.pop("GLOBAL_ITEM_WEIGHT", None)
        else:
            os.environ["GLOBAL_ITEM_WEIGHT"] = original_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = args.manifest.parent / "onereason_bata_baseline.jsonl"
    if manifest.get("name") != "bata_baseline_v1" or manifest.get("world_included") is not False:
        raise AssertionError("Manifest identity differs")
    if manifest.get("material_domain_counts") != EXPECTED_MATERIAL_DOMAINS:
        raise AssertionError("Material domain counts differ in manifest")
    if manifest.get("material_domain_weights") != MATERIAL_DOMAIN_WEIGHTS:
        raise AssertionError("Material domain weights differ in manifest")
    if manifest.get("archive_counts") != EXPECTED_ARCHIVE_COUNTS:
        raise AssertionError("Archive route counts differ in manifest")
    if manifest.get("source_counts") != EXPECTED_OUTPUT_COUNTS:
        raise AssertionError("Output route counts differ in manifest")

    audit_config(config)
    audit_dataset(manifest, dataset)
    audit_registry(dataset)
    audit_runtime_loss_route(args.launcher, args.config, args.manifest)
    print(
        "PASS bata_baseline records=222001 archive_non_user=exact beta_user=32848 "
        "recommend_metadata=48269 rec_pu=OFF pack_ratio=OFF SID8=8 canonical=4 GC_fraction=0.4"
    )


if __name__ == "__main__":
    main()
