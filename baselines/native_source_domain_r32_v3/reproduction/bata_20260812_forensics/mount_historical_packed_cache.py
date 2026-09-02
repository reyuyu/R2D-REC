#!/usr/bin/env python3
"""Expose immutable Hugging Face Arrow cache shards through load_from_disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset


EXPECTED_COLUMNS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
    "loss_weights",
    "sample_ids",
    "sample_task_ids",
    "sample_domain_weights",
    "pack_task_id",
    "images",
    "videos",
    "audios",
    "packing_params",
]
EXPECTED_ROWS = 35_380
EXPECTED_SHARDS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--mount-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", default=[])
    args = parser.parse_args()

    shards = sorted(args.cache_dir.glob(f"{args.cache_prefix}_*.arrow"))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"Expected {EXPECTED_SHARDS} cache shards, found {len(shards)}")
    if args.mount_dir.exists():
        raise FileExistsError(f"Refusing to reuse cache mount: {args.mount_dir}")

    datasets = [Dataset.from_file(str(path)) for path in shards]
    shard_rows = [len(dataset) for dataset in datasets]
    if sum(shard_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {sum(shard_rows)}")
    for index, dataset in enumerate(datasets):
        if dataset.column_names != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected columns in shard {index}: {dataset.column_names}")

    train_dir = args.mount_dir / "train"
    train_dir.mkdir(parents=True)
    data_files = []
    shard_records = []
    for index, (source, row_count) in enumerate(zip(shards, shard_rows, strict=True)):
        filename = f"data-{index:05d}-of-{len(shards):05d}.arrow"
        destination = train_dir / filename
        os.symlink(source, destination)
        data_files.append({"filename": filename})
        shard_records.append(
            {
                "index": index,
                "source": str(source),
                "source_size": source.stat().st_size,
                "source_sha256": sha256_file(source),
                "rows": row_count,
                "mount": str(destination),
            }
        )

    (args.mount_dir / "dataset_dict.json").write_text(
        json.dumps({"splits": ["train"]}, sort_keys=True), encoding="utf-8"
    )
    state = {
        "_data_files": data_files,
        "_fingerprint": "historical-bata-cache-234024f6c7f34028",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (train_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    datasets[0].info.write_to_directory(str(train_dir))

    for config_path in args.config:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["tokenized_path"] = str(args.mount_dir)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
        )

    record = {
        "status": "PASS",
        "cache_dir": str(args.cache_dir),
        "cache_prefix": args.cache_prefix,
        "mount_dir": str(args.mount_dir),
        "shard_count": len(shards),
        "row_count": sum(shard_rows),
        "columns": EXPECTED_COLUMNS,
        "shards": shard_records,
        "configs": [str(path) for path in args.config],
    }
    record_path = args.mount_dir / "HISTORICAL_CACHE_MOUNT.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
