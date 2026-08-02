#!/usr/bin/env python3
"""Create deterministic 98/2 train/dev splits for the eight multitask JSONLs.

The split is stable per JSONL line content and seed, so rerunning it does not
move examples between train and dev.  The script also registers the resulting
``*_train98`` and ``*_dev2`` names in LLaMA-Factory's dataset registry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


DATASETS = (
    "onereason_material_cot",
    "onereason_material_nocot",
    "onereason_user_action_nocot",
    "onereason_user_chain_cot",
    "onereason_user_chain_nocot",
    "onereason_recommendation_cot",
    "onereason_world_cot",
    "onereason_world_nocot",
)


def is_dev(line: bytes, seed: int, percent_basis_points: int) -> bool:
    digest = hashlib.sha256(str(seed).encode() + b"\0" + line).digest()
    return int.from_bytes(digest[:8], "big") % 10_000 < percent_basis_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("/data/lf_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/lf_data_splits"))
    parser.add_argument("--dataset-info", type=Path, default=Path("/app/LLaMA-Factory/data/dataset_info.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-percent", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    basis_points = round(args.dev_percent * 100)
    if not 0 < basis_points < 10_000:
        raise ValueError("--dev-percent must be between 0 and 100.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.dataset_info.open(encoding="utf-8") as file:
        dataset_info = json.load(file)

    manifest = {"seed": args.seed, "dev_percent": args.dev_percent, "datasets": {}}
    for name in DATASETS:
        source = args.source_dir / f"{name}.jsonl"
        train_path = args.output_dir / f"{name}_train98.jsonl"
        dev_path = args.output_dir / f"{name}_dev2.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        if not args.overwrite and (train_path.exists() or dev_path.exists()):
            raise FileExistsError(f"Split already exists for {name}; pass --overwrite to rebuild it.")

        train_count = dev_count = 0
        with source.open("rb") as reader, train_path.open("wb") as train_file, dev_path.open("wb") as dev_file:
            for line in reader:
                if not line.strip():
                    continue
                target = dev_file if is_dev(line.rstrip(b"\r\n"), args.seed, basis_points) else train_file
                target.write(line if line.endswith(b"\n") else line + b"\n")
                if target is dev_file:
                    dev_count += 1
                else:
                    train_count += 1

        if name not in dataset_info:
            raise KeyError(f"{name} is not registered in {args.dataset_info}")
        for suffix, path in (("_train98", train_path), ("_dev2", dev_path)):
            entry = copy.deepcopy(dataset_info[name])
            entry["file_name"] = str(path)
            dataset_info[name + suffix] = entry
        manifest["datasets"][name] = {
            "source": str(source),
            "train_file": str(train_path),
            "dev_file": str(dev_path),
            "train_dataset": name + "_train98",
            "dev_dataset": name + "_dev2",
            "train_count": train_count,
            "dev_count": dev_count,
        }
        print(f"{name}: train={train_count}, dev={dev_count}")

    with args.dataset_info.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (args.output_dir / "multitask_dev_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
