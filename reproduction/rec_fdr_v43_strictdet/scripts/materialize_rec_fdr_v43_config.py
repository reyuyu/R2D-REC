#!/usr/bin/env python3
"""Materialize absolute paths in the portable V4.3 training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.package_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else root / "data"
    work = args.work_root.resolve()
    config = yaml.safe_load((root / "training/training_config.template.yaml").read_text(encoding="utf-8"))
    config.update(
        {
            "model_name_or_path": str(args.base_model.resolve()),
            "dataset_dir": str(data_root),
            "curriculum_tokenized_pool_root": str(work / "cache/recommendation_32k"),
            "base_tokenized_pool_root": str(work / "cache/base_32k"),
            "rec_group_catalog_path": str(data_root / "recommendation/rec_group_catalog.json"),
            "output_dir": str(work / "output"),
        }
    )
    config["hcr"]["metadata_path"] = str(data_root / "recommendation/hcr_group_metadata.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
