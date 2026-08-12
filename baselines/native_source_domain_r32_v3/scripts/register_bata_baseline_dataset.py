#!/usr/bin/env python3
"""Register the immutable bata_baseline dataset in LLaMA-Factory and version metadata."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATASET_NAME = "onereason_bata_baseline"
VERSION_NAME = "bata_baseline_v1"


def atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".bata_baseline.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--global-registry", type=Path, default=Path("/app/LLaMA-Factory/data/dataset_info.json"))
    parser.add_argument(
        "--versions-registry", type=Path, default=Path("/app/LLaMA-Factory/data/onereason_dataset_versions.json")
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    dataset_file = dataset_dir / "onereason_bata_baseline.jsonl"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("name") != VERSION_NAME or not dataset_file.is_file():
        raise ValueError("Dataset directory is not a complete bata_baseline_v1 build")
    if manifest.get("experiment_name") != "pure bata_baseline":
        manifest["experiment_name"] = "pure bata_baseline"
        atomic_json_write(dataset_dir / "manifest.json", manifest)

    registry = json.loads(args.global_registry.read_text(encoding="utf-8"))
    registry[DATASET_NAME] = {
        "file_name": str(dataset_file),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
            "system": "system",
        },
    }
    atomic_json_write(args.global_registry, registry)

    versions = json.loads(args.versions_registry.read_text(encoding="utf-8"))
    versions.setdefault("versions", {})[VERSION_NAME] = {
        "parent": "BETA_material_aligned_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "description": (
            "Clean reproduction baseline: archive material, SID buckets and recommendation are exact; "
            "only understand_user is replaced by active BETA."
        ),
        "world_included": False,
        "extra_registry_entries": {
            DATASET_NAME: {
                "registry_name": DATASET_NAME,
                "file_name": str(dataset_file),
                "records": manifest["records"],
                "sha256": manifest["sha256"],
            }
        },
        "reproduction_contract": {
            "archive_parquet_sha256": manifest["archive_parquet_sha256"],
            "archive_non_user_projection_sha256": manifest["archive_non_user_projection_sha256"],
            "archive_non_user_projection_preserved": True,
            "intentional_difference": manifest["intentional_difference"],
            "source_counts": manifest["source_counts"],
            "material_domain_weights": manifest["material_domain_weights"],
            "recommendation_metadata": manifest["recommendation_metadata"],
        },
    }
    atomic_json_write(args.versions_registry, versions)
    print(f"REGISTERED dataset={DATASET_NAME} version={VERSION_NAME} file={dataset_file}")


if __name__ == "__main__":
    main()
