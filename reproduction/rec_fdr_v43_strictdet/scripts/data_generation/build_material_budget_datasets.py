#!/usr/bin/env python3
"""Build deterministic material-budget datasets with full user/recommendation data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_balanced_lora_dataset as balanced


TRAIN_ROOT = Path("/data/LLm-8B/code/train")
MATERIAL_NAMES = (
    "material_no_think_semantic_to_sid",
    "material_no_think_sid_to_semantic",
    "material_think_semantic_to_sid",
    "material_think_sid_to_semantic",
)
FULL_NAMES = (
    "user_think",
    "user_no_think",
    "recommendation_think",
)


def dataset_info() -> dict[str, object]:
    return {
        name: {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        for name in (*MATERIAL_NAMES, *FULL_NAMES)
    }


def build(material_rows: int, overwrite: bool) -> None:
    if material_rows <= 0 or material_rows % len(MATERIAL_NAMES):
        raise ValueError("material_rows must be positive and divisible by four")

    label = f"material{material_rows // 1000}k_userrec_full_v1"
    output = TRAIN_ROOT / "data" / label
    report_path = TRAIN_ROOT / "reports" / f"{label}.json"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise SystemExit(f"Output is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    balanced.OUTPUT = output
    quota = material_rows // len(MATERIAL_NAMES)
    report: dict[str, object] = {
        "seed": balanced.SEED,
        "source": str(balanced.SOURCE),
        "output": str(output),
        "material_budget": material_rows,
        "material_quota_per_task": quota,
        "material": {},
        "datasets": {},
    }

    selected_sids: set[tuple[str, int, int, int]] = set()
    for name in MATERIAL_NAMES:
        _, stats, current_sids = balanced.select_material(name, quota, selected_sids)
        report["material"][name] = stats
        selected_sids.update(current_sids)
        print(f"{label}: selected {name}: {quota}", flush=True)

    for name in FULL_NAMES:
        rows = balanced.copy_full_dataset(name)
        report["datasets"][name] = {"source_rows": rows, "selected_rows": rows}
        print(f"{label}: copied {name}: {rows}", flush=True)

    info_path = output / "dataset_info.json"
    info_path.write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["material_unique_sid_union"] = len(selected_sids)
    report["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": balanced.sha256_file(path)}
        for path in sorted(output.glob("*.jsonl"))
    }
    report["total_rows"] = material_rows + sum(
        entry["selected_rows"] for entry in report["datasets"].values()
    )
    temp_report = report_path.with_suffix(".json.tmp")
    temp_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temp_report, report_path)
    print(json.dumps({"dataset": label, "total_rows": report["total_rows"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-rows", type=int, nargs="+", default=[140_000, 40_000])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for material_rows in args.material_rows:
        build(material_rows, args.overwrite)


if __name__ == "__main__":
    main()
