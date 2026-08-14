#!/usr/bin/env python3
"""Compare alpha_mini_v1 with Alpha's fixed dev and probe without mutation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
MINI = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/onereason_alpha_mini_v1.jsonl")
DEV = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/dev.jsonl")
PROBE = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/probe_v1/probe.jsonl")
OUT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/alpha_mini_v1_vs_alpha_validation_leakage.json")


def _load_split_module():
    path = ROOT / "scripts/split_alpha_jiankong_validation.py"
    spec = importlib.util.spec_from_file_location("alpha_split_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _audit(split, left_path: Path, right_path: Path) -> dict:
    left, _ = split.load_rows(left_path)
    right, _ = split.load_rows(right_path)
    left_exact = {row.row_hash for row in left}
    right_exact = {row.row_hash for row in right}
    left_prompt = {row.prompt for row in left}
    right_prompt = {row.prompt for row in right}
    left_groups = {row.rec_group_id for row in left if row.task == "recommendation"}
    right_groups = {row.rec_group_id for row in right if row.task == "recommendation"}
    left_hdom = {row.rec_history_domain for row in left if row.task == "recommendation"}
    right_hdom = {row.rec_history_domain for row in right if row.task == "recommendation"}
    left_user = {row.family_node for row in left if row.task.startswith("user_")}
    right_user = {row.family_node for row in right if row.task.startswith("user_")}
    return {
        "candidate_rows": len(left),
        "reference_rows": len(right),
        "recommendation_group_id_overlap": len(left_groups & right_groups),
        "recommendation_history_domain_overlap": len(left_hdom & right_hdom),
        "user_family_overlap": len(left_user & right_user),
        "exact_full_row_overlap": len(left_exact & right_exact),
        "canonical_prompt_overlap": len(left_prompt & right_prompt),
    }


def main() -> None:
    split = _load_split_module()
    report = {"candidate": str(MINI), "fixed_alpha_dev": _audit(split, MINI, DEV), "fixed_alpha_probe": _audit(split, MINI, PROBE)}
    report["validation_safe"] = all(
        all(value == 0 for key, value in audit.items() if key.endswith("_overlap"))
        for audit in (report["fixed_alpha_dev"], report["fixed_alpha_probe"])
    )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
