#!/usr/bin/env python3
"""Fail-closed overlap audit: full alpha_mini_v1 against its filtered dev/probe."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("/data/baselines/native_source_domain_r32_v3")
TRAIN = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/onereason_alpha_mini_v1.jsonl")
ROOT_OUT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1_validation_filtered_v1")
DEV = ROOT_OUT / "dev.jsonl"
PROBE = ROOT_OUT / "probe_v1/probe.jsonl"
OUT = ROOT_OUT / "leakage_audit.json"


def main() -> None:
    spec = importlib.util.spec_from_file_location("alpha_validation_verify", ROOT / "scripts/split_alpha_jiankong_validation.py")
    assert spec and spec.loader
    split = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = split
    spec.loader.exec_module(split)
    train, _ = split.load_rows(TRAIN)
    def audit(reference):
        right, _ = split.load_rows(reference)
        left_exact, right_exact = {x.row_hash for x in train}, {x.row_hash for x in right}
        left_prompt, right_prompt = {x.prompt for x in train}, {x.prompt for x in right}
        left_groups, right_groups = {x.rec_group_id for x in train if x.task == "recommendation"}, {x.rec_group_id for x in right if x.task == "recommendation"}
        left_hdom, right_hdom = {x.rec_history_domain for x in train if x.task == "recommendation"}, {x.rec_history_domain for x in right if x.task == "recommendation"}
        left_user, right_user = {x.family_node for x in train if x.task.startswith("user_")}, {x.family_node for x in right if x.task.startswith("user_")}
        return {"reference_rows": len(right), "recommendation_group_id_overlap": len(left_groups & right_groups), "recommendation_history_domain_overlap": len(left_hdom & right_hdom), "user_family_overlap": len(left_user & right_user), "exact_full_row_overlap": len(left_exact & right_exact), "canonical_prompt_overlap": len(left_prompt & right_prompt)}
    report = {"train_rows": len(train), "dev": audit(DEV), "probe": audit(PROBE)}
    report["validation_safe"] = all(all(value == 0 for key, value in val.items() if key.endswith("_overlap")) for val in (report["dev"], report["probe"]))
    if not report["validation_safe"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
