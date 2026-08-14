#!/usr/bin/env python3
"""Read-only integrity audit for a completed alpha-jiankong split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def load_split_module():
    path = Path(__file__).with_name("split_alpha_jiankong_validation.py")
    spec = importlib.util.spec_from_file_location("alpha_split", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_counter(path: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line in handle:
            result[hashlib.sha256(line.rstrip(b"\n")).hexdigest()] += 1
    return result


def validate_rows(path: Path, required: set[str], rec_fields: set[str]) -> dict[str, int]:
    checked = recommendation_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise RuntimeError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            checked += 1
            if row["data_source"] == "recommend":
                metadata = json.loads(row["aux_metadata_json"])
                missing_metadata = rec_fields - set(metadata)
                if missing_metadata:
                    raise RuntimeError(f"{path}:{line_number} missing recommendation metadata: {sorted(missing_metadata)}")
                recommendation_rows += 1
    return {"rows": checked, "recommendation_rows": recommendation_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    module = load_split_module()
    train_path, dev_path = args.split_dir / "train.jsonl", args.split_dir / "dev.jsonl"
    required, rec_fields = set(module.REQUIRED_FIELDS), set(module.REC_METADATA_FIELDS)
    train_validation = validate_rows(train_path, required, rec_fields)
    dev_validation = validate_rows(dev_path, required, rec_fields)
    source_hashes, train_hashes, dev_hashes = raw_counter(args.source), raw_counter(train_path), raw_counter(dev_path)
    source_rows, _ = module.load_rows(args.source)
    train_rows, _ = module.load_rows(train_path)
    dev_rows, _ = module.load_rows(dev_path)
    rec_groups = lambda values: {row.rec_group_id for row in values if row.task == "recommendation"}
    rec_hdom = lambda values: {row.rec_history_domain for row in values if row.task == "recommendation"}
    prompts = lambda values: {row.prompt for row in values}
    users = lambda values: {row.family_node for row in values if row.task.startswith("user_")}
    report = {
        "source_rows": len(source_rows),
        "train": train_validation,
        "dev": dev_validation,
        "raw_multiset_conserved": source_hashes == train_hashes + dev_hashes,
        "exact_cross_side_overlap": sum((train_hashes & dev_hashes).values()),
        "recommendation_group_overlap": len(rec_groups(train_rows) & rec_groups(dev_rows)),
        "recommendation_history_domain_overlap": len(rec_hdom(train_rows) & rec_hdom(dev_rows)),
        "user_family_overlap": len(users(train_rows) & users(dev_rows)),
        "canonical_prompt_overlap": len(prompts(train_rows) & prompts(dev_rows)),
    }
    if not report["raw_multiset_conserved"] or any(
        report[key] != 0
        for key in (
            "exact_cross_side_overlap", "recommendation_group_overlap", "recommendation_history_domain_overlap",
            "user_family_overlap", "canonical_prompt_overlap",
        )
    ):
        raise RuntimeError(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
