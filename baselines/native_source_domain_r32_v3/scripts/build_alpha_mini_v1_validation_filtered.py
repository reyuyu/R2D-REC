#!/usr/bin/env python3
"""Create Alpha dev/probe views that are leakage-free against full alpha_mini_v1."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
TRAIN = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/onereason_alpha_mini_v1.jsonl")
SOURCE_SPLIT = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")
SOURCE_DEV = SOURCE_SPLIT / "dev.jsonl"
SOURCE_PROBE = SOURCE_SPLIT / "probe_v1/probe.jsonl"
OUT_ROOT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1_validation_filtered_v1")
OUT_DEV = OUT_ROOT / "dev.jsonl"
OUT_PROBE_ROOT = OUT_ROOT / "probe_v1"
OUT_PROBE = OUT_PROBE_ROOT / "probe.jsonl"


def _split_module():
    path = ROOT / "scripts/split_alpha_jiankong_validation.py"
    spec = importlib.util.spec_from_file_location("alpha_validation_filter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _keys(rows):
    return {
        "exact": {row.row_hash for row in rows},
        "prompt": {row.prompt for row in rows},
        "rec_group": {row.rec_group_id for row in rows if row.task == "recommendation"},
        "rec_hdom": {row.rec_history_domain for row in rows if row.task == "recommendation"},
        "user_family": {row.family_node for row in rows if row.task.startswith("user_")},
    }


def _leak_causes(row, train_keys):
    causes = []
    if row.row_hash in train_keys["exact"]:
        causes.append("exact_full_row")
    if row.prompt in train_keys["prompt"]:
        causes.append("canonical_prompt")
    if row.task == "recommendation":
        if row.rec_group_id in train_keys["rec_group"]:
            causes.append("recommendation_group_id")
        if row.rec_history_domain in train_keys["rec_hdom"]:
            causes.append("recommendation_history_domain")
    if row.task.startswith("user_") and row.family_node in train_keys["user_family"]:
        causes.append("user_family")
    return causes


def _filter(source: Path, destination: Path, split, train_keys):
    rows, _ = split.load_rows(source)
    keep = set()
    removed = Counter()
    removed_by_segment = Counter()
    for row in rows:
        causes = _leak_causes(row, train_keys)
        if causes:
            removed.update(causes)
            removed_by_segment[row.source_segment] += 1
        else:
            keep.add(row.index)
    kept_by_segment = Counter()
    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for index, line in enumerate(reader):
            if index in keep:
                writer.write(line)
                kept_by_segment[str(json.loads(line)["source_segment"])] += 1
    return {
        "source": str(source), "output": str(destination), "source_rows": len(rows), "kept_rows": len(keep),
        "removed_rows": len(rows) - len(keep), "removed_trigger_counts_nonexclusive": dict(sorted(removed.items())),
        "removed_by_source_segment": dict(sorted(removed_by_segment.items())), "kept_by_source_segment": dict(sorted(kept_by_segment.items())),
    }


def _domain_rows(split, path: Path):
    rows, _ = split.load_rows(path)
    result = Counter(row.rec_domain for row in rows if row.task == "recommendation")
    if any(result[domain] == 0 for domain in ("video", "prod", "ad", "living")):
        raise RuntimeError(f"filtered dev lost a recommendation domain: {dict(result)}")
    return {domain: {"original": int(result[domain])} for domain in ("video", "prod", "ad", "living")}


def main() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite filtered validation view: {OUT_ROOT}")
    split = _split_module()
    train_rows, _ = split.load_rows(TRAIN)
    train_keys = _keys(train_rows)
    OUT_PROBE_ROOT.mkdir(parents=True)
    dev_report = _filter(SOURCE_DEV, OUT_DEV, split, train_keys)
    probe_report = _filter(SOURCE_PROBE, OUT_PROBE, split, train_keys)
    info = {
        "onereason_alpha_mini_v1_dev_filtered": {
            "file_name": "dev.jsonl", "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
        }
    }
    probe_info = {
        "onereason_alpha_mini_v1_probe_filtered": {
            "file_name": "probe.jsonl", "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
        }
    }
    (OUT_ROOT / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_PROBE_ROOT / "dataset_info.json").write_text(json.dumps(probe_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_audit = {"recommendation": {"domain_rows": _domain_rows(split, OUT_DEV)}}
    (OUT_ROOT / "split_audit.json").write_text(json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "kind": "alpha_mini_v1_filtered_validation_view",
        "training_dataset_unchanged": str(TRAIN),
        "rule": "delete only fixed validation rows that overlap alpha_mini_v1 on any required Alpha leakage key",
        "dev": dev_report,
        "probe": probe_report,
        "filtered_dev_domain_rows": split_audit["recommendation"]["domain_rows"],
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
