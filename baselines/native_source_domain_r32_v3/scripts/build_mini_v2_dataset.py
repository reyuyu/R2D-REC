#!/usr/bin/env python3
"""Compose the mini_v2 training dataset from atomic task-pool files.

The aggregate user/material JSONL files are deliberately not re-ingested: their
rows are already represented by the atomic split files and would double count.
Rows are copied as parsed JSON with no field/value transformation.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


NAME = "mini_v2"
ROOT = Path("/data/lf_data_versions/alltrain")
OUT = ROOT / NAME
OUTPUT = OUT / "onereason_mini_v2.jsonl"
MANIFEST = OUT / "manifest.json"
REGISTRY = OUT / "dataset_info.json"

REQUIRED = (
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
)
REC_KEYS = (
    "recommendation_group_id", "recommendation_group_size",
    "recommendation_current_gold_sid", "recommendation_all_gold_sids",
)

INPUTS = [
    ("user", Path("/data/lf_data_versions/task_pools/懂用户/mini_v2/user_action.jsonl")),
    ("user", Path("/data/lf_data_versions/task_pools/懂用户/mini_v2/user_chain_cot.jsonl")),
    ("user", Path("/data/lf_data_versions/task_pools/懂用户/mini_v2/user_chain_nocot.jsonl")),
    ("recommendation", Path("/data/lf_data_versions/task_pools/懂推荐/mini_v2/recommendation_mini_v2.jsonl")),
    ("material", Path("/data/lf_data_versions/task_pools/懂物料/alpha_mini/material_sample.jsonl")),
    ("material", Path("/data/lf_data_versions/task_pools/懂物料/alpha_mini/sid_bucket_canonical_no_think.jsonl")),
    ("material", Path("/data/lf_data_versions/task_pools/懂物料/alpha_mini/sid_bucket_reverse.jsonl")),
]

AGGREGATES = [
    Path("/data/lf_data_versions/task_pools/懂用户/mini_v2/understand_user_mini_v2.jsonl"),
    Path("/data/lf_data_versions/task_pools/懂物料/alpha_mini/material_alpha_mini.jsonl"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_file(path: Path, role: str, duplicate_counts: collections.Counter[bytes]):
    rows = []
    stats = {
        "rows": 0,
        "json_parse_errors": 0,
        "missing_required_fields": 0,
        "aux_empty_nonrecommendation": 0,
        "aux_json_parse_errors": 0,
        "recommendation_metadata_missing": 0,
        "recommendation_metadata_parse_errors": 0,
        "recommendation_metadata_key_errors": 0,
    }
    data_sources = collections.Counter()
    source_segments = collections.Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except Exception:
                stats["json_parse_errors"] += 1
                continue
            if not isinstance(row, dict) or any(key not in row for key in REQUIRED):
                stats["missing_required_fields"] += 1
                continue
            data_source = str(row["data_source"])
            segment = str(row["source_segment"])
            data_sources[data_source] += 1
            source_segments[segment] += 1
            aux = row["aux_metadata_json"]
            if data_source == "recommend":
                if not isinstance(aux, str) or not aux:
                    stats["recommendation_metadata_missing"] += 1
                else:
                    try:
                        parsed = json.loads(aux)
                    except Exception:
                        stats["recommendation_metadata_parse_errors"] += 1
                    else:
                        if not isinstance(parsed, dict) or any(key not in parsed for key in REC_KEYS):
                            stats["recommendation_metadata_key_errors"] += 1
            elif aux in ("", None):
                # Existing non-recommendation schema uses an empty metadata field.
                # Preserve it byte-for-byte; do not turn it into "{}".
                stats["aux_empty_nonrecommendation"] += 1
            else:
                try:
                    json.loads(aux)
                except Exception:
                    stats["aux_json_parse_errors"] += 1
            rows.append(row)
            duplicate_counts[canonical(row)] += 1
    return rows, stats, data_sources, source_segments


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing dataset directory: {OUT}")
    for _, path in INPUTS:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    OUT.mkdir(parents=True)
    all_rows = []
    file_reports = []
    all_ds = collections.Counter()
    all_segments = collections.Counter()
    duplicates = collections.Counter()
    aggregate_reports = []
    for role, path in INPUTS:
        rows, stats, ds, segments = read_file(path, role, duplicates)
        if any(stats[key] for key in ("json_parse_errors", "missing_required_fields", "aux_json_parse_errors", "recommendation_metadata_missing", "recommendation_metadata_parse_errors", "recommendation_metadata_key_errors")):
            raise SystemExit(f"input validation failed for {path}: {stats}")
        all_rows.extend(rows)
        all_ds.update(ds)
        all_segments.update(segments)
        file_reports.append({
            "role": role,
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256_file(path),
            "stats": stats,
            "data_source_counts": dict(ds),
            "source_segment_counts": dict(segments),
        })
    for path in AGGREGATES:
        aggregate_reports.append({"path": str(path), "sha256": sha256_file(path), "not_reingested": True})
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    duplicate_rows = sum(count - 1 for count in duplicates.values() if count > 1)
    duplicate_groups = sum(1 for count in duplicates.values() if count > 1)
    manifest = {
        "dataset_name": NAME,
        "dataset_key": "onereason_mini_v2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_jsonl": str(OUTPUT),
        "input_files": file_reports,
        "aggregate_files_not_reingested": aggregate_reports,
        "task_pool_roots": {
            "user": "/data/lf_data_versions/task_pools/懂用户/mini_v2",
            "recommendation": "/data/lf_data_versions/task_pools/懂推荐/mini_v2",
            "material": "/data/lf_data_versions/task_pools/懂物料/alpha_mini",
        },
        "counts": {
            "user": sum(x["rows"] for x in file_reports if x["role"] == "user"),
            "recommendation": sum(x["rows"] for x in file_reports if x["role"] == "recommendation"),
            "material": sum(x["rows"] for x in file_reports if x["role"] == "material"),
            "total": len(all_rows),
        },
        "data_source_counts": dict(all_ds),
        "source_segment_counts": dict(all_segments),
        "duplicate_json_rows_not_removed": duplicate_rows,
        "duplicate_json_groups": duplicate_groups,
        "validation": {
            "required_fields": list(REQUIRED),
            "all_rows_json_valid": True,
            "all_required_fields_present": True,
            "recommendation_aux_metadata_parseable": True,
            "recommendation_aux_metadata_required_keys": list(REC_KEYS),
            "nonrecommendation_empty_aux_metadata_preserved": True,
            "nonrecommendation_empty_aux_metadata_count": sum(x["stats"]["aux_empty_nonrecommendation"] for x in file_reports),
            "count_conservation": len(all_rows) == sum(x["rows"] for x in file_reports),
            "output_sha256": sha256_file(OUTPUT),
        },
        "composition_note": "Atomic split files are used. Aggregate user/material files are recorded but not re-ingested to prevent double counting.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REGISTRY.write_text(json.dumps({"onereason_mini_v2": {"file_name": OUTPUT.name, "formatting": "alpaca", "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"}}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# mini_v2\n\n由懂用户 mini_v2、懂推荐 mini_v2、懂物料 alpha_mini 的原子 JSONL 组合而成。\n\n合并文件和逐任务统计见 `manifest.json`。原子文件逐行复制，未修改样本字段或推荐多正元数据。\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
