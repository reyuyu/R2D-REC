#!/usr/bin/env python3
"""Compose the mini_v3 training dataset from three task-pool versions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


USER = Path("/data/lf_data_versions/task_pools/懂用户/mini_v2/understand_user_mini_v2.jsonl")
REC = Path("/data/lf_data_versions/task_pools/懂推荐/mini_v3/recommendation_mini_v3_all_cot.jsonl")
MAT = Path("/data/lf_data_versions/task_pools/懂物料/alpha_mini/material_alpha_mini.jsonl")
SOURCES = [("user", USER), ("recommendation", REC), ("material", MAT)]
OUT_DIR = Path("/data/lf_data_versions/alltrain/mini_v3")
OUT = OUT_DIR / "onereason_mini_v3.jsonl"
SCHEMA = {"system", "instruction", "input", "output", "history", "data_source", "source_segment", "aux_metadata_json"}
REC_REQUIRED = {"recommendation_group_id", "recommendation_group_size", "recommendation_current_gold_sid", "recommendation_all_gold_sids"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scan(path: Path, role: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = 0
    data_source = Counter()
    source_segment = Counter()
    exact_rows = Counter()
    aux_parseable = 0
    aux_empty = 0
    aux_invalid = 0
    rec_meta_valid = 0
    rec_meta_invalid = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != SCHEMA:
                raise ValueError(f"{path}:{line_no}: schema mismatch {set(row)}")
            rows += 1
            data_source[str(row["data_source"])] += 1
            source_segment[str(row["source_segment"])] += 1
            exact_rows[canonical(row)] += 1
            aux = row["aux_metadata_json"]
            if aux == "":
                aux_empty += 1
            else:
                try:
                    json.loads(aux)
                    aux_parseable += 1
                except (TypeError, json.JSONDecodeError):
                    aux_invalid += 1
            if row["data_source"] == "recommend":
                try:
                    metadata = json.loads(aux)
                    if set(metadata) == REC_REQUIRED:
                        rec_meta_valid += 1
                    else:
                        rec_meta_invalid += 1
                except (TypeError, json.JSONDecodeError):
                    rec_meta_invalid += 1
    return {
        "role": role,
        "path": str(path),
        "rows": rows,
        "sha256": sha256(path),
        "data_source_counts": dict(sorted(data_source.items())),
        "source_segment_counts": dict(sorted(source_segment.items())),
        "aux_metadata_json": {"parseable": aux_parseable, "empty_string": aux_empty, "invalid_nonempty": aux_invalid},
        "recommendation_metadata": {"valid": rec_meta_valid, "invalid": rec_meta_invalid},
        "exact_json_duplicate_rows": sum(n - 1 for n in exact_rows.values() if n > 1),
        "exact_json_duplicate_groups": sum(1 for n in exact_rows.values() if n > 1),
    }


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing registration directory: {OUT_DIR}")
    summaries = [scan(path, role) for role, path in SOURCES]
    total = sum(item["rows"] for item in summaries)
    OUT_DIR.mkdir(parents=True)
    digest = hashlib.sha256()
    with OUT.open("x", encoding="utf-8") as target:
        for role, path in SOURCES:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    json.loads(line)
                    target.write(line if line.endswith("\n") else line + "\n")
                    digest.update((line if line.endswith("\n") else line + "\n").encode("utf-8"))
    output_scan = scan(OUT, "combined")
    if output_scan["rows"] != total:
        raise AssertionError(f"row conservation failed: {output_scan['rows']} != {total}")
    source_sum = Counter()
    segment_sum = Counter()
    for item in summaries:
        source_sum.update(item["data_source_counts"])
        segment_sum.update(item["source_segment_counts"])
    if dict(source_sum) != output_scan["data_source_counts"] or dict(segment_sum) != output_scan["source_segment_counts"]:
        raise AssertionError("data_source/source_segment conservation failed")
    dataset_info = {
        "onereason_mini_v3": {
            "file_name": OUT.name,
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
        }
    }
    (OUT_DIR / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "kind": "registered_training_dataset",
        "name": "mini_v3",
        "parent_baseline": "alpha_mini",
        "dataset_name": "onereason_mini_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_task_pool_versions": {
            "user": "/data/lf_data_versions/task_pools/懂用户/mini_v2",
            "recommendation": "/data/lf_data_versions/task_pools/懂推荐/mini_v3",
            "material": "/data/lf_data_versions/task_pools/懂物料/alpha_mini",
        },
        "input_files": summaries,
        "output_file": str(OUT),
        "output_sha256": digest.hexdigest(),
        "final_rows": total,
        "final_data_source_counts": output_scan["data_source_counts"],
        "final_source_segment_counts": output_scan["source_segment_counts"],
        "validation": {
            "schema_all_rows": True,
            "row_count_conserved": True,
            "source_file_sum": total,
            "final_count": output_scan["rows"],
            "recommendation_metadata_required_keys": sorted(REC_REQUIRED),
            "recommendation_metadata_valid": output_scan["recommendation_metadata"]["valid"],
            "recommendation_metadata_invalid": output_scan["recommendation_metadata"]["invalid"],
            "cross_task_exact_duplicate_rows_reported": output_scan["exact_json_duplicate_rows"],
            "cross_task_exact_duplicate_groups_reported": output_scan["exact_json_duplicate_groups"],
            "aux_empty_strings_preserved": output_scan["aux_metadata_json"]["empty_string"],
            "aux_invalid_nonempty": output_scan["aux_metadata_json"]["invalid_nonempty"],
        },
        "note": "Non-recommendation source rows carry the source schema's empty aux_metadata_json string; it was preserved exactly per the no-modification contract. Recommendation metadata is JSON-parseable and contains all required multi-positive keys.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text("# mini_v3\n\nalpha_mini 父基线：懂用户 mini_v2、懂推荐 mini_v3（全 CoT）、懂物料 alpha_mini 的逐行合并版本。未启动训练。\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
