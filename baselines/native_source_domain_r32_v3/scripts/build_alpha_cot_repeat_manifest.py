#!/usr/bin/env python3
"""Build the immutable train98 recommendation-CoT repeat-count manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")
SOURCE = ROOT / "train.jsonl"
TARGET = ROOT / "alpha_cot_repeat_count_manifest_v1.json"
EXPECTED_COT_ROWS = 27_186
EXPECTED_GROUPS_WITH_COT = 13_698


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Missing source train98 JSONL: {SOURCE}")
    source_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    total_rows = 0
    cot_rows = 0
    with SOURCE.open("rb") as handle:
        for lineno, raw in enumerate(handle, start=1):
            source_digest.update(raw)
            total_rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid train98 JSON at line {lineno}: {error}") from error
            if row.get("source_segment") != "recommendation_cot":
                continue
            try:
                metadata = json.loads(row["aux_metadata_json"])
                group_id = metadata["recommendation_group_id"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SystemExit(f"Invalid recommendation_cot metadata at line {lineno}: {error}") from error
            if not isinstance(group_id, str) or not group_id:
                raise SystemExit(f"Empty recommendation_group_id at line {lineno}")
            counts[group_id] += 1
            cot_rows += 1
    if cot_rows != EXPECTED_COT_ROWS or len(counts) != EXPECTED_GROUPS_WITH_COT:
        raise SystemExit(
            f"Train98 CoT audit changed: rows={cot_rows} groups={len(counts)}; "
            f"expected rows={EXPECTED_COT_ROWS} groups={EXPECTED_GROUPS_WITH_COT}."
        )
    distribution = Counter(counts.values())
    payload = {
        "kind": "alpha_cot_repeat_count_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_train_jsonl": str(SOURCE),
        "source_train_sha256": source_digest.hexdigest(),
        "train_rows": total_rows,
        "groups_with_cot": len(counts),
        "recommendation_cot_rows": cot_rows,
        "n_distribution_groups": {str(key): int(value) for key, value in sorted(distribution.items())},
        "n_distribution_rows": {str(key): int(key * value) for key, value in sorted(distribution.items())},
        "group_cot_counts": {key: int(counts[key]) for key in sorted(counts)},
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(TARGET), **{key: payload[key] for key in ("groups_with_cot", "recommendation_cot_rows", "source_train_sha256")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
