#!/usr/bin/env python3
"""Build the mini-cot-specific recommendation CoT repeat manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SOURCE = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/onereason_alpha_mini_v1.jsonl")
TARGET = SOURCE.parent / "mini_cot_repeat_count_manifest_v1.json"


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"missing source dataset: {SOURCE}")
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    total = recommendation = cot = nocot = 0
    with SOURCE.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            total += 1
            try:
                row = json.loads(raw)
                segment = row.get("source_segment")
                if row.get("data_source") == "recommend":
                    recommendation += 1
                if segment == "recommendation_nocot":
                    nocot += 1
                if segment != "recommendation_cot":
                    continue
                metadata = json.loads(row["aux_metadata_json"])
                group_id = metadata.get("recommendation_group_id")
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SystemExit(f"invalid recommendation metadata at line {line_no}: {error}") from error
            if not isinstance(group_id, str) or not group_id:
                raise SystemExit(f"missing recommendation_group_id at line {line_no}")
            counts[group_id] += 1
            cot += 1
    if not total or not cot or not counts:
        raise SystemExit("mini dataset has no valid recommendation CoT rows")
    if sum(counts.values()) != cot:
        raise SystemExit("CoT group row conservation failed")
    distribution = Counter(counts.values())
    payload = {
        "kind": "mini_cot_repeat_count_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "onereason_alpha_mini_v1",
        "source_train_jsonl": str(SOURCE),
        "source_train_sha256": digest.hexdigest(),
        "total_rows": total,
        "recommendation_rows": recommendation,
        "recommendation_cot_rows": cot,
        "recommendation_nocot_rows": nocot,
        "groups_with_cot": len(counts),
        "n_distribution_groups": {str(k): int(v) for k, v in sorted(distribution.items())},
        "n_distribution_rows": {str(k): int(k * v) for k, v in sorted(distribution.items())},
        "group_cot_counts": {key: int(counts[key]) for key in sorted(counts)},
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "source_train_sha256", "total_rows", "recommendation_rows", "recommendation_cot_rows",
        "recommendation_nocot_rows", "groups_with_cot", "n_distribution_groups",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
