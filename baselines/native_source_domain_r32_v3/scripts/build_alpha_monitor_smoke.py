"""Build a fixed, unmodified alpha sample view for monitor throughput smoke."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


SOURCE = Path("/data/lf_data_versions/alltrain/alpha-jiankong/onereason_alpha_jiankong.jsonl")
DEST_DIR = Path("/data/lf_data_versions/smoke/alpha_monitor_v1")
DEST = DEST_DIR / "onereason_alpha_monitor_smoke.jsonl"
LIMITS = {
    "material": 2048,
    "recommendation_cot": 1024,
    "recommendation_nocot": 1024,
    "user_action": 2048,
    "user_chain_cot": 1024,
    "user_chain_nocot": 1024,
}


def bucket(row: dict) -> str | None:
    if row.get("data_source") in {"material_sample", "sid_bucket_canonical_no_think", "sid_bucket_reverse"}:
        return "material"
    return row.get("source_segment")


def main() -> None:
    chosen: dict[str, list[str]] = defaultdict(list)
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = bucket(row)
            if key in LIMITS and len(chosen[key]) < LIMITS[key]:
                # Preserve the source JSON object unchanged at the semantic field level.
                chosen[key].append(line)
            if all(len(chosen[key]) == limit for key, limit in LIMITS.items()):
                break
    missing = {key: (len(chosen[key]), limit) for key, limit in LIMITS.items() if len(chosen[key]) != limit}
    if missing:
        raise RuntimeError(f"Insufficient alpha smoke samples: {missing}")
    rows = [line for key in LIMITS for line in chosen[key]]
    random.Random(20260806).shuffle(rows)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", encoding="utf-8") as handle:
        handle.writelines(rows)
    (DEST_DIR / "dataset_info.json").write_text(
        json.dumps({
            "onereason_alpha_monitor_smoke": {
                "file_name": DEST.name,
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
            }
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DEST_DIR / "manifest.json").write_text(
        json.dumps({"kind": "temporary_smoke_view", "parent": str(SOURCE), "seed": 20260806, "limits": LIMITS, "records": len(rows)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: len(chosen[key]) for key in LIMITS} | {"total": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
