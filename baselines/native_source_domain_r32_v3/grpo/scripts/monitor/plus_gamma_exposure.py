"""Build and query a compact Plus-Gamma recommendation exposure index."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable


GROUP_ID_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_SOURCES = (
    ("epoch1", Path("/data/lf_data_versions/alltrain/plus_gamma_v1/onereason_plus_gamma_epoch1.jsonl")),
    ("epoch2", Path("/data/lf_data_versions/alltrain/plus_gamma_v1/onereason_plus_gamma_epoch2.jsonl")),
)
DEFAULT_INDEX = Path("/data/GRPO/cache/plus_gamma_v1_exposure_index.json")


def build_index(sources: Iterable[tuple[str, Path]] = DEFAULT_SOURCES) -> dict[str, Any]:
    """Index exact recommendation group IDs without retaining prompts or targets."""
    membership: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows: dict[str, int] = {}
    source_unique: dict[str, int] = {}
    source_paths: dict[str, str] = {}
    for epoch, path in sources:
        counts: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("source_segment") != "recommendation_cot":
                    continue
                metadata = json.loads(row.get("aux_metadata_json") or "{}")
                group_id = metadata.get("recommendation_group_id")
                if isinstance(group_id, str) and GROUP_ID_RE.fullmatch(group_id):
                    counts[group_id] += 1
        for group_id, count in counts.items():
            membership[group_id][epoch] += count
        source_rows[epoch] = sum(counts.values())
        source_unique[epoch] = len(counts)
        source_paths[epoch] = str(path)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_key": "aux_metadata_json.recommendation_group_id",
        "match_method": "exact equality; no fuzzy matching",
        "sources": source_paths,
        "source_recommendation_cot_rows": source_rows,
        "source_unique_group_ids": source_unique,
        "unique_group_ids": len(membership),
        "groups": {
            group_id: {"epochs": sorted(counts), "row_count": sum(counts.values())}
            for group_id, counts in sorted(membership.items())
        },
    }


def load_index(path: Path = DEFAULT_INDEX) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload.get("groups"), dict) else None


def exposure(group_id: Any, index: dict[str, Any] | None) -> dict[str, Any]:
    """Return an explicit three-state result so unavailable data is never called unseen."""
    if not isinstance(group_id, str) or GROUP_ID_RE.fullmatch(group_id) is None:
        return {"status": "unknown", "seen": None, "epochs": [], "row_count": None,
                "match_key": "recommendation_group_id"}
    if index is None:
        return {"status": "unknown", "seen": None, "epochs": [], "row_count": None,
                "match_key": "recommendation_group_id"}
    matched = index["groups"].get(group_id)
    if matched is None:
        return {"status": "unseen", "seen": False, "epochs": [], "row_count": 0,
                "match_key": "recommendation_group_id"}
    return {"status": "seen", "seen": True, "epochs": list(matched["epochs"]),
            "row_count": int(matched["row_count"]), "match_key": "recommendation_group_id"}


def annotate(rows: Iterable[dict[str, Any]], index: dict[str, Any] | None) -> None:
    """Annotate sample/group records and their candidate children in place."""
    for row in rows:
        group_id = row.get("recommendation_group_id") or row.get("group_id")
        value = exposure(group_id, index)
        row.setdefault("plus_gamma_exposure", value)
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate_id = candidate.get("recommendation_group_id") or candidate.get("group_id") or group_id
                    candidate.setdefault("plus_gamma_exposure", exposure(candidate_id, index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    payload = build_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "source_recommendation_cot_rows", "source_unique_group_ids", "unique_group_ids"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
