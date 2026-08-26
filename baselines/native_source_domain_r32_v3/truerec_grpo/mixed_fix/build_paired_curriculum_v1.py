"""Pure-data helpers for fail-closed Mixed-Fix paired Curriculum2048."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


IMMUTABLE_FIELDS = (
    "recommendation_group_id", "stage", "stage_index", "epoch1_index",
    "target_domain", "fixed_domain_token", "K_A", "K_AB", "K_ABC",
    "hierarchy_class", "all_gold_abc",
)


def build_paired_record(
    source: dict[str, Any], user_content_think: str, teacher_cot: str,
    no_think_context_token_count: int, think_context_token_count: int,
) -> dict[str, Any]:
    result = {field: source[field] for field in IMMUTABLE_FIELDS}
    result.update({
        "system": source["system"],
        "user_content_nothink": source["user_content_nothink"],
        "user_content_think": user_content_think,
        "teacher_cot": teacher_cot,
        "all_gold_sids": source["all_gold_sids"],
        "history_sids": source["history_sids"],
        "no_think_context_token_count": no_think_context_token_count,
        "think_context_token_count": think_context_token_count,
        "think_minus_nothink_token_count": think_context_token_count - no_think_context_token_count,
    })
    return result


def audit_identity(
    source_records: list[dict[str, Any]], paired_records: list[dict[str, Any]], epoch1_order: list[str]
) -> None:
    source_ids = [str(row["recommendation_group_id"]) for row in source_records]
    paired_ids = [str(row["recommendation_group_id"]) for row in paired_records]
    if len(source_ids) != len(set(source_ids)) or len(paired_ids) != len(set(paired_ids)):
        raise ValueError("paired curriculum group IDs must be unique")
    if source_ids != paired_ids or source_ids != list(epoch1_order):
        raise ValueError("paired curriculum identity/order changed")
    for source, paired in zip(source_records, paired_records):
        if any(source[field] != paired[field] for field in IMMUTABLE_FIELDS):
            raise ValueError("paired curriculum immutable metadata changed")


def assert_no_context_overflow(records: Iterable[dict[str, Any]], max_positions: int) -> None:
    overflow = [
        row["recommendation_group_id"] for row in records
        if int(row["think_context_token_count"]) + 3 > max_positions
    ]
    if overflow:
        raise ValueError(f"Think context requires truncation: {overflow[:8]}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
            handle.write(raw); digest.update(raw); count += 1
    return digest.hexdigest(), count


def publish_if_pass(
    passed: bool, output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[bool, str | None]:
    """Publish only a fully gated paired dataset and bind its SHA in manifest."""
    if not passed:
        return False, None
    records_path = output_dir / "paired_curriculum2048.jsonl"
    records_sha, count = write_jsonl(records_path, rows)
    if count != 2048 or len({row["recommendation_group_id"] for row in rows}) != 2048:
        records_path.unlink(missing_ok=True)
        raise ValueError("published paired identity count failed")
    payload = dict(manifest)
    payload["paired_records"] = {
        "path": str(records_path), "sha256": records_sha, "count": count,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return True, records_sha
