#!/usr/bin/env python3
"""Build recommendation mini_v3: restore every mini_v2 NoThink row to CoT.

This intentionally creates a task-pool version only.  It preserves the chosen
mini_v2 row order, answer tails and recommendation multi-positive metadata.
Legacy CoT rows are used only to supply the shared reasoning body, keyed by the
ordered history SID sequence plus the final target domain (hdom).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lf_data_versions/task_pools")
INPUT = ROOT / "懂推荐" / "mini_v2" / "recommendation_mini_v2.jsonl"
OUTPUT_DIR = ROOT / "懂推荐" / "mini_v3"
OUTPUT = OUTPUT_DIR / "recommendation_mini_v3_all_cot.jsonl"
LEGACY_COT_SOURCES = (
    Path("/data/lf_data_versions/alltrain/v1_thought_prompt_all/onereason_recommendation_cot.jsonl"),
    Path("/data/lf_data_versions/alltrain/v2_recommendation_dual/onereason_recommendation_cot_v2_dual.jsonl"),
    Path("/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl"),
    Path("/data/lf_data_versions/alltrain/v3_recommendation_multi_positive/onereason_recommendation_cot_v3_multi_positive.jsonl"),
    Path("/data/lf_data_versions/alltrain/BETA/onereason_recommendation_cot.jsonl"),
)
SCHEMA = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                yield line_no, json.loads(line)


def target_domain(row: dict) -> str:
    # Some legacy sources retain the current gold at top level rather than in
    # aux_metadata_json.  This deliberately mirrors the successful audit.
    try:
        metadata = json.loads(row.get("aux_metadata_json", ""))
        sid = metadata.get("recommendation_current_gold_sid")
    except (TypeError, json.JSONDecodeError):
        sid = row.get("recommendation_current_gold_sid")
    if not isinstance(sid, str):
        matches = list(SID_RE.finditer(str(row.get("output", ""))))
        sid = matches[-1].group(0) if matches else ""
    match = SID_RE.fullmatch(sid)
    if not match:
        raise ValueError(f"Cannot derive a valid target domain from {sid!r}")
    return match.group(1)


def hdom(row: dict) -> str:
    # Ordered history is deliberately retained: same inventory in another
    # chronology is not assumed to share reasoning.
    history_text = "\n".join(str(row.get(k, "")) for k in ("system", "instruction", "input", "history"))
    history = [match.group(0) for match in SID_RE.finditer(history_text)]
    if not history:
        raise ValueError("Cannot derive non-empty ordered SID history")
    data = json.dumps({"sid_history": history, "target_domain": target_domain(row)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cot_body(output: str) -> str:
    end = output.find("</think>")
    if not output.startswith("<think>") or end <= len("<think>"):
        raise ValueError("Missing or empty COT think span")
    return output[:end]


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing {OUTPUT_DIR}")
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    selected = []
    for line_no, row in rows(INPUT):
        if set(row) != SCHEMA or row["data_source"] != "recommend":
            raise ValueError(f"Unexpected mini_v2 schema/source at line {line_no}")
        metadata = json.loads(row["aux_metadata_json"])
        required = {"recommendation_group_id", "recommendation_group_size", "recommendation_current_gold_sid", "recommendation_all_gold_sids"}
        if set(metadata) != required:
            raise ValueError(f"Incomplete metadata at mini_v2 line {line_no}")
        selected.append(row)

    body_by_hdom: dict[str, set[str]] = defaultdict(set)
    source_matches = Counter()
    source_valid = Counter()
    for source_path in LEGACY_COT_SOURCES:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        for _, row in rows(source_path):
            try:
                body = cot_body(row["output"])
                key = hdom(row)
            except (ValueError, json.JSONDecodeError):
                continue
            source_valid[str(source_path)] += 1
            body_by_hdom[key].add(body)
            source_matches[str(source_path)] += 1

    nocot = [(i, row) for i, row in enumerate(selected) if row["source_segment"] == "recommendation_nocot"]
    unexpected = [row["source_segment"] for row in selected if row["source_segment"] not in {"recommendation_cot", "recommendation_nocot"}]
    if unexpected:
        raise ValueError(f"Unexpected source segments: {set(unexpected)}")
    missing, ambiguous, collision = [], [], []
    # A single current group must map to one hdom; and one hdom must not refer
    # to multiple current groups, otherwise legacy reconstruction is unsafe.
    group_to_key, key_to_groups = {}, defaultdict(set)
    for index, row in nocot:
        group = json.loads(row["aux_metadata_json"])["recommendation_group_id"]
        key = hdom(row)
        if group in group_to_key and group_to_key[group] != key:
            collision.append((index, group, "group_has_multiple_hdom_keys"))
        group_to_key[group] = key
        key_to_groups[key].add(group)
        bodies = body_by_hdom.get(key, set())
        if not bodies:
            missing.append((index, group))
        elif len(bodies) != 1:
            ambiguous.append((index, group, len(bodies)))
    for key, groups in key_to_groups.items():
        if len(groups) != 1:
            collision.append((-1, ",".join(sorted(groups)), "hdom_maps_multiple_current_groups"))
    if missing or ambiguous or collision:
        print(json.dumps({"debug_source_valid": source_valid, "debug_source_keys": len(body_by_hdom), "debug_desired_keys": len(key_to_groups)}, ensure_ascii=False))
        raise RuntimeError(
            f"Unsafe legacy recovery: missing={len(missing)} ambiguous={len(ambiguous)} collision={len(collision)}"
        )

    converted_by_domain = Counter()
    output_rows, digest = [], hashlib.sha256()
    unchanged_cot = 0
    for index, row in enumerate(selected):
        updated = row
        if row["source_segment"] == "recommendation_nocot":
            updated = dict(row)
            body = next(iter(body_by_hdom[hdom(row)]))
            close = row["output"].find("</think>")
            if close < 0:
                raise ValueError(f"NoThink response missing close marker at line {index + 1}")
            # Retain the direct answer tail byte-for-byte from mini_v2.
            updated["output"] = body + row["output"][close:]
            updated["source_segment"] = "recommendation_cot"
            converted_by_domain[target_domain(row)] += 1
        else:
            unchanged_cot += 1
        output_rows.append(updated)

    if len(output_rows) != len(selected):
        raise AssertionError("Row count changed")
    if any(row["source_segment"] != "recommendation_cot" for row in output_rows):
        raise AssertionError("NoThink rows remain")
    if any(not row["output"].startswith("<think>") or "</think>" not in row["output"] for row in output_rows):
        raise AssertionError("Output without a valid think wrapper")
    # Only output and source_segment may change, and only for former NoThink.
    for before, after in zip(selected, output_rows):
        if before["source_segment"] == "recommendation_cot":
            if before != after:
                raise AssertionError("Existing CoT row mutated")
        else:
            for key in SCHEMA - {"output", "source_segment"}:
                if before[key] != after[key]:
                    raise AssertionError(f"Unexpected mutation in {key}")
            close = before["output"].find("</think>")
            if after["output"].split("</think>", 1)[1] != before["output"][close + len("</think>"):]:
                raise AssertionError("Answer tail changed")

    OUTPUT_DIR.mkdir(parents=True)
    with OUTPUT.open("x", encoding="utf-8") as target:
        for row in output_rows:
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            target.write(payload)
            digest.update(payload.encode("utf-8"))
    by_domain = Counter(target_domain(row) for row in output_rows)
    manifest = {
        "kind": "unregistered_task_pool",
        "name": "mini_v3",
        "task": "recommendation",
        "source_dataset": str(INPUT),
        "legacy_cot_sources": [str(p) for p in LEGACY_COT_SOURCES],
        "recovery_rule": "Every mini_v2 NoThink row is converted by prepending the sole legacy COT body selected with ordered history SID sequence + final target domain; direct answer tail and recommendation metadata remain from mini_v2.",
        "counts": {
            "input_rows": len(selected), "output_rows": len(output_rows),
            "input_cot_rows_unchanged": unchanged_cot,
            "input_nocot_rows_converted": len(nocot),
            "output_cot_rows": len(output_rows), "output_nocot_rows": 0,
            "by_target_domain": dict(sorted(by_domain.items())),
            "converted_by_target_domain": dict(sorted(converted_by_domain.items())),
        },
        "integrity": {
            "all_recovery_keys_have_exactly_one_body": True,
            "current_group_hdom_collisions": 0,
            "metadata_preserved": True,
            "answer_tail_preserved_for_converted_rows": True,
            "input_sha256": sha256(INPUT),
            "output_sha256": digest.hexdigest(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": OUTPUT.name,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# 懂推荐 mini_v3\n\n"
        "基于 mini_v2 构建。所有 11,192 条样本均为 CoT；原有 CoT 不变，原 NoThink 的直接答案尾部与多正样本元数据不变，仅由经过唯一性审计的历史 CoT reasoning body 补全。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
