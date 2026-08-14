#!/usr/bin/env python3
"""Build recommendation mini_v2 by restoring selected NoThink rows to CoT."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")


def read_jsonl(path: Path, recommendation_only: bool = True) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != SCHEMA:
                raise ValueError(f"Unexpected schema at {path}:{line_no}")
            if recommendation_only and row["data_source"] != "recommend":
                raise ValueError(f"Unexpected data_source at {path}:{line_no}")
            if not recommendation_only and row["data_source"] != "recommend":
                continue
            if row["data_source"] != "recommend":
                raise ValueError(f"Unexpected data_source at {path}:{line_no}")
            metadata = json.loads(row["aux_metadata_json"])
            required = {
                "recommendation_group_id", "recommendation_group_size",
                "recommendation_current_gold_sid", "recommendation_all_gold_sids",
            }
            if set(metadata) != required:
                raise ValueError(f"Incomplete recommendation metadata at {path}:{line_no}")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_from_sid(sid: str) -> str:
    match = DOMAIN_RE.search(sid)
    if not match:
        raise ValueError(f"Cannot infer recommendation domain from SID: {sid}")
    return match.group(1)


def cot_body(output: str) -> str:
    marker = "</think>"
    if not output.startswith("<think>") or marker not in output:
        raise ValueError("COT output has no valid think span")
    return output[: output.index(marker)]


def stable_key(group_id: str, sid: str, row_index: int) -> str:
    payload = f"{group_id}\0{sid}\0{row_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def largest_remainder_targets(counts: dict[str, int], total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    raw = {domain: counts[domain] * total / denominator for domain in counts}
    targets = {domain: int(raw[domain]) for domain in counts}
    remaining = total - sum(targets.values())
    order = sorted(counts, key=lambda domain: (raw[domain] - targets[domain], domain), reverse=True)
    for domain in order[:remaining]:
        targets[domain] += 1
    return targets


def main() -> None:
    root = Path("/data/lf_data_versions/task_pools")
    selected_path = root / "懂推荐" / "alpha_mini" / "recommendation_multipositive_video_top550_other_domains_all.jsonl"
    mother_path = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
    output_dir = root / "懂推荐" / "mini_v2"
    output_path = output_dir / "recommendation_mini_v2.jsonl"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    selected = read_jsonl(selected_path)
    mother = read_jsonl(mother_path, recommendation_only=False)

    # The mother BETA rows provide one shared COT reasoning body per group.
    group_bodies: dict[str, set[str]] = defaultdict(set)
    for row in mother:
        if row["source_segment"] != "recommendation_cot":
            continue
        metadata = json.loads(row["aux_metadata_json"])
        group_bodies[metadata["recommendation_group_id"]].add(cot_body(row["output"]))

    body_variants = {group: len(bodies) for group, bodies in group_bodies.items() if len(bodies) != 1}
    if body_variants:
        raise ValueError(f"COT body is not unique for {len(body_variants)} groups")

    original_nocot = [
        (index, row)
        for index, row in enumerate(selected)
        if row["source_segment"] == "recommendation_nocot"
    ]
    nocot_by_domain: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    eligible_by_domain: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    unrecoverable = []
    for index, row in original_nocot:
        metadata = json.loads(row["aux_metadata_json"])
        group = metadata["recommendation_group_id"]
        sid = metadata["recommendation_current_gold_sid"]
        domain = domain_from_sid(sid)
        nocot_by_domain[domain].append((index, row))
        if group_bodies.get(group):
            eligible_by_domain[domain].append((index, row))
        else:
            unrecoverable.append({"row_index": index, "group_id": group, "domain": domain})

    target_nocot = round(len(selected) * 0.30)
    convert_total = len(original_nocot) - target_nocot
    available_counts = {domain: len(rows) for domain, rows in eligible_by_domain.items()}
    conversion_targets = largest_remainder_targets(
        {domain: len(rows) for domain, rows in nocot_by_domain.items()}, convert_total
    )
    for domain, target in conversion_targets.items():
        if target > available_counts.get(domain, 0):
            raise ValueError(f"Not enough recoverable rows in {domain}: {target} > {available_counts.get(domain, 0)}")

    chosen = []
    for domain in sorted(nocot_by_domain):
        candidates = sorted(
            eligible_by_domain[domain],
            key=lambda item: stable_key(
                json.loads(item[1]["aux_metadata_json"])["recommendation_group_id"],
                json.loads(item[1]["aux_metadata_json"])["recommendation_current_gold_sid"],
                item[0],
            ),
        )
        chosen.extend(index for index, _ in candidates[: conversion_targets[domain]])
    chosen_set = set(chosen)
    if len(chosen_set) != convert_total:
        raise AssertionError("Unexpected conversion count")

    converted_by_domain = Counter()
    output_rows = []
    for index, row in enumerate(selected):
        if index not in chosen_set:
            output_rows.append(row)
            continue
        metadata = json.loads(row["aux_metadata_json"])
        group = metadata["recommendation_group_id"]
        body = next(iter(group_bodies[group]))
        marker = "</think>"
        tail_start = row["output"].find(marker)
        if tail_start < 0:
            raise ValueError(f"No NoThink closing marker at selected row {index}")
        updated = dict(row)
        updated["output"] = body + row["output"][tail_start:]
        updated["source_segment"] = "recommendation_cot"
        output_rows.append(updated)
        converted_by_domain[domain_from_sid(metadata["recommendation_current_gold_sid"])] += 1

    counts = Counter(row["source_segment"] for row in output_rows)
    by_domain = Counter()
    for row in output_rows:
        metadata = json.loads(row["aux_metadata_json"])
        by_domain[domain_from_sid(metadata["recommendation_current_gold_sid"])] += 1
    output_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    with output_path.open("x", encoding="utf-8") as target:
        for row in output_rows:
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            target.write(payload)
            digest.update(payload.encode("utf-8"))

    manifest = {
        "kind": "unregistered_task_pool",
        "name": "mini_v2",
        "task": "recommendation",
        "source_dataset": str(selected_path),
        "cot_mother_dataset": str(mother_path),
        "selection_rule": {
            "input_rows_preserved": True,
            "target_nothink_ratio": 0.30,
            "target_nothink_rows": target_nocot,
            "conversion_rule": "replace eligible NoThink output think span with the unique group-level COT body from the BETA mother; preserve the original answer tail and metadata",
            "domain_balancing": "convert NoThink rows in proportion to each domain's original NoThink count using largest remainder",
            "deterministic_order": "SHA256(group_id + current_gold_sid + row_index)",
        },
        "counts": {
            "input_rows": len(selected),
            "output_rows": len(output_rows),
            "source_segment": dict(sorted(counts.items())),
            "by_domain": dict(sorted(by_domain.items())),
            "input_nocot_by_domain": dict(sorted({domain: len(rows) for domain, rows in nocot_by_domain.items()}.items())),
            "eligible_nocot_by_domain": dict(sorted(available_counts.items())),
            "converted_nocot_to_cot_by_domain": dict(sorted(converted_by_domain.items())),
            "unrecoverable_nocot_rows": len(unrecoverable),
        },
        "integrity": {
            "cot_body_variant_groups": 0,
            "metadata_preserved": True,
            "input_sha256": sha256(selected_path),
            "mother_sha256": sha256(mother_path),
            "output_sha256": digest.hexdigest(),
        },
        "unrecoverable_examples": unrecoverable[:20],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": output_path.name,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f'''# 懂推荐 mini_v2\n\n基于 Alpha-Mini 推荐任务池，将可恢复的 NoThink 行按四域比例恢复为 CoT。\n\n- 输入行数：{len(selected):,}\n- 输出行数：{len(output_rows):,}\n- CoT：{counts["recommendation_cot"]:,}\n- NoThink：{counts["recommendation_nocot"]:,}（目标约 30%）\n- 不可恢复行：{len(unrecoverable):,}，保留为 NoThink\n\n恢复使用 BETA 母数据同 group 的唯一 COT body，保留原答案 tail、gold SID 和全部推荐多正 metadata。四域按原 NoThink 数量比例分配恢复数量。\n'''
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
