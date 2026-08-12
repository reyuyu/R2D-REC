#!/usr/bin/env python3
"""Build a deterministic recommendation-heavy raw subset of alpha dev2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SEED = "20260812"
VERSION = "alpha_jiankong_dev_probe_v1"
DOMAINS = ("video", "prod", "ad", "living")


def score(group_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{VERSION}|{group_id}".encode("utf-8")).hexdigest()


def domain(row: dict) -> str:
    metadata = json.loads(row["aux_metadata_json"])
    gold = metadata["recommendation_current_gold_sid"]
    for value in DOMAINS:
        if gold.startswith(f"<|{value}_begin|>"):
            return value
    raise ValueError(f"Unknown recommendation target domain: {gold!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-groups", type=int, default=110)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing probe: {args.output}")
    groups: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    with args.dev.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            if row["data_source"] != "recommend":
                continue
            metadata = json.loads(row["aux_metadata_json"])
            groups[str(metadata["recommendation_group_id"])].append((line_number, raw, row))
    if args.target_groups < len(DOMAINS) or args.target_groups > len(groups):
        raise ValueError("target-groups must be between four and the number of dev recommendation groups.")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in groups.items():
        group_domains = {domain(row) for _, _, row in rows}
        if len(group_domains) != 1:
            raise ValueError(f"Group spans target domains: {group_id}")
        by_domain[next(iter(group_domains))].append(group_id)
    selected = {min(by_domain[value], key=score) for value in DOMAINS}
    for group_id in sorted(groups, key=score):
        if len(selected) >= args.target_groups:
            break
        selected.add(group_id)
    selected_rows = [entry for group_id in selected for entry in groups[group_id]]
    selected_rows.sort(key=lambda value: value[0])
    destination = args.output.with_name(args.output.name + ".tmp")
    destination.mkdir(parents=True)
    with (destination / "probe.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for _, raw, _ in selected_rows:
            handle.write(raw)
    route_counts = Counter(row["source_segment"] for _, _, row in selected_rows)
    domain_counts = Counter(domain(row) for _, _, row in selected_rows)
    group_sizes = Counter(len(groups[group_id]) for group_id in selected)
    if not all(domain_counts[value] for value in DOMAINS):
        raise RuntimeError("Probe misses a recommendation domain.")
    if not route_counts["recommendation_cot"] or not route_counts["recommendation_nocot"]:
        raise RuntimeError("Probe misses a recommendation route.")
    fingerprints = [hashlib.sha256(raw.encode("utf-8")).hexdigest() for _, raw, _ in selected_rows]
    manifest = {
        "name": VERSION,
        "source_dataset": "onereason_alpha_jiankong_dev2",
        "source_path": str(args.dev),
        "seed": int(SEED),
        "selection": "whole recommendation_group_id, SHA256-ranked; one mandatory group per domain then global ranking",
        "selected_groups": len(selected),
        "selected_group_ids": sorted(selected),
        "raw_rows": len(selected_rows),
        "raw_row_line_numbers": [line for line, _, _ in selected_rows],
        "raw_row_sha256": fingerprints,
        "probe_sha256": hashlib.sha256("\n".join(fingerprints).encode("ascii")).hexdigest(),
        "route_counts": dict(sorted(route_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "group_row_size_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "dataset_info.json").write_text(json.dumps({
        "onereason_alpha_jiankong_dev_probe_v1": {
            "file_name": "probe.jsonl", "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
        }
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destination.rename(args.output)
    print(json.dumps({"output": str(args.output), **{key: manifest[key] for key in ("selected_groups", "raw_rows", "route_counts", "domain_counts", "probe_sha256")}}, sort_keys=True))


if __name__ == "__main__":
    main()
