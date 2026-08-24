"""Attach exact source bridges to the immutable Boundary expanded rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DOMAIN = {"video": "<|video_begin|>", "prod": "<|prod_begin|>", "ad": "<|ad_begin|>", "living": "<|living_begin|>"}
CLOSE = "</think>"


def metadata(row: dict) -> dict:
    return json.loads(row["aux_metadata_json"]) if "aux_metadata_json" in row else row


def source_contract(row: dict) -> tuple[str, str, str] | None:
    segment = str(row.get("source_segment", "")).lower()
    if not (row.get("data_source") == "recommend" and "cot" in segment and "nocot" not in segment):
        return None
    meta = metadata(row)
    group = str(meta["recommendation_group_id"])
    golds = meta["recommendation_all_gold_sids"]
    if isinstance(golds, str):
        golds = json.loads(golds)
    domains = [key for key, token in DOMAIN.items() if all(str(gold).startswith(token) for gold in golds)]
    if len(domains) != 1:
        raise ValueError(f"group {group}: cannot infer one domain")
    domain = domains[0]
    response = row.get("output", row.get("response", row.get("completion")))
    if not isinstance(response, str) or response.count(CLOSE) != 1:
        raise ValueError(f"group {group}: malformed closure")
    start = response.index(CLOSE) + len(CLOSE)
    marker = DOMAIN[domain]
    if response.count(marker, start) != 1:
        raise ValueError(f"group {group}: domain marker is not unique after closure")
    end = response.index(marker, start)
    bridge = response[start:end]
    if not bridge:
        raise ValueError(f"group {group}: empty bridge")
    return group, domain, bridge


def build(source: Path, boundary_rows: Path, output: Path, stats_output: Path) -> dict:
    bridges: dict[str, tuple[str, str]] = {}
    counters = Counter()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            try:
                contract = source_contract(row)
            except Exception as exc:
                raise ValueError(f"source line {line_number}: {exc}") from exc
            if contract is None:
                continue
            group, domain, bridge = contract
            counters["source_rows"] += 1
            if group in bridges:
                if bridges[group] != (domain, bridge):
                    raise ValueError(f"group {group}: duplicate source bridge is not byte-identical")
                counters["duplicate_source_rows"] += 1
            else:
                bridges[group] = (domain, bridge)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups, domains, bridge_hashes = set(), Counter(), Counter()
    with boundary_rows.open(encoding="utf-8") as source_rows, output.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source_rows, 1):
            if not line.strip():
                continue
            row = json.loads(line); group = str(row["boundary_group_id"])
            if group not in bridges:
                raise ValueError(f"boundary line {line_number}: missing exact bridge for {group}")
            domain, bridge = bridges[group]
            if row["target_domain"] != domain:
                raise ValueError(f"boundary line {line_number}: target domain mismatch")
            response = row["original_response"]
            start = response.index(CLOSE) + len(CLOSE); end = response.index(DOMAIN[domain], start)
            if response[start:end] != bridge:
                raise ValueError(f"boundary line {line_number}: source/boundary bridge mismatch")
            digest = hashlib.sha256(bridge.encode("utf-8")).hexdigest()
            row["exact_bridge"] = bridge; row["bridge_sha256"] = digest
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            counters["expanded_paths"] += 1; groups.add(group); domains[domain] += 1; bridge_hashes[digest] += 1
    if set(bridges) != groups:
        raise ValueError(f"group set mismatch: source={len(bridges)} boundary={len(groups)}")
    stats = {
        "source": str(source), "boundary_rows": str(boundary_rows),
        "source_recommendation_cot_rows": counters["source_rows"],
        "byte_identical_duplicates": counters["duplicate_source_rows"],
        "original_groups": len(groups), "expanded_paths": counters["expanded_paths"],
        "per_domain_expanded_paths": dict(domains), "unique_bridges": len(bridge_hashes),
        "exact_bridge_extraction_pass": True, "duplicate_bridge_byte_parity_pass": True,
    }
    stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True); parser.add_argument("--boundary-rows", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--stats-output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(Path(value) for value in (args.source, args.boundary_rows, args.output, args.stats_output))), ensure_ascii=False, indent=2))
