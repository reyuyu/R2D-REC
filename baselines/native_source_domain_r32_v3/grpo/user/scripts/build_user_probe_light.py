#!/usr/bin/env python3
"""Build the metadata-only six-prompt fixed probe for formal User GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEED = 20260820
EXPECTED_SOURCE_SHA = "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(row: dict, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{row['sample_id']}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _third_pick(rows: list[dict], third: int, salt: str) -> dict:
    ordered = sorted(rows, key=lambda row: (int(row["prompt_token_count"]), row["sample_id"]))
    start = third * len(ordered) // 3
    stop = (third + 1) * len(ordered) // 3
    cell = ordered[start:stop]
    if not cell:
        raise ValueError(f"empty metadata third for {salt}")
    return min(cell, key=lambda row: stable_key(row, salt))


def select_light_probe(rows: list[dict]) -> tuple[list[dict], dict]:
    action = [row for row in rows if row["route"] == "action"]
    chain = [row for row in rows if row["route"] == "chain"]
    if len(action) != 12 or len(chain) != 8:
        raise ValueError("source probe must contain 12 Action and 8 Chain rows")
    action_ordered = sorted(
        action,
        key=lambda row: (int(row["gold_sid_count"]), int(row["prompt_token_count"]), row["sample_id"]),
    )
    action_selected = []
    for third, label in enumerate(("short_few", "medium", "long_many")):
        start = third * len(action_ordered) // 3
        stop = (third + 1) * len(action_ordered) // 3
        action_selected.append(min(action_ordered[start:stop], key=lambda row: stable_key(row, f"action:{label}")))

    by_event = {event_count: [row for row in chain if int(row["gold_event_count"]) == event_count] for event_count in (2, 3, 4)}
    if any(not values for values in by_event.values()):
        raise ValueError("light Chain probe requires event-count 2, 3, and 4 source rows")
    chain_selected = [
        min(by_event[2], key=lambda row: (int(row["prompt_token_count"]), stable_key(row, "chain:short:2"))),
        min(by_event[3], key=lambda row: (abs(int(row["prompt_token_count"]) - sorted(int(item["prompt_token_count"]) for item in chain)[len(chain) // 2]), stable_key(row, "chain:middle:3"))),
        max(by_event[4], key=lambda row: (int(row["prompt_token_count"]), stable_key(row, "chain:long:4"))),
    ]
    selected = action_selected + chain_selected
    if len({row["sample_id"] for row in selected}) != 6:
        raise AssertionError("light probe selection contains duplicate samples")
    audit = {
        "selection_uses_rewards": False,
        "selection_fields": ["route", "gold_sid_count", "gold_event_count", "prompt_token_count", "sample_id"],
        "action_strata": ["short/few Gold SID", "medium", "long/many Gold SID"],
        "chain_strata": ["short 2-event", "middle 3-event", "long 4-event"],
        "action": [
            {"sample_id": row["sample_id"], "gold_sid_count": row["gold_sid_count"], "prompt_token_count": row["prompt_token_count"]}
            for row in action_selected
        ],
        "chain": [
            {"sample_id": row["sample_id"], "gold_event_count": row["gold_event_count"], "prompt_token_count": row["prompt_token_count"]}
            for row in chain_selected
        ],
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.source) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("frozen probe_v1 SHA mismatch")
    selected, audit = select_light_probe(read_jsonl(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "contract_version": "gr_user_probe_light_v1",
        "seed": SEED,
        "source": str(args.source),
        "source_sha256": EXPECTED_SOURCE_SHA,
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "counts": {"total": 6, "action": 3, "chain": 3},
        "sample_ids": [row["sample_id"] for row in selected],
        "selection_audit": audit,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
