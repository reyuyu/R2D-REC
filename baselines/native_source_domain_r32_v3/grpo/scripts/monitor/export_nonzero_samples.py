"""Incrementally archive Dual Beam8 samples with non-zero training signal."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_sources(dataset: Path) -> dict[str, dict[str, Any]]:
    sources = {}
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            group_id = row.get("recommendation_group_id")
            if group_id:
                sources[str(group_id)] = row
    return sources


def existing_keys(path: Path) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                key = json.loads(line).get("sample_key")
            except json.JSONDecodeError:
                continue
            if key:
                keys.add(str(key))
    return keys


def candidate_record(
    event: dict[str, Any],
    cot: dict[str, Any],
    cot_index: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_key": f"{event.get('rollout_fingerprint')}:{cot_index}",
        "experiment": "GR_REC_ThinkDualBeam8_v2",
        "step": event.get("step"),
        "rollout_id": event.get("rollout_id"),
        "rollout_fingerprint": event.get("rollout_fingerprint"),
        "recommendation_group_id": event.get("recommendation_group_id"),
        "target_domain": event.get("target_domain"),
        "cot_index": cot_index,
        "prompt": source.get("prompt"),
        "all_gold_sids": source.get("all_gold_sids", []),
        "cot_text": cot.get("cot_text"),
        "cot_length": cot.get("cot_length"),
        "cot_closed": cot.get("closed"),
        "cot_reward": cot.get("cot_reward"),
        "cot_advantage": (event.get("cot_advantages") or [None] * 4)[cot_index],
        "beam_candidate_ids": cot.get("beam_candidate_ids", []),
        "beam_sids": cot.get("beam_sids", []),
        "sid_rewards": cot.get("sid_rewards", []),
        "sid_advantages": cot.get("sid_advantages", []),
        "sid_reward_levels": cot.get("sid_reward_levels", []),
        "sid_population_std": cot.get("sid_population_std"),
        "sid_zero_std": cot.get("sid_zero_std"),
        "exact_count": cot.get("exact"),
        "ab_count": cot.get("ab"),
        "a_count": cot.get("a"),
        "invalid_count": cot.get("invalid"),
        "captured_at": event.get("timestamp"),
        "provenance": "training_capture_no_recalculation",
    }


def export_available(run_dir: Path, output_dir: Path) -> dict[str, int]:
    manifest = read_json(run_dir / "manifest.json", {})
    dataset = Path(manifest["dataset_path"]).expanduser().resolve()
    source_path = run_dir / "dual_beam8.jsonl"
    state_path = output_dir / ".export_state.json"
    nonzero_path = output_dir / "nonzero_signal_samples.jsonl"
    positive_path = output_dir / "positive_samples.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(dataset)
    state = read_json(state_path, {})
    offset = int(state.get("offset", 0))
    if not source_path.exists() or source_path.stat().st_size < offset:
        offset = 0
    nonzero_seen = existing_keys(nonzero_path)
    positive_seen = existing_keys(positive_path)
    added_nonzero = added_positive = 0

    with source_path.open("r", encoding="utf-8") as source, \
            nonzero_path.open("a", encoding="utf-8") as nonzero, \
            positive_path.open("a", encoding="utf-8") as positive:
        source.seek(offset)
        while True:
            line_offset = source.tell()
            line = source.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                source.seek(line_offset)
                break
            offset = source.tell()
            if event.get("type") != "dual_beam8" or not isinstance(event.get("cots"), list):
                continue
            group_id = str(event.get("recommendation_group_id") or "")
            source_row = sources.get(group_id, {})
            for cot_index, cot in enumerate(event["cots"]):
                sid_rewards = [float(value) for value in cot.get("sid_rewards", [])]
                cot_reward = float(cot.get("cot_reward", 0.0))
                has_nonzero = cot_reward != 0.0 or any(value != 0.0 for value in sid_rewards)
                has_positive = cot_reward > 0.0 or any(value > 0.0 for value in sid_rewards)
                if not has_nonzero:
                    continue
                record = candidate_record(event, cot, cot_index, source_row)
                key = record["sample_key"]
                if key not in nonzero_seen:
                    nonzero.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    nonzero_seen.add(key)
                    added_nonzero += 1
                if has_positive and key not in positive_seen:
                    positive.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    positive_seen.add(key)
                    added_positive += 1
        nonzero.flush()
        positive.flush()
        os.fsync(nonzero.fileno())
        os.fsync(positive.fileno())
    state_path.write_text(json.dumps({
        "source": str(source_path),
        "offset": offset,
        "nonzero_count": len(nonzero_seen),
        "positive_count": len(positive_seen),
        "updated_at_unix": time.time(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "added_nonzero": added_nonzero,
        "added_positive": added_positive,
        "nonzero_count": len(nonzero_seen),
        "positive_count": len(positive_seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "training_sample_exports"
    while True:
        result = export_available(args.run_dir.resolve(), output_dir.resolve())
        print(json.dumps(result), flush=True)
        if not args.follow:
            break
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    main()
