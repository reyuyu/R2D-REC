"""Read-only REC-PU Phase 2 metadata audit for the BETA recommendation files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommendation_pu_phase2 import (  # noqa: E402
    METADATA_FIELDS,
    RecPUMetadataError,
    build_prefix_positive_sets,
    parse_recommendation_metadata,
    sid_to_token_ids,
)


def percentile(values: list[int], fraction: float) -> int:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    files = [
        args.dataset_dir / "onereason_recommendation_cot.jsonl",
        args.dataset_dir / "onereason_recommendation_nocot.jsonl",
    ]
    totals = Counter()
    group_sizes: list[int] = []
    errors = Counter()
    duplicate_gold = 0
    sample = None
    multi_positive_example = None
    token_checked = 0

    for path in files:
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            totals["total"] += 1
            if any(field not in row for field in METADATA_FIELDS):
                totals["metadata_missing"] += 1
                continue
            raw_gold = row["recommendation_all_gold_sids"]
            if isinstance(raw_gold, list) and len(raw_gold) != len(set(raw_gold)):
                duplicate_gold += 1
            try:
                metadata = parse_recommendation_metadata(row)
                for sid in (*metadata.all_gold_sids, metadata.current_gold_sid):
                    sid_to_token_ids(sid, tokenizer)
                    token_checked += 1
            except RecPUMetadataError as error:
                errors[error.code] += 1
                continue
            totals["metadata_valid"] += 1
            group_sizes.append(metadata.group_size)
            totals["singleton"] += int(metadata.group_size == 1)
            totals["multi_positive"] += int(metadata.group_size > 1)
            if sample is None:
                sample = {
                    "group_id": metadata.group_id[:12] + "...",
                    "group_size": metadata.group_size,
                    "all_gold_sids": [sid.render() for sid in metadata.all_gold_sids],
                    "current_gold_sid": metadata.current_gold_sid.render(),
                }
            if multi_positive_example is None and metadata.group_size > 1:
                positives = build_prefix_positive_sets(metadata)
                multi_positive_example = {
                    "group_id": metadata.group_id[:12] + "...",
                    "current_gold": metadata.current_gold_sid.render(),
                    "all_gold": [sid.render() for sid in metadata.all_gold_sids],
                    "P_a": list(positives.a),
                    "P_b": list(positives.b),
                    "P_c": list(positives.c),
                }

    valid = totals["metadata_valid"]
    report = {
        "total_recommendation_samples": totals["total"],
        "metadata_valid_count": valid,
        "metadata_missing_count": totals["metadata_missing"],
        "metadata_coverage": valid / totals["total"] if totals["total"] else 0.0,
        "singleton_count": totals["singleton"],
        "singleton_ratio": totals["singleton"] / valid if valid else 0.0,
        "multi_positive_count": totals["multi_positive"],
        "multi_positive_ratio": totals["multi_positive"] / valid if valid else 0.0,
        "group_size": {
            "mean": statistics.fmean(group_sizes) if group_sizes else 0.0,
            "p50": percentile(group_sizes, 0.5) if group_sizes else 0,
            "p90": percentile(group_sizes, 0.9) if group_sizes else 0,
            "max": max(group_sizes, default=0),
        },
        "duplicate_gold_samples": duplicate_gold,
        "exception_counts": dict(errors),
        "single_token_sid_components_checked": token_checked,
        "actual_schema": list(json.loads(next(files[0].open(encoding="utf-8"))).keys()),
        "deidentified_metadata_sample": sample,
        "multi_positive_prefix_example": multi_positive_example,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
