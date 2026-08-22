"""CPU-only strict Gold CoT provenance audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units

DEFAULT_GRPO = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
DEFAULT_SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
JOIN_KEY = "aux_metadata_json.recommendation_group_id"


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(len(ordered) - 1, int(position) + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[int]) -> dict:
    return {
        "min": min(values) if values else None,
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_think_groups(path: Path) -> dict[str, dict]:
    groups = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("route") == "think":
                groups[record["recommendation_group_id"]] = record
    return groups


def load_gold(path: Path, wanted: set[str]) -> tuple[dict[str, str], dict[str, int], int]:
    variants: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_segment") != "recommendation_cot":
                continue
            metadata = json.loads(record.get("aux_metadata_json") or "{}")
            group_id = metadata.get("recommendation_group_id")
            if group_id not in wanted:
                continue
            output = record.get("output") or ""
            cot = output.split("</think>", 1)[0] + "</think>" if "</think>" in output else output
            variants[group_id].add(cot)
            row_counts[group_id] += 1
    ambiguous = sum(len(items) > 1 for items in variants.values())
    resolved = {group_id: next(iter(items)) for group_id, items in variants.items() if len(items) == 1}
    return resolved, dict(row_counts), ambiguous


def audit(grpo_path: Path = DEFAULT_GRPO, source_path: Path = DEFAULT_SOURCE, include_hash: bool = True) -> dict:
    groups = load_think_groups(grpo_path)
    gold, row_counts, ambiguous = load_gold(source_path, set(groups))
    missing = sorted(set(groups) - set(gold))
    interest_counts, grounded_counts, lengths = [], [], []
    parser_success = 0
    domain_counts = Counter()
    sample_rows = []
    for group_id, record in groups.items():
        domain_counts[record.get("target_domain", "unknown")] += 1
        if group_id not in gold:
            continue
        parsed = extract_interest_units(gold[group_id], record["prompt"])
        parser_success += int(parsed.parser_success)
        interest_counts.append(len(parsed.units))
        grounded_counts.append(sum(bool(unit.grounded_evidence_sids) for unit in parsed.units))
        lengths.append(len(gold[group_id]))
    rng = random.Random(20260822)
    for group_id in rng.sample(sorted(gold), min(12, len(gold))):
        record = groups[group_id]
        parsed = extract_interest_units(gold[group_id], record["prompt"])
        sample_rows.append({
            "group_id": group_id,
            "gold_cot_source_key": group_id,
            "target_domain": record.get("target_domain"),
            "gold_interest_count": len(parsed.units),
        })
    duplicate_groups = sum(count > 1 for count in row_counts.values())
    duplicate_extra_rows = sum(max(0, count - 1) for count in row_counts.values())
    report = {
        "experiment": "GR_REC_Think_CompositeInterest_v1",
        "join": {
            "key": JOIN_KEY,
            "method": "exact recommendation_group_id equality; no fuzzy matching",
            "duplicate_resolution": "allowed only when all CoT prefixes are byte-identical",
        },
        "source": {
            "grpo_dataset": str(grpo_path),
            "gold_dataset": str(source_path),
            "gold_dataset_size_bytes": source_path.stat().st_size,
            "gold_dataset_sha256": file_sha256(source_path) if include_hash else None,
        },
        "counts": {
            "think_groups": len(groups),
            "matched": len(gold),
            "missing": len(missing),
            "duplicate_join_groups": duplicate_groups,
            "duplicate_extra_rows": duplicate_extra_rows,
            "duplicate_unresolved": ambiguous,
            "ambiguous": ambiguous,
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "gold_parser": {
            "success": parser_success,
            "evaluated": len(gold),
            "success_rate": parser_success / len(gold) if gold else 0.0,
            "interest_count": distribution(interest_counts),
            "grounded_interest_count": distribution(grounded_counts),
            "cot_character_length": distribution(lengths),
        },
        "missing_group_ids": missing,
        "deterministic_samples": sample_rows,
        "data_provenance_ready": not missing and ambiguous == 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    report = audit(include_hash=not args.skip_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print("DATA_PROVENANCE_READY=" + ("YES" if report["data_provenance_ready"] else "NO"))


if __name__ == "__main__":
    main()
