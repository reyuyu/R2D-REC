#!/usr/bin/env python3
"""Offline S_prefix scale and parser reward-hacking audits."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

from grpo_sid import parse_sid

from gr_rec_dsr_v1.dsr_objectives import prefix_support
from gr_rec_dsr_v1.dsr_parser import canonical_sid, parse_interest_section


DATA = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
FORMAL = Path("/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/probes.jsonl")
FORMAL_TRAIN = Path("/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/traces/traces.jsonl")
DSR = Path("/data/GRPO/runs/GR-REC-DSR-V1-SMOKE-20260818-01/traces/traces.jsonl")
OUTPUT = Path("/data/GRPO/outputs/GR-REC-DSR-V1-OFFLINE-AUDIT.json")


def _gold_set(values):
    return {sid for value in values if (sid := parse_sid(value)) is not None}


def _bucket(count):
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def _stats(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": values[max(0, math.ceil(0.9 * len(values)) - 1)],
        "min": values[0],
        "max": values[-1],
    }


def _load_prompt_rows():
    rows = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["route"] == "think":
            rows[row["recommendation_group_id"]] = row
    return rows


def _records():
    prompt_rows = _load_prompt_rows()
    records = []
    for line in FORMAL.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        for index, candidate in enumerate(event["think"]["candidates"]):
            records.append({
                "source": "gr_rec_v1_probe",
                "step": event["step"],
                "group_id": event["group_id"],
                "candidate_id": index,
                "domain": event["target_domain"],
                "prompt": event["think_prompt"],
                "gold_sids": event["gold_sids"],
                "completion": candidate["completion"],
                "beam_sids": candidate["beam_sids"],
            })
    for line in DSR.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("route") != "think":
            continue
        row = prompt_rows[event["group_id"]]
        for index, candidate in enumerate(event["candidates"]):
            records.append({
                "source": "dsr_smoke",
                "step": event["step"],
                "group_id": event["group_id"],
                "candidate_id": index,
                "domain": row["target_domain"],
                "prompt": row["prompt"],
                "gold_sids": event["gold_sids"],
                "completion": candidate["completion"],
                "beam_sids": candidate["beam_sids"],
            })
    for line in FORMAL_TRAIN.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("route") != "think":
            continue
        row = prompt_rows[event["group_id"]]
        for index, candidate in enumerate(event["candidates"]):
            records.append({
                "source": "gr_rec_v1_train_trace",
                "step": event["step"],
                "group_id": event["group_id"],
                "candidate_id": index,
                "domain": row["target_domain"],
                "prompt": row["prompt"],
                "gold_sids": event["gold_sids"],
                "completion": candidate["completion"],
                "beam_sids": candidate["beam_sids"],
            })
    return records


def _prefix_audit(records):
    grouped = defaultdict(lambda: defaultdict(list))
    included = 0
    skipped = 0
    for record in records:
        if not record["beam_sids"] or len(record["beam_sids"]) != 32:
            skipped += 1
            continue
        included += 1
        gold = _gold_set(record["gold_sids"])
        gold_a_count = len({sid[:2] for sid in gold})
        key = f"{record['domain']}|{_bucket(gold_a_count)}"
        values = prefix_support(
            [tuple(value) if value is not None else None for value in record["beam_sids"]], gold
        )
        for name, value in values.items():
            grouped[key][name].append(value)
        grouped[key]["gold_a_count"].append(gold_a_count)
    distributions = {
        key: {
            "domain": key.split("|", 1)[0],
            "gold_a_bucket": key.split("|", 1)[1],
            "candidate_count": len(values["s_prefix"]),
            "gold_a_count": _stats(values["gold_a_count"]),
            "s_a": _stats(values["s_a"]),
            "s_ab": _stats(values["s_ab"]),
            "s_prefix": _stats(values["s_prefix"]),
        }
        for key, values in sorted(grouped.items())
    }
    return {
        "included_candidates": included,
        "skipped_without_beam32": skipped,
        "distributions": distributions,
    }


def _parser_audit(records):
    sample = sorted(records, key=lambda item: (
        0 if item["source"] == "dsr_smoke" else 1,
        item["step"], item["group_id"], item["candidate_id"],
    ))[:50]
    raw_counts = []
    grounded_counts = []
    evidence_per_bullet = []
    unique_grounded = []
    reuse_rates = []
    repeated_title_completions = 0
    title_counter = Counter()
    suspected = []
    success = 0
    fake_evidence = 0
    total_evidence = 0
    for record in sample:
        parsed = parse_interest_section(record["completion"], record["prompt"])
        success += int(parsed.parser_success)
        raw_counts.append(parsed.bullet_count)
        grounded_counts.append(parsed.grounded_count)
        titles = [item.title.strip().lower() for item in parsed.bullets if item.title.strip()]
        title_counter.update(titles)
        repeated_title_completions += int(len(titles) != len(set(titles)))
        occurrence = Counter(
            sid for bullet in parsed.grounded_bullets for sid in set(bullet.grounded_evidence)
        )
        repeated = {sid for sid, count in occurrence.items() if count > 1}
        grounded_unique = set(occurrence)
        unique_grounded.append(len(grounded_unique))
        reuse_rates.append(len(repeated) / len(grounded_unique) if grounded_unique else 0.0)
        for bullet in parsed.bullets:
            evidence_per_bullet.append(len(bullet.evidence))
            total_evidence += len(bullet.evidence)
            fake_evidence += len(set(bullet.evidence) - set(bullet.grounded_evidence))
        if parsed.grounded_count >= 2 and repeated:
            suspected.append({
                "source": record["source"],
                "step": record["step"],
                "group_id": record["group_id"],
                "candidate_id": record["candidate_id"],
                "titles": titles,
                "repeated_grounded_sids": [canonical_sid(value) for value in sorted(repeated)],
                "grounded_count": parsed.grounded_count,
            })
    return {
        "sample_count": len(sample),
        "source_counts": dict(Counter(item["source"] for item in sample)),
        "parser_success_rate": success / len(sample),
        "raw_interest_count": _stats(raw_counts),
        "grounded_interest_count": _stats(grounded_counts),
        "evidence_sid_count_per_bullet": _stats(evidence_per_bullet),
        "grounded_unique_sid_per_completion": _stats(unique_grounded),
        "cross_bullet_grounded_sid_reuse_rate": _stats(reuse_rates),
        "completions_with_repeated_titles": repeated_title_completions,
        "fake_evidence_fraction": fake_evidence / total_evidence if total_evidence else 0.0,
        "top_normalized_titles": title_counter.most_common(20),
        "suspected_mechanical_copy_count": len(suspected),
        "suspected_mechanical_copy_examples": suspected[:10],
    }


def main():
    records = _records()
    payload = {
        "record_count": len(records),
        "prefix_scale": _prefix_audit(records),
        "parser_hacking": _parser_audit(records),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
