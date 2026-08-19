"""CPU-only gate-frequency audit over existing NoThink probes and traces."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from grpo_sid import parse_sid
from nothink_bridge import BRIDGE_COLLAPSE_AB, BRIDGE_DEAD_A, plan_nothink_bridge


SOURCES = {
    "dsr_simple_full_probe": Path(
        "/data/GRPO/runs/GR-REC-DSR-SIMPLE-V1-FULL-E1-GATE200-20260819/probes.jsonl"
    ),
    "dsr_simple_full_trace": Path(
        "/data/GRPO/runs/GR-REC-DSR-SIMPLE-V1-FULL-E1-GATE200-20260819/traces/traces.jsonl"
    ),
    "gr_rec_dsr_pilot_probe": Path(
        "/data/GRPO/runs/GR-REC-DSR-V1-PILOT200-20260818/probes.jsonl"
    ),
    "gr_rec_dsr_pilot_trace": Path(
        "/data/GRPO/runs/GR-REC-DSR-V1-PILOT200-20260818/traces/traces.jsonl"
    ),
}


def _bucket(count):
    return "1-2" if count <= 2 else "3-4" if count <= 4 else "5+"


def _a_bucket(count):
    return "1" if count == 1 else "2" if count == 2 else "3+"


def _rows(path, source):
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if source.endswith("probe"):
            candidates = record.get("nothink", {}).get("candidates", [])
            target_domain = record.get("target_domain")
        else:
            if record.get("route") != "no_think" or record.get("scope") != "global_group":
                continue
            candidates = record.get("candidates", [])
            parsed_gold = [parse_sid(value) for value in record.get("gold_sids", [])]
            target_domain = next((sid[0] for sid in parsed_gold if sid is not None), None)
        if len(candidates) != 8 or target_domain is None:
            continue
        gold = [parse_sid(value) for value in record.get("gold_sids", [])]
        gold = [sid for sid in gold if sid is not None]
        rewards = [candidate.get("reward") for candidate in candidates]
        predicted = [candidate.get("parsed_sid") for candidate in candidates]
        if any(value is None for value in rewards):
            continue
        plan = plan_nothink_bridge(rewards, predicted, gold, target_domain)
        unique_a = len({(sid[0], sid[1]) for sid in gold if sid[0] == target_domain})
        yield {
            "source": source,
            "step": record.get("step"),
            "group_id": record.get("group_id"),
            "domain": target_domain,
            "gold_count": len(gold),
            "gold_count_bucket": _bucket(len(gold)),
            "unique_gold_a": unique_a,
            "unique_gold_a_bucket": _a_bucket(unique_a),
            "branch": plan.branch,
            "dead_a_target_count": len(plan.gold_a_targets) if plan.branch == BRIDGE_DEAD_A else None,
            "missing_a_target_count": len(plan.missing_a_targets) if plan.branch == BRIDGE_COLLAPSE_AB else None,
            "current_b_target_count": len(plan.current_b_targets) if plan.branch == BRIDGE_COLLAPSE_AB else None,
        }


def _summary(rows):
    counts = Counter(row["branch"] for row in rows)
    total = len(rows)
    return {
        "n": total,
        "counts": dict(sorted(counts.items())),
        "rates": {key: value / total for key, value in sorted(counts.items())} if total else {},
    }


def _percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _exposure(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def build_audit():
    observations = []
    unavailable = []
    for source, path in SOURCES.items():
        if path.exists():
            observations.extend(_rows(path, source))
        else:
            unavailable.append({"source": source, "reason": "file missing", "path": str(path)})
    unavailable.append({
        "source": "dsr_simple_full_simple_forensic",
        "reason": "UNAVAILABLE: file contains Think candidates only; no NoThink predicted SID groups",
    })

    result = {
        "contract": "GR_REC_ThinkExactClamp_Ablation_v1 Phase 2 bridge gate replay",
        "cpu_only": True,
        "overall": _summary(observations),
        "by_source": {},
        "by_domain": {},
        "by_gold_count": {},
        "by_unique_gold_a": {},
        "exposure": {
            "dead_zero_a_unique_gold_a_targets": _exposure([
                row["dead_a_target_count"] for row in observations
                if row["dead_a_target_count"] is not None
            ]),
            "a_collapse_missing_a_targets": _exposure([
                row["missing_a_target_count"] for row in observations
                if row["missing_a_target_count"] is not None
            ]),
            "a_collapse_current_b_targets": _exposure([
                row["current_b_target_count"] for row in observations
                if row["current_b_target_count"] is not None
            ]),
        },
        "unavailable": unavailable,
        "weighting_note": "Each observation is one G8 group; target counts use mean CE and do not scale group loss.",
    }
    dimensions = {
        "by_source": "source",
        "by_domain": "domain",
        "by_gold_count": "gold_count_bucket",
        "by_unique_gold_a": "unique_gold_a_bucket",
    }
    for output_key, field in dimensions.items():
        grouped = defaultdict(list)
        for row in observations:
            grouped[str(row[field])].append(row)
        result[output_key] = {key: _summary(value) for key, value in sorted(grouped.items())}
    return result, observations


if __name__ == "__main__":
    audit, rows = build_audit()
    output_dir = Path(__file__).resolve().parent
    (output_dir / "nothink_bridge_history_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "nothink_bridge_history_observations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
