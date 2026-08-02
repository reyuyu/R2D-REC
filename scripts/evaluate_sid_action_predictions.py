#!/usr/bin/env python3
"""Compute approximate SID retrieval and action-select metrics from predictions.

Each JSONL record needs ``prediction`` plus either ``reference`` or ``output``.
SID is the exact three-part tuple (s_a, s_b, s_c); content type is deliberately
ignored to match the requested approximate metric definition.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SID_RE = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def unique_sids(text: str) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(SID_RE.findall(text)))


def score_records(records: list[dict]) -> dict:
    hit_counts = {1: 0, 8: 0, 16: 0, 32: 0}
    tp = fp = fn = evaluated = any_hit = 0
    coverage_sum = 0.0
    for record in records:
        prediction = str(record["prediction"])
        reference = str(record.get("reference", record.get("output", "")))
        pred_sids, gold_sids = unique_sids(prediction), set(unique_sids(reference))
        if not gold_sids:
            continue
        evaluated += 1
        overlap = set(pred_sids) & gold_sids
        any_hit += bool(overlap)
        coverage_sum += len(overlap) / len(gold_sids)
        for k in hit_counts:
            hit_counts[k] += bool(set(pred_sids[:k]) & gold_sids)
        pred_set = set(pred_sids)
        tp += len(overlap)
        fp += len(pred_set - gold_sids)
        fn += len(gold_sids - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "examples": evaluated,
        "sid_hit_at": {str(k): hit_counts[k] / evaluated if evaluated else 0.0 for k in hit_counts},
        "any_sid_hit_rate": any_hit / evaluated if evaluated else 0.0,
        "mean_gold_sid_coverage": coverage_sum / evaluated if evaluated else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    by_task: dict[str, list[dict]] = {}
    for record in records:
        by_task.setdefault(str(record.get("task", "all")), []).append(record)
    result = {
        "overall": score_records(records),
        "by_task": {task: score_records(task_records) for task, task_records in by_task.items()},
        "definition": (
            "SID uses exact (s_a,s_b,s_c). Hit@K is query-level any-gold-SID hit in the first K unique "
            "predictions. Action coverage is the mean fraction of all gold SIDs found; F1 is micro-F1 over SID sets."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
