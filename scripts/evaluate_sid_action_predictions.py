#!/usr/bin/env python3
"""Compute SID retrieval, validity, duplication and stopping metrics.

Each JSONL record needs ``prediction`` plus either ``reference`` or ``output``.
Full Action Select SID validity uses the domain+a+b+c four-tuple. The legacy
three-part retrieval metrics are retained for compatibility.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SID_RE = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
FULL_SID_RE = re.compile(
    r"(<\|(?:video|prod|living|ad)_begin\|>)\s*"
    r"(<s_a_\d+>)\s*(<s_b_\d+>)\s*(<s_c_\d+>)"
)
SEMANTIC_TOKEN_RE = re.compile(
    r"<\|(?:video|prod|living|ad)_begin\|>|<s_[abc]_\d+>"
)


def unique_sids(text: str) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(SID_RE.findall(text)))


def full_sids(text: str) -> list[tuple[str, str, str, str]]:
    return FULL_SID_RE.findall(text)


def score_records(records: list[dict]) -> dict:
    hit_counts = {1: 0, 8: 0, 16: 0, 32: 0}
    tp = fp = fn = evaluated = any_hit = 0
    full_tp = full_fp = full_fn = 0
    predicted = valid = hallucinated = duplicates = format_errors = unfinished = 0
    legal_wrong = over_selected = under_selected = 0
    count_error_sum = 0
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
        pred_full = full_sids(prediction)
        pred_full_set = set(pred_full)
        gold_full_set = set(full_sids(reference))
        history_text = str(record.get("history", record.get("prompt", record.get("input", ""))))
        history_full_set = set(full_sids(history_text))
        predicted += len(pred_full)
        duplicates += len(pred_full) - len(pred_full_set)
        if history_full_set:
            valid += len(pred_full_set & history_full_set)
            hallucinated += len(pred_full_set - history_full_set)
            legal_wrong += len((pred_full_set & history_full_set) - gold_full_set)
        semantic_count = len(SEMANTIC_TOKEN_RE.findall(prediction))
        format_errors += bool(semantic_count != 4 * len(pred_full) or (gold_full_set and not pred_full))
        unfinished += bool(record.get("finished") is False or record.get("finish_reason") == "length")
        count_error_sum += abs(len(pred_full_set) - len(gold_full_set))
        over_selected += max(0, len(pred_full_set) - len(gold_full_set))
        under_selected += max(0, len(gold_full_set) - len(pred_full_set))
        full_overlap = pred_full_set & gold_full_set
        full_tp += len(full_overlap)
        full_fp += len(pred_full_set - gold_full_set)
        full_fn += len(gold_full_set - pred_full_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    full_precision = full_tp / (full_tp + full_fp) if full_tp + full_fp else 0.0
    full_recall = full_tp / (full_tp + full_fn) if full_tp + full_fn else 0.0
    validity_total = valid + hallucinated
    return {
        "examples": evaluated,
        "sid_hit_at": {str(k): hit_counts[k] / evaluated if evaluated else 0.0 for k in hit_counts},
        "any_sid_hit_rate": any_hit / evaluated if evaluated else 0.0,
        "mean_gold_sid_coverage": coverage_sum / evaluated if evaluated else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "action_valid_sid_rate": valid / validity_total if validity_total else 0.0,
        "action_hallucinated_sid_rate": hallucinated / validity_total if validity_total else 0.0,
        "action_duplicate_sid_rate": duplicates / predicted if predicted else 0.0,
        "action_unique_sid_rate": (predicted - duplicates) / predicted if predicted else 0.0,
        "action_prediction_count_error": count_error_sum / evaluated if evaluated else 0.0,
        "action_set_precision": full_precision,
        "action_set_recall": full_recall,
        "action_set_f1": (
            2 * full_precision * full_recall / (full_precision + full_recall)
            if full_precision + full_recall
            else 0.0
        ),
        "action_error_counts": {
            "history_out_sid": hallucinated,
            "duplicate_sid": duplicates,
            "legal_but_wrong_sid": legal_wrong,
            "missed_sid": full_fn,
            "over_selected_sid": over_selected,
            "under_selected_sid": under_selected,
            "format_error": format_errors,
            "not_stopped": unfinished,
        },
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
            "Legacy Hit@K uses (s_a,s_b,s_c). Action validity, duplication and set metrics use the exact "
            "domain+s_a+s_b+s_c tuple. Validity requires a history/prompt/input field. not_stopped uses an "
            "explicit finished=false or finish_reason=length signal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
