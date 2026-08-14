#!/usr/bin/env python3
"""Exact CPU parity audit for Alpha-mini baseline vs mini-cot cache."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL = "/data/models/onereason-8b-pretrain-competition"
IGNORE_INDEX = -100
TASK_RECOMMENDATION = 1


def fixed_2d(column, chunk: int) -> np.ndarray:
    return np.asarray(column.chunk(chunk).values.to_numpy(zero_copy_only=False)).reshape(-1, 8192)


def segment_bounds(sample_ids: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(sample_ids >= 0)
    if not active.size:
        return []
    end = int(active[-1]) + 1
    starts = np.flatnonzero(np.r_[True, sample_ids[1:end] != sample_ids[: end - 1]])
    ends = np.r_[starts[1:], end]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if sample_ids[a] >= 0]


def audit(old_path: Path, new_path: Path) -> dict:
    old = load_from_disk(str(old_path))["train"]
    new = load_from_disk(str(new_path))["train"]
    if len(old) != len(new):
        raise RuntimeError(f"pack count mismatch old={len(old)} new={len(new)}")
    if old.column_names != new.column_names:
        raise RuntimeError(f"column mismatch old={old.column_names} new={new.column_names}")
    unchanged = [c for c in old.column_names if c != "loss_weights"]
    mismatch = [c for c in unchanged if not old._data.table.column(c).equals(new._data.table.column(c))]
    if mismatch:
        raise RuntimeError(f"non-loss columns differ: {mismatch}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    vocab = tok.get_vocab()
    open_id, close_id = int(vocab["<think>"]), int(vocab["</think>"])
    labels_col = old._data.table.column("labels")
    sample_col = old._data.table.column("sample_ids")
    task_col = old._data.table.column("sample_task_ids")
    old_w_col = old._data.table.column("loss_weights")
    new_w_col = new._data.table.column("loss_weights")
    target_col = old._data.table.column("rec_pu_targets_json")
    changed = expected = unexpected = 0
    values = Counter()
    weights = {key: [0.0, 0.0] for key in (
        "cot_body", "cot_post_think", "cot_final_sid", "nothink", "recommendation_total",
    )}
    span_rows = 0
    for chunk in range(labels_col.num_chunks):
        labels = fixed_2d(labels_col, chunk).astype(np.int64, copy=False)
        samples = fixed_2d(sample_col, chunk).astype(np.int64, copy=False)
        tasks = fixed_2d(task_col, chunk).astype(np.int64, copy=False)
        oldw = fixed_2d(old_w_col, chunk).astype(np.float64, copy=False)
        neww = fixed_2d(new_w_col, chunk).astype(np.float64, copy=False)
        target_rows = target_col.chunk(chunk)
        diffs = ~np.isclose(oldw, neww, rtol=0.0, atol=1e-12)
        changed += int(diffs.sum())
        for row in range(labels.shape[0]):
            row_labels, row_samples, row_tasks = labels[row], samples[row], tasks[row]
            row_old, row_new, row_diff = oldw[row], neww[row], diffs[row]
            cot_ranges: list[tuple[int, int]] = []
            final_positions: set[int] = set()
            for target in json.loads(target_rows[row].as_py() or "[]"):
                if target.get("source_segment") != "recommendation_cot":
                    continue
                start, end = target.get("segment_start"), target.get("segment_end")
                if not isinstance(start, int) or not isinstance(end, int):
                    raise RuntimeError("invalid recommendation target boundary")
                cot_ranges.append((start, end))
                a = target.get("a_label_position")
                if not isinstance(a, int):
                    raise RuntimeError("missing a label position")
                final_positions.add(a - 1)
                for level in ("a", "b", "c"):
                    value = target.get(f"{level}_label_position")
                    if not isinstance(value, int):
                        raise RuntimeError("missing final SID position")
                    final_positions.add(value)
            allowed_change = np.zeros(8192, dtype=bool)
            for start, end in segment_bounds(row_samples):
                valid = (row_labels != IGNORE_INDEX) & (row_samples == row_samples[start])
                if not np.any(valid[start:end] & (row_tasks[start:end] == TASK_RECOMMENDATION)):
                    continue
                is_cot = (start, end) in cot_ranges
                open_pos = np.flatnonzero((row_labels == open_id) & valid)
                close_pos = np.flatnonzero((row_labels == close_id) & valid)
                if is_cot:
                    local_open = [p for p in open_pos if start <= p < end]
                    local_close = [p for p in close_pos if start <= p < end]
                    if len(local_open) != 1 or len(local_close) != 1 or local_open[0] >= local_close[0]:
                        raise RuntimeError("invalid CoT think span")
                    body = np.zeros(8192, dtype=bool)
                    body[local_open[0] : local_close[0] + 1] = True
                    body &= valid & (row_tasks == TASK_RECOMMENDATION)
                    expected += int(body.sum())
                    allowed_change |= body
                    values.update(float(v) for v in row_new[body])
                    span_rows += 1
                    post = valid & (row_tasks == TASK_RECOMMENDATION) & (np.arange(8192) > local_close[0]) & (np.arange(8192) < end)
                    weights["cot_body"][0] += float(row_old[body].sum()); weights["cot_body"][1] += float(row_new[body].sum())
                    weights["cot_post_think"][0] += float(row_old[post].sum()); weights["cot_post_think"][1] += float(row_new[post].sum())
                else:
                    segment = valid & (np.arange(8192) >= start) & (np.arange(8192) < end)
                    weights["nothink"][0] += float(row_old[segment].sum()); weights["nothink"][1] += float(row_new[segment].sum())
            if final_positions:
                positions = np.array(sorted(final_positions), dtype=np.int64)
                positions = positions[(positions >= 0) & (positions < 8192)]
                if positions.size:
                    weights["cot_final_sid"][0] += float(row_old[positions].sum()); weights["cot_final_sid"][1] += float(row_new[positions].sum())
                    if not np.allclose(row_new[positions], 8.0, rtol=0.0, atol=1e-12):
                        raise RuntimeError("final Gold SID weight is not exactly 8")
            unexpected += int(np.count_nonzero(row_diff & ~allowed_change))
            rec = (row_labels != IGNORE_INDEX) & (row_tasks == TASK_RECOMMENDATION)
            weights["recommendation_total"][0] += float(row_old[rec].sum()); weights["recommendation_total"][1] += float(row_new[rec].sum())
    if changed != expected or unexpected:
        raise RuntimeError(f"loss weight changes not limited to CoT body: changed={changed}, expected={expected}, unexpected={unexpected}")
    legal = {0.5 / n for n in range(1, 19)}
    if any(not any(math.isclose(value, item, abs_tol=1e-12) for item in legal) for value in values):
        raise RuntimeError(f"unexpected CoT weights: {sorted(values)}")
    return {
        "status": "MINI_COT_REPEAT_CACHE_AUDIT_PASS",
        "packs": len(new),
        "changed_loss_weight_positions": changed,
        "cot_span_segments": span_rows,
        "changed_values": {str(k): int(v) for k, v in sorted(values.items())},
        "weighted_mass": {key: {"old": oldv, "new": newv, "ratio": (newv / oldv if oldv else None)} for key, (oldv, newv) in weights.items()},
        "columns_unchanged_except_loss_weights": unchanged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.old, args.new)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
