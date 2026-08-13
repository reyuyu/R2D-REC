#!/usr/bin/env python3
"""Fail-closed parity and objective audit for Alpha CoT repeat weighting."""

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
EXPECTED_PACKS = 33_810
EXPECTED_COT_BODY_OLD = 25_232_442.0
EXPECTED_COT_BODY_NEW = 4_892_145.5
EXPECTED_FINAL_SID_MASS = 869_952.0
EXPECTED_NOCOT_MASS = 939_196.0
EXPECTED_REC_OLD = 27_365_089.0
EXPECTED_REC_NEW = 7_024_792.5


class AuditError(RuntimeError):
    pass


def _fixed_2d(column, chunk: int) -> np.ndarray:
    return np.asarray(column.chunk(chunk).values.to_numpy(zero_copy_only=False)).reshape(-1, 8192)


def _segment_bounds(sample_ids: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(sample_ids >= 0)
    if not active.size:
        return []
    end_active = int(active[-1]) + 1
    starts = np.flatnonzero(np.r_[True, sample_ids[1:end_active] != sample_ids[: end_active - 1]])
    ends = np.r_[starts[1:], end_active]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if sample_ids[start] >= 0]


def _component_ids(tokenizer) -> set[int]:
    vocab = tokenizer.get_vocab()
    return {
        int(token_id)
        for token, token_id in vocab.items()
        if token.startswith("<s_a_") or token.startswith("<s_b_") or token.startswith("<s_c_")
    }


def _target_positions(raw: str) -> list[int]:
    """Tolerantly recover the four final-SID positions serialized by the launcher."""
    result: list[int] = []
    for target in json.loads(raw or "[]"):
        if target.get("source_segment") != "recommendation_cot":
            continue
        # The metadata serializes a/b/c labels.  The final domain token is
        # immediately before a_label_position in the audited SID grammar.
        a_position = target.get("a_label_position")
        if not isinstance(a_position, int):
            raise ValueError("Target has no a_label_position.")
        result.append(a_position - 1)
        for key in ("a_label_position", "b_label_position", "c_label_position"):
            value = target.get(key)
            if not isinstance(value, int):
                raise ValueError(f"Target has invalid {key}.")
            result.append(value)
    return sorted(set(result))


def _cot_segments_and_final_positions(raw: str) -> tuple[set[tuple[int, int]], list[int]]:
    """Use persisted source metadata, never infer CoT from empty no-think tags."""
    cot_segments: set[tuple[int, int]] = set()
    final_positions: list[int] = []
    for target in json.loads(raw or "[]"):
        if target.get("source_segment") != "recommendation_cot":
            continue
        start, end = target.get("segment_start"), target.get("segment_end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("Invalid CoT target segment boundaries.")
        cot_segments.add((start, end))
        final_positions.extend(_target_positions(json.dumps([target])))
    return cot_segments, sorted(set(final_positions))


def audit(old_path: Path, new_path: Path) -> dict[str, object]:
    old = load_from_disk(str(old_path))["train"]
    new = load_from_disk(str(new_path))["train"]
    if len(old) != EXPECTED_PACKS or len(new) != EXPECTED_PACKS:
        raise AuditError(f"Pack count mismatch old={len(old)} new={len(new)} expected={EXPECTED_PACKS}")
    if old.column_names != new.column_names:
        raise AuditError(f"Column mismatch old={old.column_names} new={new.column_names}")
    unchanged_columns = [name for name in old.column_names if name != "loss_weights"]
    mismatch_columns = [name for name in unchanged_columns if not old._data.table.column(name).equals(new._data.table.column(name))]
    if mismatch_columns:
        raise AuditError(f"Non-loss cache columns differ: {mismatch_columns}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    vocab = tokenizer.get_vocab()
    open_id, close_id = int(vocab["<think>"]), int(vocab["</think>"])
    component_ids = _component_ids(tokenizer)
    labels_col = old._data.table.column("labels")
    sample_col = old._data.table.column("sample_ids")
    task_col = old._data.table.column("sample_task_ids")
    old_w_col = old._data.table.column("loss_weights")
    new_w_col = new._data.table.column("loss_weights")
    target_col = old._data.table.column("rec_pu_targets_json") if "rec_pu_targets_json" in old.column_names else None

    changed = 0
    expected_body_positions = 0
    unexpected_changed = 0
    no_think_old = no_think_new = 0.0
    cot_body_old = cot_body_new = 0.0
    rec_old = rec_new = 0.0
    final_sid_old = final_sid_new = 0.0
    changed_values: Counter[float] = Counter()
    target_errors = 0

    for chunk in range(labels_col.num_chunks):
        labels = _fixed_2d(labels_col, chunk).astype(np.int64, copy=False)
        sample_ids = _fixed_2d(sample_col, chunk).astype(np.int64, copy=False)
        task_ids = _fixed_2d(task_col, chunk).astype(np.int64, copy=False)
        old_w = _fixed_2d(old_w_col, chunk).astype(np.float64, copy=False)
        new_w = _fixed_2d(new_w_col, chunk).astype(np.float64, copy=False)
        diffs = ~np.isclose(old_w, new_w, rtol=0.0, atol=1e-12)
        valid = labels != IGNORE_INDEX
        rec_valid = valid & (task_ids == TASK_RECOMMENDATION)
        rec_old += float(old_w[rec_valid].sum())
        rec_new += float(new_w[rec_valid].sum())
        changed += int(diffs.sum())
        for row in range(labels.shape[0]):
            row_labels, row_sample, row_tasks = labels[row], sample_ids[row], task_ids[row]
            row_old, row_new, row_diff = old_w[row], new_w[row], diffs[row]
            if target_col is None:
                raise AuditError("rec_pu_targets_json is required to distinguish CoT from no-think.")
            try:
                cot_segments, final_positions = _cot_segments_and_final_positions(target_col.chunk(chunk)[row].as_py())
            except Exception as error:
                raise AuditError(f"Cannot parse packed recommendation metadata row={row}: {error}") from error
            for start, end in _segment_bounds(row_sample):
                segment_labels = row_labels[start:end]
                segment_tasks = row_tasks[start:end]
                if not np.any((segment_labels != IGNORE_INDEX) & (segment_tasks == TASK_RECOMMENDATION)):
                    continue
                open_pos = np.flatnonzero(segment_labels == open_id)
                close_pos = np.flatnonzero(segment_labels == close_id)
                is_cot = (start, end) in cot_segments
                segment_valid = segment_labels != IGNORE_INDEX
                if not is_cot:
                    if np.any(row_diff[start:end]):
                        unexpected_changed += int(row_diff[start:end].sum())
                    no_think_old += float(row_old[start:end][segment_valid].sum())
                    no_think_new += float(row_new[start:end][segment_valid].sum())
                    continue
                if len(open_pos) != 1 or len(close_pos) != 1 or int(open_pos[0]) >= int(close_pos[0]):
                    raise AuditError(f"Invalid persisted CoT think span in packed segment start={start} end={end}.")
                body_start, body_end = start + int(open_pos[0]), start + int(close_pos[0])
                body_mask = np.zeros(8192, dtype=np.bool_)
                body_mask[body_start : body_end + 1] = True
                body_mask &= valid[row] & (row_tasks == TASK_RECOMMENDATION) & (row_sample == row_sample[start])
                expected_body_positions += int(body_mask.sum())
                cot_body_old += float(row_old[body_mask].sum())
                cot_body_new += float(row_new[body_mask].sum())
                changed_values.update(float(value) for value in row_new[body_mask])
                unexpected_changed += int(np.count_nonzero(row_diff & ~body_mask & (row_sample == row_sample[start])))

            for position in final_positions:
                if not (0 <= position < 8192):
                    target_errors += 1
                    continue
                token = int(labels[row, position])
                if token == IGNORE_INDEX:
                    target_errors += 1
                    continue
                final_sid_old += float(old_w[row, position])
                final_sid_new += float(new_w[row, position])

    if changed != expected_body_positions or unexpected_changed:
        raise AuditError(
            f"Changed positions are not exactly recommendation_cot body: changed={changed} "
            f"expected_body={expected_body_positions} unexpected={unexpected_changed}"
        )
    allowed = {0.5 / divisor for divisor in range(1, 19)}
    if any(not any(math.isclose(value, legal, rel_tol=0.0, abs_tol=1e-12) for legal in allowed) for value in changed_values):
        raise AuditError(f"Unexpected CoT body weight(s): {sorted(changed_values)}")
    if target_errors or not math.isclose(final_sid_old, EXPECTED_FINAL_SID_MASS, abs_tol=1e-6) or final_sid_old != final_sid_new:
        raise AuditError(f"Final SID invariant failed old={final_sid_old} new={final_sid_new} errors={target_errors}")
    expected = {
        "cot_body_old": EXPECTED_COT_BODY_OLD,
        "cot_body_new": EXPECTED_COT_BODY_NEW,
        "no_think_old": EXPECTED_NOCOT_MASS,
        "no_think_new": EXPECTED_NOCOT_MASS,
        "rec_old": EXPECTED_REC_OLD,
        "rec_new": EXPECTED_REC_NEW,
    }
    actual = {
        "cot_body_old": cot_body_old,
        "cot_body_new": cot_body_new,
        "no_think_old": no_think_old,
        "no_think_new": no_think_new,
        "rec_old": rec_old,
        "rec_new": rec_new,
    }
    failed = {name: (actual[name], expected[name]) for name in expected if not math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-4)}
    if failed:
        raise AuditError(f"Static mass audit mismatch: {failed}")
    return {
        "status": "ALPHA_COT_REPEAT_CACHE_AUDIT_PASS",
        "old_cache": str(old_path), "new_cache": str(new_path), "packs": len(new),
        "unchanged_columns": unchanged_columns, "changed_loss_weight_positions": changed,
        "cot_body_old_weighted_mass": cot_body_old, "cot_body_new_weighted_mass": cot_body_new,
        "cot_body_ratio": cot_body_new / cot_body_old,
        "final_sid_old_weighted_mass": final_sid_old, "final_sid_new_weighted_mass": final_sid_new,
        "nothink_old_weighted_mass": no_think_old, "nothink_new_weighted_mass": no_think_new,
        "recommendation_old_weighted_mass": rec_old, "recommendation_new_weighted_mass": rec_new,
        "changed_weight_values": {str(key): int(value) for key, value in sorted(changed_values.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.old, args.new)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except AuditError as error:
        print(f"ALPHA_COT_REPEAT_CACHE_AUDIT_FAIL: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
