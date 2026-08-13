"""Fail-closed content audit for Alpha's packed SID8 static cache.

The environment variable is intentionally not trusted here: this module scans
the persisted Arrow cache that a formal run will actually consume.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


TASKS = ("material", "recommendation", "user_action", "user_chain")
EXPECTED_ALL = {0.0: 235_784_325, 1.0: 37_758_578, 2.0: 0, 3.0: 0, 4.0: 765_393, 8.0: 2_663_224}
EXPECTED_ITEM = {"material": 313_148, "recommendation": 994_312, "user_action": 899_460, "user_chain": 456_304}
EXPECTED_CANONICAL_SEGMENTS = 11_072
MODEL = "/data/models/onereason-8b-pretrain-competition"


class AlphaSID8CacheContractError(RuntimeError):
    pass


def _item_lookup() -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    vocab = tokenizer.get_vocab()
    item_ids = [
        int(token_id)
        for token, token_id in vocab.items()
        if ((token.startswith("<s_a_") or token.startswith("<s_b_") or token.startswith("<s_c_")) and token.endswith(">"))
        or token in {"<|video_begin|>", "<|prod_begin|>", "<|ad_begin|>", "<|living_begin|>", "<|search_begin|>"}
    ]
    lookup = np.zeros(max(vocab.values()) + 1, dtype=np.bool_)
    lookup[np.asarray(item_ids, dtype=np.int64)] = True
    return lookup


def _update(counter: Counter[float], values: np.ndarray, mask: np.ndarray) -> None:
    if mask.any():
        values_unique, counts = np.unique(values[mask], return_counts=True)
        counter.update({float(value): int(count) for value, count in zip(values_unique, counts)})


def scan_alpha_sid8_cache(cache_path: str | Path) -> dict[str, object]:
    """Scan all cache tokens and return contract-relevant counts.

    Raises ``AlphaSID8CacheContractError`` on any invalid cache contents.
    """
    cache_path = Path(cache_path)
    dataset_dict = load_from_disk(str(cache_path))
    if "train" not in dataset_dict:
        raise AlphaSID8CacheContractError(f"Missing train split: {cache_path}")
    dataset = dataset_dict["train"]
    required = {"labels", "loss_weights", "sample_ids", "sample_task_ids"}
    if not required <= set(dataset.column_names):
        raise AlphaSID8CacheContractError(f"Missing cache columns: {sorted(required - set(dataset.column_names))}")

    lookup = _item_lookup()
    table = dataset._data.table
    columns = {name: table.column(name) for name in required}
    all_weights: Counter[float] = Counter()
    supervised_weights: Counter[float] = Counter()
    item_weights = {task: Counter() for task in TASKS}
    text_weights = {task: Counter() for task in TASKS}
    canonical_segments = 0
    errors: Counter[str] = Counter()

    chunks = columns["labels"].num_chunks
    for chunk in range(chunks):
        labels = np.asarray(columns["labels"].chunk(chunk).values.to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1, 8192)
        weights = np.asarray(columns["loss_weights"].chunk(chunk).values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 8192)
        sample_ids = np.asarray(columns["sample_ids"].chunk(chunk).values.to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1, 8192)
        task_ids = np.asarray(columns["sample_task_ids"].chunk(chunk).values.to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1, 8192)
        valid = labels != -100
        flat_labels, flat_weights, flat_tasks = labels.ravel(), weights.ravel(), task_ids.ravel()
        flat_valid = valid.ravel()
        _update(all_weights, flat_weights, np.ones_like(flat_valid, dtype=np.bool_))
        _update(supervised_weights, flat_weights, flat_valid)
        errors["ignored_nonzero"] += int(np.count_nonzero((~flat_valid) & (flat_weights != 0.0)))
        errors["supervised_zero"] += int(np.count_nonzero(flat_valid & (flat_weights == 0.0)))
        errors["invalid_task"] += int(np.count_nonzero(flat_valid & ~np.isin(flat_tasks, (0, 1, 2, 3))))
        in_range = (flat_labels >= 0) & (flat_labels < len(lookup))
        is_item = np.zeros(flat_labels.shape, dtype=np.bool_)
        is_item[in_range] = lookup[flat_labels[in_range]]
        for task_id, task in enumerate(TASKS):
            task_mask = flat_valid & (flat_tasks == task_id)
            _update(item_weights[task], flat_weights, task_mask & is_item)
            _update(text_weights[task], flat_weights, task_mask & ~is_item)

        # Segment-level canonical contract: only material canonical segments
        # may carry response-wide weight 4, and every supervised token there
        # must be exactly 4.
        for row in range(labels.shape[0]):
            starts = np.flatnonzero(np.r_[True, sample_ids[row, 1:] != sample_ids[row, :-1]])
            ends = np.r_[starts[1:], labels.shape[1]]
            for start, end in zip(starts, ends):
                segment_valid = valid[row, start:end]
                if not segment_valid.any():
                    continue
                segment_tasks = np.unique(task_ids[row, start:end][segment_valid])
                if len(segment_tasks) != 1:
                    errors["mixed_task_segment"] += 1
                    continue
                segment_task = int(segment_tasks[0])
                segment_weights = weights[row, start:end][segment_valid]
                if np.any(segment_weights == 4.0):
                    if segment_task != 0 or not np.all(segment_weights == 4.0):
                        errors["invalid_canonical_segment"] += 1
                    else:
                        canonical_segments += 1

    report = {
        "cache": str(cache_path),
        "packs": len(dataset),
        "all_weight_counts": {str(key): int(all_weights.get(key, 0)) for key in sorted(EXPECTED_ALL)},
        "supervised_weight_counts": {str(key): int(supervised_weights.get(key, 0)) for key in sorted(supervised_weights)},
        "item_weight_counts": {task: {str(key): int(value) for key, value in sorted(counter.items())} for task, counter in item_weights.items()},
        "text_weight_counts": {task: {str(key): int(value) for key, value in sorted(counter.items())} for task, counter in text_weights.items()},
        "canonical_segments": canonical_segments,
        "errors": {key: int(value) for key, value in errors.items() if value},
    }
    failures: list[str] = []
    if {float(key): value for key, value in ((key, all_weights.get(key, 0)) for key in EXPECTED_ALL)} != EXPECTED_ALL:
        failures.append(f"all_weight_counts={report['all_weight_counts']}")
    for task, expected in EXPECTED_ITEM.items():
        if item_weights[task].get(8.0, 0) != expected or sum(item_weights[task].values()) != expected:
            failures.append(f"{task}_item_weights={dict(item_weights[task])}")
        allowed_text_weights = {1.0, 4.0} if task == "material" else {1.0}
        if any(weight not in allowed_text_weights for weight in text_weights[task]):
            failures.append(f"{task}_text_weights={dict(text_weights[task])}")
    if canonical_segments != EXPECTED_CANONICAL_SEGMENTS:
        failures.append(f"canonical_segments={canonical_segments}")
    nonzero_errors = {key: value for key, value in errors.items() if value}
    if nonzero_errors:
        failures.append(f"alignment_errors={nonzero_errors}")
    if failures:
        raise AlphaSID8CacheContractError("; ".join(failures))
    return report
