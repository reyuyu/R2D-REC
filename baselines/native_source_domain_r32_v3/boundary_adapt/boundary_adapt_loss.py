"""ABC-only, full-path multi-positive loss for Boundary Adaptation."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def assert_three_labels(labels: torch.Tensor) -> None:
    counts = labels.ne(IGNORE_INDEX).sum(-1)
    if not bool(torch.all(counts == 3)):
        raise ValueError(f"BOUNDARY_SUPERVISED_TOKEN_COUNT_PER_PATH=3 violated: {counts.tolist()}")


def per_path_boundary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one mean-ABC CE value per expanded path."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    assert_three_labels(labels)
    shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shifted_labels)
    return (token_loss * mask).sum(-1) / 3.0


def row_uniform_group_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor, *, total_paths: int, total_groups: int) -> torch.Tensor:
    """Unbiased row-uniform estimator of mean_group(mean_positive(path CE))."""
    if sample_weights.shape != (labels.shape[0],):
        raise ValueError("sample_weights must be [B]")
    if total_paths <= 0 or total_groups <= 0:
        raise ValueError("dataset provenance totals must be positive")
    # Do not normalize by the weights in this minibatch: B=1 would cancel 1/K.
    return (per_path_boundary_loss(logits, labels) * sample_weights).mean() * (float(total_paths) / float(total_groups))


def load_and_validate_provenance(rows_path: str | Path, stats_path: str | Path) -> dict[str, int]:
    """Load manifest totals and fail closed unless the JSONL exactly matches it."""
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    total_paths, total_groups = int(stats["expanded_paths"]), int(stats["original_groups"])
    rows = 0
    sizes: Counter[str] = Counter()
    weight_sums: Counter[str] = Counter()
    declared: dict[str, int] = {}
    with Path(rows_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line); group = str(row["boundary_group_id"])
            k, weight = int(row["boundary_group_size"]), float(row["boundary_sample_weight"])
            if k <= 0 or abs(weight - 1.0 / k) > 1e-12:
                raise ValueError(f"invalid K/weight at row {line_number}")
            if group in declared and declared[group] != k:
                raise ValueError(f"inconsistent K for group {group}")
            declared[group] = k; sizes[group] += 1; weight_sums[group] += weight; rows += 1
    if rows != total_paths or len(sizes) != total_groups:
        raise ValueError(f"manifest mismatch: rows={rows}/{total_paths}, groups={len(sizes)}/{total_groups}")
    for group, observed_k in sizes.items():
        if observed_k != declared[group] or abs(weight_sums[group] - 1.0) > 1e-10:
            raise ValueError(f"group {group} does not form a complete positive set")
    return {"total_paths": total_paths, "total_groups": total_groups}
