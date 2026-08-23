"""ABC-only, full-path multi-positive loss for Boundary Adaptation."""
from __future__ import annotations
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def assert_three_labels(labels: torch.Tensor) -> None:
    counts = labels.ne(IGNORE_INDEX).sum(-1)
    if not bool(torch.all(counts == 3)):
        raise ValueError(f"BOUNDARY_SUPERVISED_TOKEN_COUNT_PER_PATH=3 violated: {counts.tolist()}")

def weighted_boundary_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    """Mean of complete ABC paths, with expanded positives summing to one/group."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if sample_weights.shape != (labels.shape[0],):
        raise ValueError("sample_weights must be [B]")
    assert_three_labels(labels)
    shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).view_as(shifted_labels)
    path_loss = (token_loss * mask).sum(-1) / 3.0
    return (path_loss * sample_weights).sum() / sample_weights.sum()
