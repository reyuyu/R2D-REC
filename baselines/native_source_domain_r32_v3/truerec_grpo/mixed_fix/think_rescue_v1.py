"""Pure CPU A-coverage rescue math for the Mixed-Fix Think route."""
from __future__ import annotations

from typing import Sequence

import torch


A_RESCUE_REDUCTION = "one_group_level_A_distribution"


def uniform_positive_soft_ce(
    logits: torch.Tensor, target_token_ids: Sequence[int]
) -> torch.Tensor:
    """Uniform cross entropy over a deduplicated positive token set."""
    if logits.ndim != 1:
        raise ValueError("A rescue logits must be one vocabulary vector")
    targets = sorted({int(token_id) for token_id in target_token_ids})
    if not targets:
        raise ValueError("A rescue target set cannot be empty")
    if targets[0] < 0 or targets[-1] >= logits.shape[0]:
        raise IndexError("A rescue target token ID outside vocabulary")
    index = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return -torch.log_softmax(logits, dim=-1).index_select(0, index).mean()
