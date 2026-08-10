"""REC-PU Phase 1: per-SID-position positive-unlabeled loss.

This module is intentionally independent from dataset metadata, packing, and
Trainer code. It receives a current SID layer vocabulary and an observed
positive subset for one decoder position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch


_LEVEL_INDEX_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


@dataclass(frozen=True)
class RecPUMasks:
    """Disjoint P/U/O masks over the final vocabulary dimension."""

    positive: torch.Tensor
    unlabeled: torch.Tensor
    other: torch.Tensor


def _normalise_ids(ids: Iterable[int], name: str, vocab_size: int) -> tuple[int, ...]:
    result = tuple(sorted({int(token_id) for token_id in ids}))
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if result[0] < 0 or result[-1] >= vocab_size:
        raise ValueError(f"{name} contains an id outside [0, {vocab_size}).")
    return result


def build_rec_pu_masks(
    vocab_size: int,
    positive_ids: Iterable[int],
    same_level_ids: Iterable[int],
    *,
    device: torch.device | None = None,
) -> RecPUMasks:
    """Build P/U/O masks for one SID component layer.

    ``same_level_ids`` must contain only tokens from the component currently
    predicted (all ``s_a`` *or* all ``s_b`` *or* all ``s_c`` ids). Therefore
    b/c/ordinary vocabulary tokens remain O when predicting a.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    positives = _normalise_ids(positive_ids, "positive_ids", vocab_size)
    same_level = _normalise_ids(same_level_ids, "same_level_ids", vocab_size)
    if not set(positives).issubset(same_level):
        raise ValueError("positive_ids must be a subset of same_level_ids.")

    positive = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    level = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    positive[list(positives)] = True
    level[list(same_level)] = True
    unlabeled = level & ~positive
    other = ~level
    return RecPUMasks(positive=positive, unlabeled=unlabeled, other=other)


def attenuate_unlabeled_logits(logits: torch.Tensor, unlabeled_mask: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    """Keep logits forward-identical while reducing only U backward gradients.

    For U this is mathematically equivalent to
    ``beta * z + (1 - beta) * z.detach()``. The rearranged expression avoids
    floating-point roundoff, so its forward value is bit-identical to ``z``.
    """

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if logits.size(-1) != unlabeled_mask.numel():
        raise ValueError("unlabeled_mask length must match logits.size(-1).")
    mask = unlabeled_mask.to(device=logits.device, dtype=torch.bool)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)
    attenuated = logits.detach() + beta * (logits - logits.detach())
    return torch.where(mask, attenuated, logits)


def _cached_level_indices(token_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Return a device-local level index without retaining a vocab-sized mask."""

    key = (str(device), token_ids)
    indices = _LEVEL_INDEX_CACHE.get(key)
    if indices is None:
        indices = torch.tensor(token_ids, dtype=torch.long, device=device)
        _LEVEL_INDEX_CACHE[key] = indices
    return indices


class _RecPUPositionLoss(torch.autograd.Function):
    """Memory-efficient autograd equivalent of the detached-logit construction.

    The earlier direct implementation materialized ``z_tilde`` and a full-vocab
    mask for every selected SID position.  This function leaves the forward
    calculation on the original logits (which is exactly identical) and applies
    the same U-only chain-rule factor in backward.  It keeps no per-position
    vocab-sized transformed tensor or mask alive in the autograd graph.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        same_level_ids: tuple[int, ...],
        positive_ids: tuple[int, ...],
        beta: float,
    ) -> torch.Tensor:
        # Saving a 1-D view here would retain the complete packed-logits
        # storage. Save only this position's probability vector instead.
        probabilities = torch.softmax(logits, dim=-1)
        ctx.save_for_backward(probabilities)
        ctx.same_level_ids = same_level_ids
        ctx.positive_ids = positive_ids
        ctx.beta = beta
        positive_index = torch.tensor(positive_ids, dtype=torch.long, device=logits.device)
        return torch.logsumexp(logits, dim=-1) - logits.index_select(-1, positive_index).mean(dim=-1)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        (probabilities,) = ctx.saved_tensors
        grad = probabilities
        while grad_output.ndim < grad.ndim:
            grad_output = grad_output.unsqueeze(-1)
        grad = grad * grad_output

        level_index = _cached_level_indices(ctx.same_level_ids, probabilities.device)
        positive_index = torch.tensor(ctx.positive_ids, dtype=torch.long, device=probabilities.device)
        # Same-level tokens first receive beta; observed positives are then
        # restored to their unattenuated CE/multi-positive objective gradient.
        grad.index_copy_(-1, level_index, grad.index_select(-1, level_index) * ctx.beta)
        positive_grad = probabilities.index_select(-1, positive_index)
        positive_grad = positive_grad - (1.0 / len(ctx.positive_ids))
        grad.index_copy_(-1, positive_index, positive_grad * grad_output)
        return grad, None, None, None


class _RecPUBatchedPositionLoss(torch.autograd.Function):
    """Vectorized REC-PU positions for one SID component level.

    The production path supplies ``[K, vocab]`` logits from one advanced index
    operation per level.  It retains those compact selected logits and small
    ragged-positive index tensors only; probabilities are recomputed in
    backward, so neither a per-position probability vector nor a full-vocab
    P/U mask is kept alive.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        level_ids: torch.Tensor,
        positive_rows: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_counts: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(logits, positive_rows, positive_ids, positive_counts)
        ctx.level_ids = level_ids
        ctx.beta = beta
        positive_values = logits[positive_rows, positive_ids]
        positive_sums = torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)
        positive_sums.index_add_(0, positive_rows, positive_values)
        return torch.logsumexp(logits, dim=-1) - positive_sums / positive_counts.to(dtype=logits.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        logits, positive_rows, positive_ids, positive_counts = ctx.saved_tensors
        probabilities = torch.softmax(logits, dim=-1)
        gradient = probabilities * grad_output.reshape(-1, 1).to(dtype=probabilities.dtype)

        # All same-level SID candidates receive beta in the denominator, then
        # sparse positive entries are overwritten with their normal objective
        # gradient p - 1/|P|.
        gradient[:, ctx.level_ids] *= ctx.beta
        positive_gradient = probabilities[positive_rows, positive_ids]
        positive_gradient = positive_gradient - positive_counts[positive_rows].reciprocal().to(
            dtype=probabilities.dtype
        )
        gradient[positive_rows, positive_ids] = positive_gradient * grad_output[positive_rows].to(
            dtype=probabilities.dtype
        )
        return gradient, None, None, None, None, None


def rec_pu_position_loss(
    logits: torch.Tensor,
    positive_ids: Iterable[int],
    same_level_ids: Iterable[int],
    *,
    beta: float = 0.05,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[torch.Tensor, RecPUMasks]:
    """Compute ``logsumexp(z_tilde_all) - mean(z_tilde_P)`` at SID positions.

    The final dimension is vocabulary. Leading dimensions are independent
    positions/batches. This Phase 1 API deliberately has no packing or
    metadata assumptions.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension.")
    masks = build_rec_pu_masks(
        logits.size(-1), positive_ids, same_level_ids, device=logits.device
    )
    positive_ids = _normalise_ids(positive_ids, "positive_ids", logits.size(-1))
    same_level_ids = _normalise_ids(same_level_ids, "same_level_ids", logits.size(-1))
    per_position = _RecPUPositionLoss.apply(logits, same_level_ids, positive_ids, beta)
    if reduction == "none":
        return per_position, masks
    if reduction == "mean":
        return per_position.mean(), masks
    raise ValueError(f"Unsupported reduction {reduction!r}.")


def rec_pu_batched_position_loss(
    logits: torch.Tensor,
    positive_ids_by_position: Sequence[Iterable[int]],
    same_level_ids: Iterable[int],
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Compute REC-PU for K positions of one SID level without vocab masks.

    ``logits`` is ``[K, vocab]``. Positive sets are represented sparsely as
    flattened token ids with their owning row indices; they may differ per
    position because prefix-conditioned multi-positive candidate sets differ.
    """

    if logits.ndim != 2:
        raise ValueError("Batched REC-PU logits must have shape [positions, vocab].")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if len(positive_ids_by_position) != logits.size(0):
        raise ValueError("positive_ids_by_position must have one set per logits row.")
    level = _normalise_ids(same_level_ids, "same_level_ids", logits.size(-1))
    rows: list[int] = []
    flat_positive_ids: list[int] = []
    counts: list[float] = []
    level_set = set(level)
    for row_index, raw_positive_ids in enumerate(positive_ids_by_position):
        positives = _normalise_ids(raw_positive_ids, "positive_ids", logits.size(-1))
        if not set(positives).issubset(level_set):
            raise ValueError("positive_ids must be a subset of same_level_ids.")
        rows.extend([row_index] * len(positives))
        flat_positive_ids.extend(positives)
        counts.append(float(len(positives)))

    device = logits.device
    return _RecPUBatchedPositionLoss.apply(
        logits,
        _cached_level_indices(level, device),
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(flat_positive_ids, dtype=torch.long, device=device),
        torch.tensor(counts, dtype=logits.dtype, device=device),
        beta,
    )
