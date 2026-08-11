"""Set-PU loss for recommendation final SID components.

The loss is an ordinary PyTorch scalar objective.  It deliberately contains
no custom autograd function or detached-logit surrogate: its backward is the
true derivative of its forward value.
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
    """Build mutually exclusive P/U/O masks for one SID component layer."""

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


def _cached_level_indices(token_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    key = (str(device), token_ids)
    indices = _LEVEL_INDEX_CACHE.get(key)
    if indices is None:
        indices = torch.tensor(token_ids, dtype=torch.long, device=device)
        _LEVEL_INDEX_CACHE[key] = indices
    return indices


def _set_pu_per_position(
    logits: torch.Tensor,
    positive_ids_by_position: Sequence[tuple[int, ...]],
    level_ids: tuple[int, ...],
    alpha: float,
) -> torch.Tensor:
    """Return ``log D - logsumexp(z[P])`` with direct autograd.

    ``D`` keeps P and O at their original logits and applies ``log(alpha)``
    only to U.  ``non_u`` and U are reduced separately to avoid a full
    per-position modified-vocabulary copy.
    """

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1].")
    vocab_size = logits.size(-1)
    level_index = _cached_level_indices(level_ids, logits.device)
    # U differs per row because P is prefix-conditioned.  K is the number of
    # final SID positions, normally small, so a compact row loop avoids a
    # long-lived K x vocab transformed logits tensor while remaining pure
    # PyTorch autograd.
    terms: list[torch.Tensor] = []
    log_alpha = None if alpha == 0.0 else logits.new_tensor(alpha).log()
    for row, positives in enumerate(positive_ids_by_position):
        positive_index = torch.tensor(positives, dtype=torch.long, device=logits.device)
        is_positive_in_level = torch.isin(level_index, positive_index)
        u_index = level_index[~is_positive_in_level]
        # P and O form the unmodified denominator branch.  The boolean mask
        # is local to this selected position and never uses stop-gradient.
        non_u_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
        non_u_mask[u_index] = False
        log_non_u = torch.logsumexp(logits[row].masked_fill(~non_u_mask, float("-inf")), dim=-1)
        if u_index.numel() == 0 or alpha == 0.0:
            log_den = log_non_u
        else:
            log_u = torch.logsumexp(logits[row].index_select(0, u_index), dim=-1)
            log_den = torch.logaddexp(log_non_u, log_alpha + log_u)
        log_pos = torch.logsumexp(logits[row].index_select(0, positive_index), dim=-1)
        terms.append(log_den - log_pos)
    return torch.stack(terms)


def rec_pu_position_loss(
    logits: torch.Tensor,
    positive_ids: Iterable[int],
    same_level_ids: Iterable[int],
    *,
    beta: float = 0.05,
    alpha: float | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[torch.Tensor, RecPUMasks]:
    """Set-PU for independent positions; ``beta`` remains an old-name alias.

    The forward objective is ``log(sum_P exp(z)+alpha*sum_U exp(z)+sum_O
    exp(z)) - logsumexp(z[P])`` and ordinary autograd supplies its gradient.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension.")
    alpha = beta if alpha is None else alpha
    masks = build_rec_pu_masks(logits.size(-1), positive_ids, same_level_ids, device=logits.device)
    positives = _normalise_ids(positive_ids, "positive_ids", logits.size(-1))
    level = _normalise_ids(same_level_ids, "same_level_ids", logits.size(-1))
    flat = logits.reshape(-1, logits.size(-1))
    per_position = _set_pu_per_position(flat, [positives] * flat.size(0), level, alpha).reshape(logits.shape[:-1])
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
    alpha: float | None = None,
) -> torch.Tensor:
    """Vectorized-interface Set-PU values for ``[positions, vocab]`` logits.

    ``beta`` remains accepted for old YAML/launcher compatibility; semantically
    it is now the scalar denominator weight ``alpha``, not a gradient scale.
    """

    if logits.ndim != 2:
        raise ValueError("Batched REC-PU logits must have shape [positions, vocab].")
    alpha = beta if alpha is None else alpha
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if len(positive_ids_by_position) != logits.size(0):
        raise ValueError("positive_ids_by_position must have one set per logits row.")
    level = _normalise_ids(same_level_ids, "same_level_ids", logits.size(-1))
    level_set = set(level)
    positives_by_row = []
    for raw_positive_ids in positive_ids_by_position:
        positives = _normalise_ids(raw_positive_ids, "positive_ids", logits.size(-1))
        if not set(positives).issubset(level_set):
            raise ValueError("positive_ids must be a subset of same_level_ids.")
        positives_by_row.append(positives)
    return _set_pu_per_position(logits, positives_by_row, level, alpha)
