"""Tiny-model-compatible policy/log-prob bridge for MC_USER_v1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from user_mc_objective import mc_unit_credit_loss


MC_TEMPERATURE = 0.9


def _validate_policy_inputs(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> None:
    tensors = (prompt_ids, prompt_mask, completion_ids, completion_mask)
    if not all(torch.is_tensor(tensor) and tensor.ndim == 2 for tensor in tensors):
        raise ValueError("MC policy inputs must all be rank-2 tensors")
    if prompt_ids.shape != prompt_mask.shape:
        raise ValueError("prompt_ids and prompt_mask must have identical shape")
    if completion_ids.shape != completion_mask.shape:
        raise ValueError("completion_ids and completion_mask must have identical shape")
    if prompt_ids.size(0) != completion_ids.size(0):
        raise ValueError("prompt and completion batch sizes differ")
    if prompt_ids.size(0) == 0 or prompt_ids.size(1) == 0 or completion_ids.size(1) == 0:
        raise ValueError("MC policy inputs must have non-empty batch, prompt, and completion axes")


def get_mc_completion_logps(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float = MC_TEMPERATURE,
) -> torch.Tensor:
    """Return causal log-probabilities for each completion token."""

    _validate_policy_inputs(prompt_ids, prompt_mask, completion_ids, completion_mask)
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("model must return full [B,P+T,V] logits")

    completion_length = completion_ids.size(1)
    shifted_completion_logits = logits[:, :-1, :][:, -completion_length:, :]
    log_probs = torch.log_softmax(shifted_completion_logits / temperature, dim=-1)
    return torch.gather(log_probs, 2, completion_ids.unsqueeze(-1)).squeeze(-1)


def compute_mc_model_loss(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    credit_units_per_candidate: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    """Bridge model logits to the existing pure MC unit-credit objective."""

    per_token_logps = get_mc_completion_logps(
        model,
        batch["prompt_ids"],
        batch["prompt_mask"],
        batch["completion_ids"],
        batch["completion_mask"],
        temperature=MC_TEMPERATURE,
    )
    loss, metadata = mc_unit_credit_loss(
        per_token_logps,
        credit_units_per_candidate,
        batch["completion_mask"],
    )
    return loss, metadata, per_token_logps


__all__ = ["MC_TEMPERATURE", "compute_mc_model_loss", "get_mc_completion_logps"]
