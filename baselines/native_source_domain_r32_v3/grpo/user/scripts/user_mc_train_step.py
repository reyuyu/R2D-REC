"""Minimal one-step optimizer bridge for MC_USER_v1 correctness checks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from user_mc_policy import compute_mc_model_loss


class MCTrainStepError(RuntimeError):
    pass


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise MCTrainStepError("model has no trainable parameters")
    return parameters


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise MCTrainStepError("gradient contains NaN or Inf")
        squared_norm += float(gradient.double().square().sum())
    grad_norm = math.sqrt(squared_norm)
    if not math.isfinite(grad_norm):
        raise MCTrainStepError("gradient norm is NaN or Inf")
    return grad_norm


def _parameter_delta(
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
) -> tuple[float, float]:
    squared_norm = 0.0
    max_abs = 0.0
    for parameter, initial in zip(parameters, before):
        delta = parameter.detach() - initial
        squared_norm += float(delta.double().square().sum())
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max()))
    return math.sqrt(squared_norm), max_abs


def mc_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    credit_units_per_candidate: Sequence[Sequence[Mapping[str, Any]]],
    forward_batch_size: int = 1,
) -> dict[str, Any]:
    """Run exactly one finite-guarded MC backward and optimizer step."""

    optimizer.zero_grad(set_to_none=True)
    parameters = _trainable_parameters(model)
    before = [parameter.detach().clone() for parameter in parameters]
    loss, objective_metadata, per_token_logps = compute_mc_model_loss(
        model,
        batch,
        credit_units_per_candidate,
        forward_batch_size=forward_batch_size,
    )
    if not bool(torch.isfinite(loss).all()):
        raise MCTrainStepError("loss is NaN or Inf")
    if not bool(torch.isfinite(per_token_logps).all()):
        raise MCTrainStepError("completion log probability is NaN or Inf")

    loss.backward()
    grad_norm = _gradient_norm(parameters)
    optimizer.step()
    parameter_delta_l2, parameter_delta_max_abs = _parameter_delta(
        parameters, before
    )
    return {
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "finite": True,
        "objective_metadata": objective_metadata,
        "parameter_delta_l2": parameter_delta_l2,
        "parameter_delta_max_abs": parameter_delta_max_abs,
    }


__all__ = ["MCTrainStepError", "mc_optimizer_step"]
