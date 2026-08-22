"""Four-rank candidate-parallel optimizer step for the MC_USER hybrid loss."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from user_mc_hybrid_objective import ADVANTAGE_EPSILON
from user_mc_objective import mc_unit_credit_loss
from user_mc_policy import get_mc_completion_logps


HYBRID_WORLD_SIZE = 4


class MCHybridDistributedError(RuntimeError):
    pass


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared += parameter.grad.detach().double().pow(2).sum().cpu()
    value = float(torch.sqrt(squared)) if found else 0.0
    if not math.isfinite(value):
        raise MCHybridDistributedError("non-finite distributed gradient norm")
    return value


def distributed_hybrid_optimizer_step(
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    local_batch: Mapping[str, torch.Tensor],
    local_reward: float | torch.Tensor,
    local_credit_units: Sequence[Sequence[Mapping[str, Any]]],
    *,
    sequence_weight: float = 1.0,
    local_weight: float = 0.3,
    forward_batch_size: int = 1,
) -> dict[str, Any]:
    """Run one K4 hybrid update with one candidate on each DDP rank."""

    if not dist.is_initialized() or dist.get_world_size() != HYBRID_WORLD_SIZE:
        raise MCHybridDistributedError("hybrid distributed step requires world_size == 4")
    if not isinstance(ddp_model, DistributedDataParallel):
        raise MCHybridDistributedError("training model must be DistributedDataParallel")
    if local_batch["completion_ids"].shape[0] != 1:
        raise MCHybridDistributedError("each rank must contain exactly one candidate")
    if len(local_credit_units) != 1:
        raise MCHybridDistributedError("local credit units must contain one candidate")
    if not math.isfinite(sequence_weight) or not math.isfinite(local_weight):
        raise MCHybridDistributedError("hybrid weights must be finite")

    optimizer.zero_grad(set_to_none=True)
    per_token_logps = get_mc_completion_logps(
        ddp_model,
        local_batch["prompt_ids"],
        local_batch["prompt_mask"],
        local_batch["completion_ids"],
        local_batch["completion_mask"],
        forward_batch_size=forward_batch_size,
    )
    if per_token_logps.shape[0] != 1 or not bool(torch.isfinite(per_token_logps).all()):
        raise MCHybridDistributedError("non-finite or non-local completion logps")

    reward = torch.as_tensor(
        local_reward,
        dtype=per_token_logps.dtype,
        device=per_token_logps.device,
    ).reshape(())
    if not bool(torch.isfinite(reward)):
        raise MCHybridDistributedError("local reward must be finite")
    reward = reward.detach()
    gathered_rewards = [torch.zeros_like(reward) for _ in range(HYBRID_WORLD_SIZE)]
    dist.all_gather(gathered_rewards, reward)
    group_rewards_tensor = torch.stack(gathered_rewards).detach()
    reward_mean = group_rewards_tensor.mean()
    reward_std = group_rewards_tensor.std(unbiased=False)
    if float(reward_std) < ADVANTAGE_EPSILON:
        advantages = torch.zeros_like(group_rewards_tensor)
    else:
        advantages = (
            group_rewards_tensor - reward_mean
        ) / (reward_std + ADVANTAGE_EPSILON)
    advantage = advantages[dist.get_rank()].detach()

    completion_mask = local_batch["completion_mask"].to(per_token_logps.dtype)
    real_token_count = completion_mask.sum()
    if float(real_token_count) <= 0.0:
        raise MCHybridDistributedError("local completion must contain a real token")
    completion_mean_logp = (per_token_logps * completion_mask).sum() / real_token_count
    sequence_loss = -advantage * completion_mean_logp
    local_loss, local_metadata = mc_unit_credit_loss(
        per_token_logps,
        local_credit_units,
        local_batch["completion_mask"],
    )
    total_loss = sequence_weight * sequence_loss + local_weight * local_loss
    if not bool(torch.isfinite(total_loss)):
        raise MCHybridDistributedError("non-finite distributed hybrid loss")

    local_has_signal = bool(float(advantage.abs()) > 0.0) or int(
        local_metadata["active_unit_count"]
    ) > 0
    global_signal_count = torch.tensor(
        int(local_has_signal),
        dtype=torch.long,
        device=per_token_logps.device,
    )
    dist.all_reduce(global_signal_count, op=dist.ReduceOp.SUM)
    if int(global_signal_count) == 0:
        return {
            "local_reward": float(reward),
            "group_rewards": [float(item) for item in group_rewards_tensor],
            "group_reward_mean": float(reward_mean),
            "group_reward_std": float(reward_std),
            "group_reward_spread": float(
                group_rewards_tensor.max() - group_rewards_tensor.min()
            ),
            "sequence_advantage": float(advantage),
            "sequence_loss": float(sequence_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "local_active_unit_count": int(local_metadata["active_unit_count"]),
            "local_objective_metadata": local_metadata,
            "global_signal_rank_count": 0,
            "optimizer_step_performed": False,
            "skipped_update": True,
            "grad_norm": 0.0,
        }

    # Even a rank-local differentiable zero must participate in this collective.
    total_loss.backward()
    parameters = [
        parameter for parameter in ddp_model.module.parameters() if parameter.requires_grad
    ]
    grad_norm = _gradient_norm(parameters)
    grad_norm_bounds = torch.tensor(
        [grad_norm, grad_norm],
        dtype=torch.float64,
        device=per_token_logps.device,
    )
    dist.all_reduce(grad_norm_bounds[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(grad_norm_bounds[1:], op=dist.ReduceOp.MAX)
    if float(grad_norm_bounds[1] - grad_norm_bounds[0]) > 1e-6:
        raise MCHybridDistributedError("DDP hybrid gradient norms diverged")
    optimizer.step()
    return {
        "local_reward": float(reward),
        "group_rewards": [float(item) for item in group_rewards_tensor],
        "group_reward_mean": float(reward_mean),
        "group_reward_std": float(reward_std),
        "group_reward_spread": float(
            group_rewards_tensor.max() - group_rewards_tensor.min()
        ),
        "sequence_advantage": float(advantage),
        "sequence_loss": float(sequence_loss.detach()),
        "local_loss": float(local_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "local_active_unit_count": int(local_metadata["active_unit_count"]),
        "local_objective_metadata": local_metadata,
        "global_signal_rank_count": int(global_signal_count),
        "optimizer_step_performed": True,
        "skipped_update": False,
        "grad_norm": grad_norm,
    }


__all__ = [
    "HYBRID_WORLD_SIZE",
    "MCHybridDistributedError",
    "distributed_hybrid_optimizer_step",
]
