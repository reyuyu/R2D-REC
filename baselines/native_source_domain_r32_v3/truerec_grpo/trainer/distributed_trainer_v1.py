"""Four-rank local-G2 streaming backward for one preserved global TrueRec G8 group."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
import torch.distributed as dist

from batch_collator_v1 import collate_business_group
from rollout_runtime_v1 import BusinessGroupRollout, G
from truerec_grpo_trainer_v1 import format_credit_tensors, gather_padded_action_logps_rows
from truerec_loss_v1 import HPR_LAMBDA, frontier_ppo_loss, multi_positive_log_mass_loss
from truerec_runtime_v1 import build_group_runtime_plan


DDP_WORLD_SIZE = 4
LOCAL_G = 2


@dataclass(frozen=True)
class DistributedStreamingBackward:
    global_frontier_value: float
    global_hpr_value_raw: float
    global_hpr_value_weighted: float
    global_total_value: float
    local_current_logps: tuple[tuple[float, ...], ...]
    physical_forward_calls: int
    physical_backward_calls: int
    hpr_extra_forward_calls: int
    runtime_plan_hash: str


def require_four_rank_group() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed TrueRec requires an initialized process group")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != DDP_WORLD_SIZE:
        raise RuntimeError(f"distributed TrueRec requires world_size={DDP_WORLD_SIZE}")
    return rank, world_size


def runtime_plan_sha256(runtime) -> str:
    value = {
        "token_credits": runtime.token_credits.tolist(),
        "token_credit_mask": runtime.token_credit_mask.tolist(),
        "hpr": {
            "trigger": runtime.hpr.trigger,
            "sites": [
                {
                    "level": site.level,
                    "target_token_ids": list(site.target_token_ids),
                    "onpolicy_positions": [list(item) for item in site.onpolicy_positions],
                }
                for site in runtime.hpr.sites
            ],
        },
        "monitoring": runtime.monitoring,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def all_rank_values_equal(value: Any) -> bool:
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return all(item == values[0] for item in values)


def distributed_fail_if(local_failure: bool, message: str, device: torch.device) -> None:
    flag = torch.tensor(int(local_failure), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()):
        raise RuntimeError(message)


class DistributedTrueRecGRPOTrainerV1:
    """Backpropagate this rank's globally normalized contribution with DDP scaling."""

    def __init__(
        self, policy, token_to_id, pad_token_id: int, *, device: torch.device,
        streaming_microbatch_size: int, epsilon: float = 0.2,
    ) -> None:
        if streaming_microbatch_size not in (1, 2):
            raise ValueError("streaming_microbatch_size must be 1 or 2")
        self.policy = policy
        self.token_to_id = token_to_id
        self.pad_token_id = int(pad_token_id)
        self.device = device
        self.streaming_microbatch_size = int(streaming_microbatch_size)
        self.epsilon = float(epsilon)

    def backward_global_group(self, group: BusinessGroupRollout) -> DistributedStreamingBackward:
        rank, world_size = require_four_rank_group()
        if len(group.candidates) != G:
            raise ValueError("distributed trainer requires one global G8 group")
        local_start, local_stop = rank * LOCAL_G, (rank + 1) * LOCAL_G
        batch = collate_business_group(group, self.pad_token_id, "right")
        runtime = build_group_runtime_plan(
            [candidate.metrics for candidate in group.candidates], group.all_gold_abc, self.token_to_id,
        )
        plan_hash = runtime_plan_sha256(runtime)
        if not all_rank_values_equal(plan_hash):
            raise RuntimeError("global runtime plan hash differs across ranks")
        format_credits, format_mask = format_credit_tensors(group)
        site_count = len(runtime.hpr.sites)
        local_frontier_value = 0.0
        local_hpr_value = 0.0
        current_rows: list[tuple[float, ...]] = []
        forward_calls = backward_calls = 0
        mb = self.streaming_microbatch_size
        for start in range(local_start, local_stop, mb):
            stop = min(start + mb, local_stop)
            input_ids = batch.input_ids[start:stop].to(self.device)
            attention_mask = batch.attention_mask[start:stop].to(self.device)
            output = self.policy(input_ids=input_ids, attention_mask=attention_mask)
            forward_calls += 1
            logits = output.logits if hasattr(output, "logits") else output
            current = gather_padded_action_logps_rows(logits, batch, start, stop)
            for local_row, candidate in enumerate(group.candidates[start:stop]):
                current_rows.append(tuple(float(value) for value in current[local_row, :len(candidate.completion_ids)].detach().float().cpu()))
            old_logps = batch.old_logps[start:stop].to(current.device)
            hierarchy = frontier_ppo_loss(
                current.unsqueeze(0), old_logps.unsqueeze(0),
                runtime.token_credits[start:stop].to(current.device).unsqueeze(0),
                runtime.token_credit_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
            )
            format_loss = frontier_ppo_loss(
                current.unsqueeze(0), old_logps.unsqueeze(0),
                format_credits[start:stop].to(current.device).unsqueeze(0),
                format_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
            )
            local_frontier = (hierarchy.loss + format_loss.loss) * ((stop - start) / G)
            local_hpr = logits.sum() * 0.0
            if site_count:
                for site in runtime.hpr.sites:
                    weight = 1.0 / (site_count * len(site.onpolicy_positions))
                    for sample_index, action_position in site.onpolicy_positions:
                        if start <= sample_index < stop:
                            causal_index = int(batch.causal_logit_indices[sample_index, action_position])
                            local_hpr = local_hpr + weight * multi_positive_log_mass_loss(
                                logits[sample_index - start, causal_index], site.target_token_ids,
                            )
            local_total = local_frontier + HPR_LAMBDA * local_hpr
            local_frontier_value += float(local_frontier.detach())
            local_hpr_value += float(local_hpr.detach())
            (world_size * local_total).backward()
            backward_calls += 1
            del local_total, local_hpr, local_frontier, format_loss, hierarchy
            del old_logps, current, logits, output, attention_mask, input_ids
        values = torch.tensor([local_frontier_value, local_hpr_value], dtype=torch.float64, device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        global_frontier, global_hpr = (float(value) for value in values.cpu())
        return DistributedStreamingBackward(
            global_frontier, global_hpr, HPR_LAMBDA * global_hpr,
            global_frontier + HPR_LAMBDA * global_hpr,
            tuple(current_rows), forward_calls, backward_calls, 0, plan_hash,
        )
