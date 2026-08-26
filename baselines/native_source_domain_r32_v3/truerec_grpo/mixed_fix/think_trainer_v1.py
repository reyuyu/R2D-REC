"""Standalone shared-forward Think loss and memory-safe streaming backward."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (Path(__file__).resolve().parent, ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from batch_collator_v1 import collate_business_group  # noqa: E402
from rollout_runtime_v1 import BusinessGroupRollout, G  # noqa: E402
from think_rescue_v1 import uniform_positive_soft_ce  # noqa: E402
from think_runtime_v1 import assert_shared_a_prefix, build_think_runtime_plan  # noqa: E402
from truerec_grpo_trainer_v1 import (  # noqa: E402
    format_credit_tensors, gather_padded_action_logps_rows,
)
from truerec_loss_v1 import HPR_LAMBDA, frontier_ppo_loss, multi_positive_log_mass_loss  # noqa: E402


A_RESCUE_LAMBDA = 0.02
BC_HPR_LAMBDA = HPR_LAMBDA
VALID_MICROBATCH_SIZES = (1, 2)


@dataclass(frozen=True)
class ThinkIntegratedGroupLoss:
    semantic_frontier_loss: torch.Tensor
    format_loss: torch.Tensor
    frontier_total_loss: torch.Tensor
    a_rescue_loss_raw: torch.Tensor
    a_rescue_loss_weighted: torch.Tensor
    bc_hpr_loss_raw: torch.Tensor
    bc_hpr_loss_weighted: torch.Tensor
    total_loss: torch.Tensor
    monitoring: dict[str, float | int | bool]


@dataclass(frozen=True)
class ThinkStreamingBackward:
    semantic_frontier_value: float
    format_value: float
    frontier_total_value: float
    a_rescue_value_raw: float
    a_rescue_value_weighted: float
    bc_hpr_value_raw: float
    bc_hpr_value_weighted: float
    total_value: float
    monitoring: dict[str, float | int | bool]


class ThinkGRPOTrainerV1:
    def __init__(
        self, policy, token_to_id, pad_token_id: int, *, epsilon: float = 0.2,
        padding_side: str = "right", device=None, streaming_microbatch_size: int = 2,
    ):
        if streaming_microbatch_size not in VALID_MICROBATCH_SIZES:
            raise ValueError(f"streaming microbatch must be one of {VALID_MICROBATCH_SIZES}")
        self.policy = policy
        self.token_to_id = token_to_id
        self.pad_token_id = int(pad_token_id)
        self.epsilon = float(epsilon)
        self.padding_side = padding_side
        self.device = device
        self.streaming_microbatch_size = int(streaming_microbatch_size)
        self.physical_policy_forward_calls = 0
        self.streaming_backward_calls = 0
        self.a_rescue_extra_forward_calls = 0
        self.bc_hpr_extra_forward_calls = 0

    def _prepare(self, group: BusinessGroupRollout):
        batch = collate_business_group(group, self.pad_token_id, self.padding_side)
        assert_shared_a_prefix(batch)
        metrics = [candidate.metrics for candidate in group.candidates]
        runtime = build_think_runtime_plan(metrics, group.all_gold_abc, self.token_to_id)
        format_credits, format_mask = format_credit_tensors(group)
        return batch, metrics, runtime, format_credits, format_mask

    def _forward_chunk(self, batch, start: int, stop: int):
        input_ids = batch.input_ids[start:stop]
        attention_mask = batch.attention_mask[start:stop]
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        output = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        self.physical_policy_forward_calls += 1
        return output.logits if hasattr(output, "logits") else output

    def _frontier_chunk(self, logits, batch, runtime, format_credits, format_mask, start, stop):
        current = gather_padded_action_logps_rows(logits, batch, start, stop)
        old = batch.old_logps[start:stop].to(device=current.device, dtype=current.dtype)
        semantic = frontier_ppo_loss(
            current.unsqueeze(0), old.unsqueeze(0),
            runtime.token_credits[start:stop].to(current).unsqueeze(0),
            runtime.token_credit_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
        )
        fmt = frontier_ppo_loss(
            current.unsqueeze(0), old.unsqueeze(0),
            format_credits[start:stop].to(current).unsqueeze(0),
            format_mask[start:stop].to(current.device).unsqueeze(0), self.epsilon,
        )
        scale = (stop - start) / G
        return semantic.loss * scale, fmt.loss * scale

    @staticmethod
    def _bc_hpr_chunk(logits, batch, runtime, start: int, stop: int):
        zero = logits.sum() * 0.0
        if not runtime.bc_hpr.sites:
            return zero
        contribution = zero
        site_count = len(runtime.bc_hpr.sites)
        for site in runtime.bc_hpr.sites:
            weight = 1.0 / (site_count * len(site.onpolicy_positions))
            for sample_index, action_position in site.onpolicy_positions:
                if start <= sample_index < stop:
                    causal = int(batch.causal_logit_indices[sample_index, action_position])
                    contribution = contribution + weight * multi_positive_log_mass_loss(
                        logits[sample_index - start, causal], site.target_token_ids,
                    )
        return contribution

    @staticmethod
    def _a_rescue_chunk(logits, batch, runtime, start: int, stop: int):
        zero = logits.sum() * 0.0
        if not runtime.a_rescue.triggered or not (start <= 0 < stop):
            return zero
        causal = int(batch.causal_logit_indices[0, 0])
        return uniform_positive_soft_ce(logits[0 - start, causal], runtime.a_rescue.target_token_ids)

    @staticmethod
    def _monitor(metrics, runtime, values: dict[str, float], forwards: int, backwards: int):
        monitor: dict[str, Any] = dict(runtime.monitoring)
        monitor["format_valid_rate"] = sum(bool(item["format_valid"]) for item in metrics) / G
        monitor.update(values)
        monitor.update({
            "physical_forward_calls": forwards,
            "streaming_backward_calls": backwards,
            "A_RESCUE_EXTRA_FORWARD_CALLS": 0,
            "BC_HPR_EXTRA_FORWARD_CALLS": 0,
        })
        return monitor

    def compute_group(self, group: BusinessGroupRollout) -> ThinkIntegratedGroupLoss:
        """Reference objective; retains chunk graphs until the returned loss is backpropagated."""
        batch, metrics, runtime, format_credits, format_mask = self._prepare(group)
        semantic_parts, format_parts, a_parts, hpr_parts = [], [], [], []
        calls_before = self.physical_policy_forward_calls
        for start in range(0, G, self.streaming_microbatch_size):
            stop = min(start + self.streaming_microbatch_size, G)
            logits = self._forward_chunk(batch, start, stop)
            semantic, fmt = self._frontier_chunk(
                logits, batch, runtime, format_credits, format_mask, start, stop,
            )
            semantic_parts.append(semantic)
            format_parts.append(fmt)
            a_parts.append(self._a_rescue_chunk(logits, batch, runtime, start, stop))
            hpr_parts.append(self._bc_hpr_chunk(logits, batch, runtime, start, stop))
        semantic = torch.stack(semantic_parts).sum()
        fmt = torch.stack(format_parts).sum()
        a_raw = torch.stack(a_parts).sum()
        hpr_raw = torch.stack(hpr_parts).sum()
        a_weighted = A_RESCUE_LAMBDA * a_raw
        hpr_weighted = BC_HPR_LAMBDA * hpr_raw
        frontier_total = semantic + fmt
        total = frontier_total + a_weighted + hpr_weighted
        values = {
            "semantic_frontier_loss": float(semantic.detach()),
            "format_loss": float(fmt.detach()),
            "frontier_total_loss": float(frontier_total.detach()),
            "a_rescue_loss_raw": float(a_raw.detach()),
            "a_rescue_loss_weighted": float(a_weighted.detach()),
            "bc_hpr_loss_raw": float(hpr_raw.detach()),
            "bc_hpr_loss_weighted": float(hpr_weighted.detach()),
            "total_loss": float(total.detach()),
        }
        monitor = self._monitor(metrics, runtime, values, self.physical_policy_forward_calls - calls_before, 0)
        return ThinkIntegratedGroupLoss(
            semantic, fmt, frontier_total, a_raw, a_weighted, hpr_raw, hpr_weighted, total, monitor,
        )

    def backward_group_streaming(self, group: BusinessGroupRollout) -> ThinkStreamingBackward:
        """Backpropagate each exact G8 chunk contribution and release its graph immediately."""
        batch, metrics, runtime, format_credits, format_mask = self._prepare(group)
        values = {key: 0.0 for key in (
            "semantic_frontier_loss", "format_loss", "a_rescue_loss_raw",
            "bc_hpr_loss_raw", "backpropagated_chunk_total_loss",
        )}
        calls_before = self.physical_policy_forward_calls
        backwards_before = self.streaming_backward_calls
        for start in range(0, G, self.streaming_microbatch_size):
            stop = min(start + self.streaming_microbatch_size, G)
            logits = self._forward_chunk(batch, start, stop)
            semantic, fmt = self._frontier_chunk(
                logits, batch, runtime, format_credits, format_mask, start, stop,
            )
            a_raw = self._a_rescue_chunk(logits, batch, runtime, start, stop)
            hpr_raw = self._bc_hpr_chunk(logits, batch, runtime, start, stop)
            chunk_total = semantic + fmt + A_RESCUE_LAMBDA * a_raw + BC_HPR_LAMBDA * hpr_raw
            values["semantic_frontier_loss"] += float(semantic.detach())
            values["format_loss"] += float(fmt.detach())
            values["a_rescue_loss_raw"] += float(a_raw.detach())
            values["bc_hpr_loss_raw"] += float(hpr_raw.detach())
            values["backpropagated_chunk_total_loss"] += float(chunk_total.detach())
            chunk_total.backward()
            self.streaming_backward_calls += 1
            del chunk_total, hpr_raw, a_raw, fmt, semantic, logits
        semantic = values["semantic_frontier_loss"]
        fmt = values["format_loss"]
        a_raw = values["a_rescue_loss_raw"]
        hpr_raw = values["bc_hpr_loss_raw"]
        a_weighted = A_RESCUE_LAMBDA * a_raw
        hpr_weighted = BC_HPR_LAMBDA * hpr_raw
        frontier_total = semantic + fmt
        total = frontier_total + a_weighted + hpr_weighted
        detached = {
            "semantic_frontier_loss": semantic, "format_loss": fmt,
            "frontier_total_loss": frontier_total, "a_rescue_loss_raw": a_raw,
            "a_rescue_loss_weighted": a_weighted, "bc_hpr_loss_raw": hpr_raw,
            "bc_hpr_loss_weighted": hpr_weighted, "total_loss": total,
        }
        forwards = self.physical_policy_forward_calls - calls_before
        backwards = self.streaming_backward_calls - backwards_before
        monitor = self._monitor(metrics, runtime, detached, forwards, backwards)
        monitor["backpropagated_chunk_total_loss"] = values["backpropagated_chunk_total_loss"]
        monitor["monitoring_composition_roundoff_abs"] = abs(values["backpropagated_chunk_total_loss"] - total)
        return ThinkStreamingBackward(
            semantic, fmt, frontier_total, a_raw, a_weighted, hpr_raw, hpr_weighted, total, monitor,
        )
