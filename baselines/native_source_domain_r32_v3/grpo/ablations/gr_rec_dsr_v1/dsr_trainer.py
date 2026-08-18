"""Isolated DSR trainer subclass. GR_REC_v1 files remain untouched."""
from __future__ import annotations

import json
import os
import time

import torch

from grpo_trl_trainer import ROUTE_ID, RecGRPOTrainer, route_multiplier

from .dsr_objectives import (
    nothink_unlikelihood_loss,
)
from .dsr_monitor import summarize_nothink_records, summarize_think_records
from .dsr_runtime import get_capture


class DsrGRPOTrainer(RecGRPOTrainer):
    """Adds independent Think and NoThink objectives to baseline GRPO."""

    def __init__(self, *args, **kwargs):
        self.dsr_think_lambda = float(kwargs.pop("dsr_think_lambda", os.environ.get("DSR_THINK_LAMBDA", "0.10")))
        self.dsr_nothink_scale = float(kwargs.pop("dsr_nothink_scale", os.environ.get("DSR_NOTHINK_SCALE", "1.0")))
        tokenizer = kwargs.get("processing_class")
        self._dsr_capture = kwargs.pop("dsr_capture", None) or get_capture(tokenizer)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_groups(records, group_size):
        if len(records) % group_size:
            raise RuntimeError(f"DSR records {len(records)} not divisible by G={group_size}")
        for start in range(0, len(records), group_size):
            ids = {item["group_id"] for item in records[start:start + group_size]}
            if len(ids) != 1:
                raise RuntimeError(f"DSR group mixes recommendation_group_id values: {sorted(ids)}")

    def _write_dsr_monitor(self, payload, traces):
        if not self._monitor_enabled() or self.accelerator.process_index != 0:
            return
        event = {
            "type": "dsr_rollout",
            "rollout_id": self._smoke_rollout_id,
            "step": self.state.global_step,
            **payload,
        }
        self._monitor._append("dsr_metrics.jsonl", event)
        if traces:
            self._monitor._append("dsr_traces.jsonl", {
                "type": "dsr_trace",
                "rollout_id": self._smoke_rollout_id,
                "step": self.state.global_step,
                "route": payload["route"],
                "candidates": traces,
            })

    def _generate_and_score_completions(self, inputs):
        from trl.trainer.grpo_trainer import gather_object

        self._dsr_capture.reset()
        output = super()._generate_and_score_completions(inputs)
        local_records = self._dsr_capture.records
        if len(local_records) != len(inputs):
            raise RuntimeError(f"DSR captured {len(local_records)} records for {len(inputs)} local inputs")
        gather_started = time.perf_counter()
        records = gather_object(local_records)
        gather_wall_sec = time.perf_counter() - gather_started
        route = inputs[0]["route"]
        group_size = 4 if route == "think" else 8
        self._validate_groups(records, group_size)
        local_count = len(local_records)
        process_slice = slice(
            self.accelerator.process_index * local_count,
            (self.accelerator.process_index + 1) * local_count,
        )
        device = output["completion_ids"].device
        payload = {
            "route": route,
            "tokenizer_audit": self._dsr_capture.tokenizer_audit,
            "dsr_record_gather_wall_sec": gather_wall_sec,
        }
        traces = []

        if route == "think":
            summary, scores, advantage_values = summarize_think_records(records)
            advantages = torch.tensor(advantage_values, dtype=torch.float32)
            output["dsr_aux_advantages"] = advantages[process_slice].to(device)
            output["dsr_sa_positions"] = torch.full((local_count,), -1, dtype=torch.long, device=device)
            output["dsr_frequency_weights"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            output["dsr_rescue_coefficients"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            payload.update(summary)
            payload["think_aux_active_rate"] = float(self.dsr_think_lambda != 0.0)
            traces = [{
                "group_id": item["group_id"],
                "cot": item["cot"],
                "parsed": item["parsed"],
                "s_cot": item["s_cot"],
                "s_prefix": item["s_prefix"],
                "s_explore": item["s_explore"],
                "s_dead": item["s_dead"],
            } for item in records[:2]]
        else:
            positions = []
            weights = []
            coefficients = []
            summary, plans = summarize_nothink_records(records, self.dsr_nothink_scale)
            for plan in plans:
                positions.extend(plan.positions if plan.active else [-1] * group_size)
                weights.extend(plan.frequency_weights if plan.active else [0.0] * group_size)
                coefficients.extend([plan.coefficient if plan.active else 0.0] * group_size)
            output["dsr_aux_advantages"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            output["dsr_sa_positions"] = torch.tensor(positions[process_slice], dtype=torch.long, device=device)
            output["dsr_frequency_weights"] = torch.tensor(weights[process_slice], dtype=torch.float32, device=device)
            output["dsr_rescue_coefficients"] = torch.tensor(coefficients[process_slice], dtype=torch.float32, device=device)
            payload.update(summary)
            traces = [{
                "group_id": records[start]["group_id"],
                "primary_rewards": [item["primary_reward"] for item in records[start:start + group_size]],
                "predicted_as": list(plan.predicted_as),
                "frequency_weights": list(plan.frequency_weights),
                "concentration": plan.concentration,
                "gold_unique_a": plan.gold_unique_a,
                "lambda_a": plan.lambda_a,
                "sa_positions": list(plan.positions),
                "active": plan.active,
            } for plan, start in zip(plans, range(0, len(records), group_size)) if plan.active][:2]

        self._smoke_log[-1]["dsr"] = payload
        self._write_dsr_monitor(payload, traces)
        return output

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Baseline loss copy plus DSR terms, using the same model forward."""
        t0 = time.time()
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.loss_type == "grpo":
            per_sample_loss = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            multiplier = route_multiplier(inputs["route_id"])
            primary_loss = (per_sample_loss * multiplier).mean()
            primary_loss = primary_loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            primary_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            primary_loss = primary_loss / self.current_gradient_accumulation_steps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        route_is_think = int(inputs["route_id"][0]) == ROUTE_ID["think"]
        think_aux_loss = per_token_logps.sum() * 0.0
        rescue_loss = per_token_logps.sum() * 0.0
        loss = primary_loss
        if route_is_think and self.dsr_think_lambda != 0.0:
            aux_advantages = inputs["dsr_aux_advantages"]
            aux_loss1 = coef_1 * aux_advantages.unsqueeze(1)
            aux_loss2 = coef_2 * aux_advantages.unsqueeze(1)
            aux_per_token = -torch.min(aux_loss1, aux_loss2)
            aux_per_sample = (aux_per_token * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            think_aux_loss = aux_per_sample.mean() / self.current_gradient_accumulation_steps
            loss = loss + self.dsr_think_lambda * think_aux_loss
        elif not route_is_think and self.dsr_nothink_scale != 0.0:
            rescue_loss = nothink_unlikelihood_loss(
                per_token_logps,
                inputs["dsr_sa_positions"],
                inputs["dsr_frequency_weights"],
                inputs["dsr_rescue_coefficients"],
            ) / self.current_gradient_accumulation_steps
            loss = loss + rescue_loss

        with torch.no_grad():
            flat = log_ratio[completion_mask.bool()].float()
            ratio_flat = torch.exp(flat)
            pe = self._smoke_policy_epoch.get(self._smoke_rollout_id, 0)
            entry = {
                f"policy_epoch_{pe+1}": 1,
                f"step_{pe+1}": self.state.global_step,
                f"ratio_mean_ep{pe+1}": float(ratio_flat.mean()),
                f"clip_fraction_ep{pe+1}": float(((ratio_flat - 1.0).abs() > self.epsilon_low).float().mean()),
                f"approx_kl_ep{pe+1}": float((ratio_flat - 1.0 - flat).mean()),
                f"policy_fwb_sec_ep{pe+1}": round(time.time() - t0, 2),
                f"primary_loss_ep{pe+1}": float(primary_loss),
                f"think_aux_loss_ep{pe+1}": float(think_aux_loss),
                f"nothink_rescue_loss_ep{pe+1}": float(rescue_loss),
                f"dsr_total_loss_ep{pe+1}": float(loss),
            }
            if not route_is_think and bool((inputs["dsr_sa_positions"] >= 0).any()):
                valid = inputs["dsr_sa_positions"] >= 0
                rows = torch.arange(per_token_logps.size(0), device=per_token_logps.device)[valid]
                selected = per_token_logps[rows, inputs["dsr_sa_positions"][valid]]
                entry[f"sampled_sa_logps_ep{pe+1}"] = selected.detach().float().cpu().tolist()
            if self._detailed_monitor:
                entry.update({
                    f"ratio_std_ep{pe+1}": float(ratio_flat.std(unbiased=False)),
                    f"ratio_p95_ep{pe+1}": float(ratio_flat.quantile(0.95)),
                    f"ratio_p99_ep{pe+1}": float(ratio_flat.quantile(0.99)),
                    f"max_abs_log_ratio_ep{pe+1}": float(flat.abs().max()),
                    f"action_tokens_ep{pe+1}": float(completion_mask.sum()),
                })
            self._smoke_policy_epoch[self._smoke_rollout_id] = pe + 1
            self._smoke_log[-1].update(entry)
        return loss

    def log(self, logs, start_time=None):
        """Write DSR-only step scalars after the frozen baseline logger."""
        result = super().log(logs, start_time)
        if not self._monitor_enabled() or self.accelerator.process_index != 0 or "loss" not in logs:
            return result
        rollout = self._smoke_log[-1] if self._smoke_log else {}
        policy_epoch = self._smoke_policy_epoch.get(self._smoke_rollout_id, 0)
        raw_aux = rollout.get(f"think_aux_loss_ep{policy_epoch}")
        self._monitor._append("dsr_steps.jsonl", {
            "type": "dsr_step",
            "step": self.state.global_step,
            "rollout_id": rollout.get("rollout_id"),
            "route": rollout.get("route"),
            "primary_loss": rollout.get(f"primary_loss_ep{policy_epoch}"),
            "think_aux_loss_raw": raw_aux,
            "think_aux_contribution": (
                self.dsr_think_lambda * raw_aux if raw_aux is not None else None
            ),
            "nothink_rescue_loss": rollout.get(f"nothink_rescue_loss_ep{policy_epoch}"),
            "dsr_total_loss": rollout.get(f"dsr_total_loss_ep{policy_epoch}"),
            "ratio_mean": rollout.get(f"ratio_mean_ep{policy_epoch}"),
            "clip_fraction": rollout.get(f"clip_fraction_ep{policy_epoch}"),
            "approx_kl": rollout.get(f"approx_kl_ep{policy_epoch}"),
            "grad_norm": logs.get("grad_norm"),
        })
        return result
