"""Isolated DSR trainer subclass. GR_REC_v1 files remain untouched."""
from __future__ import annotations

import json
import os
import time

import torch

from grpo_trl_trainer import ROUTE_ID, RecGRPOTrainer, route_multiplier

from .dsr_objectives import (
    build_nothink_rescue_plan,
    choose_think_aux_scores,
    group_aux_advantages,
    nothink_unlikelihood_loss,
)
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
        event = {"type": "dsr_rollout", "rollout_id": self._smoke_rollout_id, **payload}
        self._monitor._append("dsr_metrics.jsonl", event)
        if traces:
            self._monitor._append("dsr_traces.jsonl", {
                "type": "dsr_trace",
                "rollout_id": self._smoke_rollout_id,
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
        records = gather_object(local_records)
        route = inputs[0]["route"]
        group_size = 4 if route == "think" else 8
        self._validate_groups(records, group_size)
        local_count = len(local_records)
        process_slice = slice(
            self.accelerator.process_index * local_count,
            (self.accelerator.process_index + 1) * local_count,
        )
        device = output["completion_ids"].device
        payload = {"route": route, "tokenizer_audit": self._dsr_capture.tokenizer_audit}
        traces = []

        if route == "think":
            scores = []
            branches = []
            for start in range(0, len(records), group_size):
                group_scores, branch = choose_think_aux_scores(records[start:start + group_size])
                scores.extend(group_scores)
                branches.append(branch)
            advantages = group_aux_advantages(scores, group_size)
            output["dsr_aux_advantages"] = advantages[process_slice].to(device)
            output["dsr_sa_positions"] = torch.full((local_count,), -1, dtype=torch.long, device=device)
            output["dsr_frequency_weights"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            output["dsr_rescue_coefficients"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            grounded_counts = [int(item["grounded_count"]) for item in records]
            dist = {str(value): sum(count == value for count in grounded_counts) for value in range(5)}
            dist["5+"] = sum(count >= 5 for count in grounded_counts)
            aux_std_zero = 0
            for start in range(0, len(scores), group_size):
                if float(torch.tensor(scores[start:start + group_size]).std(correction=0)) == 0.0:
                    aux_std_zero += 1
            primary_zero_std = 0
            all_zero = 0
            for start in range(0, len(records), group_size):
                primary = [item["primary_reward"] for item in records[start:start + group_size]]
                primary_zero_std += float(torch.tensor(primary).std(correction=0)) == 0.0
                all_zero += all(float(value) == 0.0 for value in primary)
            payload.update({
                "parser_success_rate": sum(bool(item["parsed"]["parser_success"]) for item in records) / len(records),
                "grounded_interest_count_mean": sum(grounded_counts) / len(grounded_counts),
                "grounded_interest_count_distribution": dist,
                "s_cot_mean": sum(float(item["s_cot"]) for item in records) / len(records),
                "s_cot_std": float(torch.tensor([item["s_cot"] for item in records]).std(correction=0)),
                "d_cot_mean": sum(float(item["evidence_diversity"]) for item in records) / len(records),
                "s_prefix_mean": sum(float(item["s_prefix"]) for item in records) / len(records),
                "s_explore_mean": sum(float(item["s_explore"]) for item in records) / len(records),
                "s_dead_mean": sum(float(item["s_dead"]) for item in records) / len(records),
                "think_aux_active_rate": float(self.dsr_think_lambda != 0.0),
                "think_aux_zero_std_rate": aux_std_zero / (len(records) / group_size),
                "primary_zero_std_rate": primary_zero_std / (len(records) / group_size),
                "all_zero_rate": all_zero / (len(records) / group_size),
                "unique_a_mean": sum(int(item["unique_a"]) for item in records) / len(records),
                "a_entropy_mean": sum(float(item["a_entropy_norm"]) for item in records) / len(records),
                "correct_a_support": sum(float(item["s_a"]) for item in records) / len(records),
                "correct_ab_support": sum(float(item["s_ab"]) for item in records) / len(records),
                "branches": dict((name, branches.count(name)) for name in sorted(set(branches))),
                "aux_scores": scores,
                "aux_advantages": advantages.tolist(),
            })
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
            plans = []
            for start in range(0, len(records), group_size):
                group = records[start:start + group_size]
                plan = build_nothink_rescue_plan(
                    [item["primary_reward"] for item in group],
                    [item["predicted_a"] for item in group],
                    [item["sa_position"] for item in group],
                    group[0]["gold_as"],
                    rescue_scale=self.dsr_nothink_scale,
                )
                plans.append(plan)
                positions.extend(plan.positions if plan.active else [-1] * group_size)
                weights.extend(plan.frequency_weights if plan.active else [0.0] * group_size)
                coefficients.extend([plan.coefficient if plan.active else 0.0] * group_size)
            output["dsr_aux_advantages"] = torch.zeros(local_count, dtype=torch.float32, device=device)
            output["dsr_sa_positions"] = torch.tensor(positions[process_slice], dtype=torch.long, device=device)
            output["dsr_frequency_weights"] = torch.tensor(weights[process_slice], dtype=torch.float32, device=device)
            output["dsr_rescue_coefficients"] = torch.tensor(coefficients[process_slice], dtype=torch.float32, device=device)
            all_zero_plans = [
                all(records[start + index]["primary_reward"] == 0.0 for index in range(group_size))
                for start in range(0, len(records), group_size)
            ]
            concentrations = [plan.concentration for plan, zero in zip(plans, all_zero_plans) if zero]
            sorted_concentrations = sorted(concentrations)
            p90_index = min(len(sorted_concentrations) - 1, int(0.9 * len(sorted_concentrations))) if sorted_concentrations else 0
            payload.update({
                "all_zero_group_rate": sum(all_zero_plans) / len(plans),
                "all_zero_same_a_rate": sum(zero and plan.concentration == 1.0 for plan, zero in zip(plans, all_zero_plans)) / len(plans),
                "a_concentration_mean": sum(concentrations) / len(concentrations) if concentrations else 0.0,
                "a_concentration_p90": sorted_concentrations[p90_index] if sorted_concentrations else 0.0,
                "rescue_active_groups": sum(plan.active for plan in plans),
                "sparse_rescue_count": sum(plan.active and plan.gold_unique_a < 3 for plan in plans),
                "dense_rescue_count": sum(plan.active and plan.gold_unique_a >= 3 for plan in plans),
                "plans": [plan.__dict__ | {"coefficient": plan.coefficient} for plan in plans],
            })
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
