"""NoThink-only trainer with strict format and first-error frontier credit."""

from __future__ import annotations

import torch

from grpo_sid import parse_sid
from grpo_trl_trainer import ROUTE_ID, RecGRPOTrainer
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer
from nothink_bridge import ddp_group_weight, plan_dead_zero_bridge
from nothink_hierarchical_credit import hierarchy_state, locate_domain_commitment_token

try:
    from .format_validator import validate_nothink_completion
    from .frontier_credit import FORMAT_ADV_TOTAL, plan_frontier_credits
except ImportError:
    from format_validator import validate_nothink_completion
    from frontier_credit import FORMAT_ADV_TOTAL, plan_frontier_credits


FORMAT_REASONS = (
    "nonempty_think",
    "unexpected_prose_before_sid",
    "invalid_sid",
    "malformed_template",
    "other",
)


class NoThinkOnlyFrontierTrainer(ThinkExactClampRecGRPOTrainer):
    """Apply one exclusive format penalty or sparse D/A/B/C frontier credit."""

    def _calculate_rewards(self, inputs, *args, **kwargs):
        if not inputs or any(item.get("route") != "no_think" for item in inputs):
            raise RuntimeError("NoThink-only Frontier trainer received a non-NoThink batch")
        # Bypass the parent hierarchy preparation; its loss and bridge implementation
        # remain inherited after this class installs the Frontier runtime.
        rewards_per_func = RecGRPOTrainer._calculate_rewards(self, inputs, *args, **kwargs)
        completions = kwargs.get("completions")
        if completions is None and len(args) >= 2:
            completions = args[1]
        completion_ids = kwargs.get("completion_ids")
        if completion_ids is None and len(args) >= 3:
            completion_ids = args[2]
        self._think_exact_clamp_route = "no_think"
        return self._prepare_frontier_runtime(
            inputs, completions, completion_ids, rewards_per_func
        )

    def _prepare_frontier_runtime(
        self, inputs, completions, completion_ids, rewards_per_func,
    ):
        from trl.trainer.grpo_trainer import gather_object

        if completions is None or completion_ids is None:
            raise RuntimeError("Frontier reward preparation requires completions and token ids")
        local_records = []
        for local_index, (item, completion, candidate_ids) in enumerate(
            zip(inputs, completions, completion_ids)
        ):
            text = completion if isinstance(completion, str) else completion[0]["content"]
            validation = validate_nothink_completion(text)
            alignment = None
            if validation.valid:
                try:
                    alignment = locate_domain_commitment_token(
                        candidate_ids, validation.parsed_sid, self.processing_class
                    )
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        "format-valid completion has no safe hierarchy-token alignment"
                    ) from exc
            local_records.append({
                "group_id": item["recommendation_group_id"],
                "rank": self.accelerator.process_index,
                "local_index": local_index,
                "format_valid": validation.valid,
                "format_violation_reason": validation.reason,
                "predicted_sid": validation.parsed_sid,
                "token_positions": alignment.hierarchy_token_positions if alignment else None,
                "domain_commitment_eligible": bool(alignment and alignment.eligible),
                "domain_commitment_mode": validation.mode,
                "domain_commitment_token_position": (
                    alignment.commitment_token_position if alignment else None
                ),
                "sid_domain_token_position": (
                    alignment.sid_domain_token_position if alignment else None
                ),
                "domain_commitment_failure": alignment.failure if alignment else None,
                "gold_sids": [parse_sid(value) for value in item["all_gold_sids"]],
                "target_domain": item["target_domain"],
                "prompt": item["prompt"],
            })
        global_records = gather_object(local_records)
        if len(global_records) != rewards_per_func.size(0):
            raise RuntimeError("Frontier metadata and gathered rewards are misaligned")

        rewards_per_func = rewards_per_func.clone()
        if rewards_per_func.size(1) != 1:
            raise RuntimeError("NoThink-only Frontier requires exactly one scalar reward")
        for index, record in enumerate(global_records):
            if not record["format_valid"]:
                rewards_per_func[index, 0] = -1.0
        weighted_rewards = (
            rewards_per_func
            * self.reward_weights.to(self.accelerator.device).unsqueeze(0)
        ).nansum(dim=1).tolist()

        grouped = {}
        for record, reward in zip(global_records, weighted_rewards):
            bucket = grouped.setdefault(record["group_id"], {"records": [], "rewards": []})
            bucket["records"].append(record)
            bucket["rewards"].append(reward)
        bridge_plans = {}
        frontier_plans = {}
        credits_by_candidate = {}
        kinds_by_candidate = {}
        commitment_by_group = {}
        hierarchy_by_group = {}
        format_by_group = {}
        for group_id, group in grouped.items():
            records = group["records"]
            if len(records) != 8:
                raise RuntimeError(f"NoThink Frontier group {group_id!r} is not G8")
            head = records[0]
            states = [
                hierarchy_state(record["predicted_sid"], head["gold_sids"], head["target_domain"])
                for record in records
            ]
            valid = [bool(record["format_valid"]) for record in records]
            frontier = plan_frontier_credits(states, valid)
            frontier_plans[group_id] = frontier
            for record, candidate in zip(records, frontier.candidates):
                key = (record["rank"], record["local_index"])
                credits_by_candidate[key] = candidate.credits
                kinds_by_candidate[key] = candidate.kinds
            commitment_by_group[group_id] = {
                "eligible_count": sum(record["domain_commitment_eligible"] for record in records),
                "branch_count": sum(record["domain_commitment_mode"] == "branch" for record in records),
                "direct_sid_fallback_count": sum(
                    record["domain_commitment_mode"] == "direct_sid_fallback" for record in records
                ),
                "unresolved_count": sum(
                    record["format_valid"] and not record["domain_commitment_eligible"]
                    for record in records
                ),
            }
            hierarchy_by_group[group_id] = {
                "stage_active": frontier.positive_active,
                "b_success_count": sum(state.ab_correct and is_valid for state, is_valid in zip(states, valid)),
                "c_success_count": sum(state.exact and is_valid for state, is_valid in zip(states, valid)),
            }
            reason_counts = {reason: 0 for reason in FORMAT_REASONS}
            for record in records:
                if not record["format_valid"]:
                    reason = record["format_violation_reason"]
                    reason_counts[reason if reason in reason_counts else "other"] += 1
            format_by_group[group_id] = {
                "valid_count": sum(valid),
                "violation_count": 8 - sum(valid),
                "reason_counts": reason_counts,
            }
            bridge_plans[group_id] = plan_dead_zero_bridge(
                group["rewards"], head["gold_sids"], head["target_domain"]
            )

        local_group_ids = {item["recommendation_group_id"] for item in inputs}
        if len(local_group_ids) != 1:
            raise RuntimeError("configured NoThink local batch must contain one group")
        group_id = next(iter(local_group_ids))
        group = grouped[group_id]
        ranks_for_group = len({record["rank"] for record in group["records"]})
        runtime = self._build_bridge_runtime(
            inputs[0],
            bridge_plans[group_id],
            any(plan.active for plan in bridge_plans.values()),
            ddp_group_weight(self.accelerator.num_processes, len(grouped), ranks_for_group),
            group["rewards"],
            commitment_by_group[group_id],
            hierarchy_by_group[group_id],
            [record["predicted_sid"] for record in group["records"]],
            [
                credits_by_candidate[(self.accelerator.process_index, index)]
                for index in range(len(inputs))
            ],
            [record["token_positions"] for record in local_records],
        )
        frontier = frontier_plans[group_id]
        format_summary = format_by_group[group_id]
        runtime.update({
            "credit_assignment": "strict_format_then_first_error_frontier_v1",
            "format_valid": tuple(record["format_valid"] for record in local_records),
            "format_violation_reasons": tuple(
                record["format_violation_reason"] for record in local_records
            ),
            "token_credit_kinds": tuple(
                kinds_by_candidate[(self.accelerator.process_index, index)]
                for index in range(len(inputs))
            ),
            "format_valid_candidate_count": format_summary["valid_count"],
            "format_violation_candidate_count": format_summary["violation_count"],
            "format_violation_rate": format_summary["violation_count"] / 8.0,
            "format_violation_reason_counts": format_summary["reason_counts"],
            "frontier_negative_counts": frontier.frontier_negative_counts,
            "frontier_active": frontier.frontier_active,
            "positive_stage_active": frontier.positive_active,
            "zero_signal_taxonomy": frontier.taxonomy,
        })
        self._nothink_bridge_runtime = runtime
        return rewards_per_func

    def _attach_nothink_token_advantages(self, output):
        runtime = self._nothink_bridge_runtime
        token_advantages = torch.zeros_like(
            output["completion_ids"], dtype=output["advantages"].dtype
        )
        sequence_penalty_mask = torch.zeros_like(output["completion_mask"], dtype=torch.bool)
        if len(runtime["token_credits"]) != token_advantages.size(0):
            raise RuntimeError("Frontier credit rows are not batch-aligned")
        for row, (valid, credits, positions) in enumerate(zip(
            runtime["format_valid"], runtime["token_credits"], runtime["token_positions"]
        )):
            if not valid:
                mask = output["completion_mask"][row].bool()
                length = int(mask.sum().item())
                if length <= 0:
                    raise RuntimeError("format violation has no generated completion token")
                token_advantages[row, mask] = FORMAT_ADV_TOTAL / length
                sequence_penalty_mask[row, mask] = True
                continue
            if positions is None:
                raise RuntimeError("format-valid candidate has no hierarchy token positions")
            for credit, position in zip(credits, positions):
                if position is None:
                    if credit != 0:
                        raise RuntimeError("nonzero Frontier credit has no token position")
                    continue
                token_advantages[row, position] = credit
        output["token_advantages"] = token_advantages
        output["sequence_penalty_mask"] = sequence_penalty_mask

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not bool((inputs["route_id"] == ROUTE_ID["no_think"]).all()):
            raise RuntimeError("NoThink-only Frontier trainer received a non-NoThink loss batch")
        return super()._compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _frontier_monitor_fields(self):
        runtime = self._nothink_bridge_runtime
        counts = runtime["frontier_negative_counts"]
        active = runtime["frontier_active"]
        positive = runtime["positive_stage_active"]
        fields = {
            "credit_assignment": runtime["credit_assignment"],
            "format_valid_candidate_count": runtime["format_valid_candidate_count"],
            "format_violation_candidate_count": runtime["format_violation_candidate_count"],
            "format_violation_rate": runtime["format_violation_rate"],
            "format_violation_reason_counts": runtime["format_violation_reason_counts"],
            "format_penalty_total_per_violation": FORMAT_ADV_TOTAL,
            "zero_signal_taxonomy": runtime["zero_signal_taxonomy"],
        }
        for index, name in enumerate(("domain", "a", "b", "c")):
            fields[f"{name}_frontier_negative_count"] = counts[index]
            fields[f"{name}_frontier_active"] = active[index]
            fields[f"{name}_positive_active"] = positive[index]
        fields["candidates"] = [
            {
                "format_valid": valid,
                "format_violation_reason": reason,
                "credit_representation": (
                    "format_sequence_penalty" if not valid else "hierarchical_sparse_token"
                ),
                "format_penalty_total": FORMAT_ADV_TOTAL if not valid else None,
                "stage_credits": list(credits),
                "stage_credit_kinds": list(kinds),
                "token_positions": list(positions) if positions is not None else None,
            }
            for valid, reason, credits, kinds, positions in zip(
                runtime["format_valid"],
                runtime["format_violation_reasons"],
                runtime["token_credits"],
                runtime["token_credit_kinds"],
                runtime["token_positions"],
            )
        ]
        return fields

    def _record_bridge_loss(self, primary_loss, total_loss, a_loss, bridge_loss, weighted):
        super()._record_bridge_loss(primary_loss, total_loss, a_loss, bridge_loss, weighted)
        event = {
            "type": "nothink_frontier_loss",
            "rollout_id": self._smoke_rollout_id,
            "step": self.state.global_step,
            **self._frontier_monitor_fields(),
        }
        if self._smoke_log:
            self._smoke_log[-1].update(event)
        if self._monitor_enabled():
            self._monitor._append(
                f"ranks/rank{self.accelerator.process_index}-nothink-frontier.jsonl", event
            )

    def _write_nothink_bridge_plan(self, entry):
        super()._write_nothink_bridge_plan(entry)
        if not self._monitor_enabled() or self._nothink_bridge_runtime is None:
            return
        event = {
            "type": "nothink_frontier_plan",
            "rollout_id": entry["rollout_id"],
            "step": self.state.global_step,
            **self._frontier_monitor_fields(),
        }
        entry.update(event)
        self._monitor._append(
            f"ranks/rank{self.accelerator.process_index}-nothink-frontier.jsonl", event
        )
