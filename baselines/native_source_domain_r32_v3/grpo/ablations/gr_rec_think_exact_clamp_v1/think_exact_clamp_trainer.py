"""Think ExactClamp plus a gated NoThink hierarchical teacher bridge."""

from __future__ import annotations

import torch

from grpo_model import encode_prompt
from grpo_sid import final_sid, parse_sid
from grpo_trl_trainer import ROUTE_ID, RecGRPOTrainer

try:
    from .think_diagnostics import interest_diagnostics
    from .think_exact_clamp import think_exact_clamp_advantages
    from .nothink_bridge import (
        BRIDGE_COLLAPSE_AB,
        BRIDGE_DEAD_A,
        BRIDGE_LAMBDA,
        BRIDGE_OFF,
        bridge_token_text,
        ddp_group_weight,
        plan_nothink_bridge,
        require_single_token,
        uniform_multi_positive_ce,
    )
except ImportError:  # Direct script/PYTHONPATH entry point.
    from think_diagnostics import interest_diagnostics
    from think_exact_clamp import think_exact_clamp_advantages
    from nothink_bridge import (
        BRIDGE_COLLAPSE_AB,
        BRIDGE_DEAD_A,
        BRIDGE_LAMBDA,
        BRIDGE_OFF,
        bridge_token_text,
        ddp_group_weight,
        plan_nothink_bridge,
        require_single_token,
        uniform_multi_positive_ce,
    )


class ThinkExactClampRecGRPOTrainer(RecGRPOTrainer):
    """Use ExactClamp for Think and a token-local bridge for gated NoThink groups."""

    bridge_lambda = BRIDGE_LAMBDA

    def _calculate_rewards(self, inputs, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(inputs, *args, **kwargs)
        self._think_exact_clamp_route = inputs[0]["route"]
        if self._think_exact_clamp_route == "think":
            self._nothink_bridge_runtime = None
            self._think_exact_clamp_global_rewards = (
                rewards_per_func * self.reward_weights.to(self.accelerator.device).unsqueeze(0)
            ).nansum(dim=1)
        else:
            completions = kwargs.get("completions")
            if completions is None and len(args) >= 2:
                completions = args[1]
            self._prepare_nothink_bridge(inputs, completions, rewards_per_func)
        return rewards_per_func

    def _prepare_nothink_bridge(self, inputs, completions, rewards_per_func):
        from trl.trainer.grpo_trainer import gather_object

        weighted_rewards = (
            rewards_per_func * self.reward_weights.to(self.accelerator.device).unsqueeze(0)
        ).nansum(dim=1).tolist()
        local_records = []
        for item, completion in zip(inputs, completions):
            text = completion if isinstance(completion, str) else completion[0]["content"]
            local_records.append({
                "group_id": item["recommendation_group_id"],
                "rank": self.accelerator.process_index,
                "predicted_sid": final_sid(text),
                "gold_sids": [parse_sid(value) for value in item["all_gold_sids"]],
                "target_domain": item["target_domain"],
                "prompt": item["prompt"],
            })
        global_records = gather_object(local_records)
        if len(global_records) != len(weighted_rewards):
            raise RuntimeError("bridge metadata and gathered rewards are misaligned")

        grouped = {}
        for record, reward in zip(global_records, weighted_rewards):
            bucket = grouped.setdefault(record["group_id"], {"records": [], "rewards": []})
            bucket["records"].append(record)
            bucket["rewards"].append(reward)
        plans = {}
        for group_id, group in grouped.items():
            records = group["records"]
            if len(records) != 8:
                raise RuntimeError(f"NoThink bridge group {group_id!r} is not G8")
            head = records[0]
            plans[group_id] = plan_nothink_bridge(
                group["rewards"],
                [record["predicted_sid"] for record in records],
                head["gold_sids"],
                head["target_domain"],
            )

        local_group_ids = {item["recommendation_group_id"] for item in inputs}
        if len(local_group_ids) != 1:
            raise RuntimeError("configured NoThink local batch must contain one group")
        group_id = next(iter(local_group_ids))
        plan = plans[group_id]
        group = grouped[group_id]
        ranks_for_group = len({record["rank"] for record in group["records"]})
        group_weight = ddp_group_weight(
            self.accelerator.num_processes, len(grouped), ranks_for_group
        )
        self._nothink_bridge_runtime = self._build_bridge_runtime(
            inputs[0], plan, any(candidate.active for candidate in plans.values()),
            group_weight, group["rewards"],
            [record["predicted_sid"] for record in group["records"]],
        )

    def _build_bridge_runtime(
        self, item, plan, global_active, ddp_group_weight, rewards, predicted_sids,
    ):
        prompt_ids = encode_prompt(self.processing_class, item["prompt"])
        domain_id = require_single_token(
            self.processing_class, bridge_token_text("domain", item["target_domain"])
        )
        a_targets = plan.gold_a_targets if plan.branch == BRIDGE_DEAD_A else plan.missing_a_targets
        a_target_ids = tuple(
            require_single_token(self.processing_class, bridge_token_text("a", value))
            for value in a_targets
        )
        b_target_ids = tuple(
            require_single_token(self.processing_class, bridge_token_text("b", value))
            for value in plan.current_b_targets
        )
        current_a_id = (
            require_single_token(self.processing_class, bridge_token_text("a", plan.current_a))
            if plan.current_a is not None else None
        )
        gold = {sid for value in item["all_gold_sids"] if (sid := parse_sid(value)) is not None}
        predicted = [sid for sid in predicted_sids if sid is not None]
        gold_a = {(sid[0], sid[1]) for sid in gold}
        gold_ab = {(sid[0], sid[1], sid[2]) for sid in gold}
        denominator = len(predicted_sids)
        return {
            "plan": plan,
            "global_active": bool(global_active),
            "ddp_group_weight": float(ddp_group_weight),
            "prompt_ids": tuple(prompt_ids),
            "domain_id": domain_id,
            "current_a_id": current_a_id,
            "a_target_ids": a_target_ids,
            "b_target_ids": b_target_ids,
            "rewards": tuple(rewards),
            "gold_unique_a_count": len(gold_a),
            "pred_unique_a_count": len({sid[1] for sid in predicted}),
            "gold_a_candidate_hit_rate": sum((sid[0], sid[1]) in gold_a for sid in predicted) / denominator,
            "gold_ab_candidate_hit_rate": sum((sid[0], sid[1], sid[2]) in gold_ab for sid in predicted) / denominator,
            "exact_candidate_hit_rate": sum(sid in gold for sid in predicted) / denominator,
            "wrong_domain_rate": sum(sid[0] != item["target_domain"] for sid in predicted) / denominator,
            "valid_sid_rate": len(predicted) / denominator,
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self._think_exact_clamp_route != "think":
            del self._think_exact_clamp_route
            return output

        global_advantages = think_exact_clamp_advantages(
            self._think_exact_clamp_global_rewards, self.num_generations
        )
        local_count = output["advantages"].numel()
        start = self.accelerator.process_index * local_count
        stop = start + local_count
        output["advantages"] = global_advantages[start:stop]

        if self._detailed_monitor:
            self._logs["advantages"][-global_advantages.numel():] = (
                self._think_exact_clamp_advantages_list
            )
        if self._parity_audit and self._parity_log:
            self._parity_log[-1]["advantages"] = self._think_exact_clamp_advantages_list[start:stop]
        del self._think_exact_clamp_route
        del self._think_exact_clamp_global_rewards
        del self._think_exact_clamp_advantages_list
        return output

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        primary_loss = super()._compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        route_ids = inputs["route_id"]
        if int(route_ids[0]) != ROUTE_ID["no_think"] or self.bridge_lambda == 0:
            return primary_loss
        runtime = self._nothink_bridge_runtime
        if runtime is None:
            raise RuntimeError("NoThink bridge runtime was not prepared for this rollout")
        if not runtime["global_active"]:
            zero = primary_loss.detach() * 0.0
            self._record_bridge_loss(primary_loss, primary_loss, None, None, zero, zero)
            return primary_loss

        prompt = runtime["prompt_ids"] + (runtime["domain_id"],)
        b_query = prompt + ((runtime["current_a_id"],) if runtime["current_a_id"] is not None else ())
        queries = (prompt, b_query)
        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
            pad_id = self.processing_class.eos_token_id
        width = max(map(len, queries))
        input_ids = torch.full((2, width), pad_id, dtype=torch.long, device=primary_loss.device)
        attention_mask = torch.zeros((2, width), dtype=torch.long, device=primary_loss.device)
        for row, query in enumerate(queries):
            query_tensor = torch.tensor(query, dtype=torch.long, device=primary_loss.device)
            input_ids[row, -len(query):] = query_tensor
            attention_mask[row, -len(query):] = 1
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1, :]

        dummy_targets = (runtime["domain_id"],)
        a_loss = uniform_multi_positive_ce(logits[0], runtime["a_target_ids"] or dummy_targets)
        b_loss = uniform_multi_positive_ce(logits[1], runtime["b_target_ids"] or dummy_targets)
        branch = runtime["plan"].branch
        if branch == BRIDGE_DEAD_A:
            bridge_loss = a_loss
        elif branch == BRIDGE_COLLAPSE_AB:
            bridge_loss = 0.5 * a_loss + 0.5 * b_loss
        elif branch == BRIDGE_OFF:
            bridge_loss = (a_loss + b_loss) * 0.0
        else:
            raise RuntimeError(f"unknown bridge branch: {branch}")
        bridge_loss = bridge_loss * runtime["ddp_group_weight"]
        weighted = self.bridge_lambda * bridge_loss / self.current_gradient_accumulation_steps
        total_loss = primary_loss + weighted
        self._record_bridge_loss(
            primary_loss, total_loss, a_loss, b_loss, bridge_loss,
            self.bridge_lambda * bridge_loss,
        )
        return total_loss

    def _record_bridge_loss(self, primary_loss, total_loss, a_loss, b_loss, bridge_loss, weighted):
        runtime = self._nothink_bridge_runtime
        plan = runtime["plan"]
        event = {
            "type": "nothink_hierarchical_bridge_loss",
            "rollout_id": self._smoke_rollout_id,
            "step": self.state.global_step,
            "bridge_branch": plan.branch,
            "bridge_active": plan.active,
            "bridge_lambda": self.bridge_lambda,
            "gold_unique_a_count": runtime["gold_unique_a_count"],
            "pred_unique_a_count": runtime["pred_unique_a_count"],
            "current_a": plan.current_a,
            "current_a_is_gold": plan.current_a in plan.gold_a_targets if plan.current_a is not None else None,
            "dead_a_target_count": len(plan.gold_a_targets) if plan.branch == BRIDGE_DEAD_A else 0,
            "missing_a_target_count": len(plan.missing_a_targets),
            "current_b_target_count": len(plan.current_b_targets),
            "bridge_a_loss_raw": float(a_loss.detach()) if plan.active and a_loss is not None else None,
            "bridge_b_loss_raw": (
                float(b_loss.detach())
                if plan.branch == BRIDGE_COLLAPSE_AB and b_loss is not None else None
            ),
            "bridge_total_raw": float(bridge_loss.detach()),
            "bridge_total_weighted": float(weighted.detach()),
            "primary_loss": float(primary_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "primary_zero_std": len(set(runtime["rewards"])) == 1,
            "primary_all_zero": runtime["rewards"] == (0.0,) * 8,
            "gold_a_candidate_hit_rate": runtime["gold_a_candidate_hit_rate"],
            "gold_ab_candidate_hit_rate": runtime["gold_ab_candidate_hit_rate"],
            "exact_candidate_hit_rate": runtime["exact_candidate_hit_rate"],
            "wrong_domain_rate": runtime["wrong_domain_rate"],
            "valid_sid_rate": runtime["valid_sid_rate"],
        }
        if self._smoke_log:
            self._smoke_log[-1].update(event)
        if self._monitor_enabled():
            self._monitor._append(
                f"ranks/rank{self.accelerator.process_index}-nothink-bridge.jsonl", event
            )

    def _write_rollout_monitor(
        self, entry, inputs, completion_ids_list, completions_text, local_rewards,
        beam_call, group_ids_all, completion_ids_all, global_rewards,
    ):
        if self._think_exact_clamp_route != "think":
            self._write_nothink_bridge_plan(entry)
            return super()._write_rollout_monitor(
                entry, inputs, completion_ids_list, completions_text, local_rewards,
                beam_call, group_ids_all, completion_ids_all, global_rewards,
            )

        grouped = [global_rewards[index:index + 4] for index in range(0, len(global_rewards), 4)]
        global_advantages = []
        global_group_stats = []
        for rewards in grouped:
            mean = sum(rewards) / 4
            raw = [(reward - mean) / 8.0 for reward in rewards]
            advantages = [
                0.0 if reward >= 8.0 and advantage < 0 else advantage
                for reward, advantage in zip(rewards, raw)
            ]
            global_advantages.extend(advantages)
            high = [advantage for reward, advantage in zip(rewards, advantages) if reward >= 8.0]
            global_group_stats.append({
                "rewards": rewards,
                "exact_candidate_count": sum(reward >= 8.0 for reward in rewards),
                "multi_positive_candidate_count": sum(reward > 8.0 for reward in rewards),
                "high_quality_negative_rate": (
                    sum(advantage < 0 for advantage in high) / len(high) if high else 0.0
                ),
                "advantage_abs_mean": sum(abs(value) for value in advantages) / 4,
            })
        self._think_exact_clamp_advantages_list = global_advantages
        entry.update({
            "advantage_rule": "think_exact_clamp_v1",
            "high_quality_negative_rate": sum(
                item["high_quality_negative_rate"] for item in global_group_stats
            ) / len(global_group_stats),
            "advantage_abs_mean": sum(
                item["advantage_abs_mean"] for item in global_group_stats
            ) / len(global_group_stats),
        })
        if self._smoke_log:
            self._smoke_log[-1].update(entry)
        self._write_think_exact_clamp_diagnostics(
            entry, inputs, completion_ids_list, completions_text, local_rewards, beam_call
        )
        return super()._write_rollout_monitor(
            entry, inputs, completion_ids_list, completions_text, local_rewards,
            beam_call, group_ids_all, completion_ids_all, global_rewards,
        )

    def _write_nothink_bridge_plan(self, entry):
        if not self._monitor_enabled() or self._nothink_bridge_runtime is None:
            return
        runtime = self._nothink_bridge_runtime
        plan = runtime["plan"]
        event = {
            "type": "nothink_hierarchical_bridge_plan",
            "rollout_id": entry["rollout_id"],
            "step": self.state.global_step,
            "bridge_branch": plan.branch,
            "bridge_active": plan.active,
            "bridge_lambda": self.bridge_lambda,
            "gold_unique_a_count": runtime["gold_unique_a_count"],
            "pred_unique_a_count": runtime["pred_unique_a_count"],
            "current_a": plan.current_a,
            "current_a_is_gold": plan.current_a in plan.gold_a_targets if plan.current_a is not None else None,
            "dead_a_target_count": len(plan.gold_a_targets) if plan.branch == BRIDGE_DEAD_A else 0,
            "missing_a_target_count": len(plan.missing_a_targets),
            "current_b_target_count": len(plan.current_b_targets),
            "primary_zero_std": len(set(runtime["rewards"])) == 1,
            "primary_all_zero": runtime["rewards"] == (0.0,) * 8,
            "gold_a_candidate_hit_rate": runtime["gold_a_candidate_hit_rate"],
            "gold_ab_candidate_hit_rate": runtime["gold_ab_candidate_hit_rate"],
            "exact_candidate_hit_rate": runtime["exact_candidate_hit_rate"],
            "wrong_domain_rate": runtime["wrong_domain_rate"],
            "valid_sid_rate": runtime["valid_sid_rate"],
        }
        entry.update(event)
        self._monitor._append(
            f"ranks/rank{self.accelerator.process_index}-nothink-bridge.jsonl", event
        )

    def _write_think_exact_clamp_diagnostics(
        self, entry, inputs, completion_ids_list, completions_text, local_rewards, beam_call,
    ):
        if not self._monitor_enabled():
            return
        local_results = beam_call.get("local_results", []) if beam_call else []
        groups = []
        for start in range(0, len(inputs), 4):
            stop = start + 4
            rewards = local_rewards[start:stop]
            mean = sum(rewards) / len(rewards)
            raw = [(reward - mean) / 8.0 for reward in rewards]
            advantages = [
                0.0 if reward >= 8.0 and advantage < 0 else advantage
                for reward, advantage in zip(rewards, raw)
            ]
            results = local_results[start:stop]
            diagnostics = [
                {
                    "completion_length": len(completion_ids_list[index]),
                    **interest_diagnostics(completions_text[index], inputs[index]["prompt"]),
                }
                for index in range(start, stop)
            ]
            gold = {tuple(sid) for item in inputs[start:stop] for sid in item["all_gold_sids"]}
            exact_covered = {
                tuple(sid)
                for result in results
                for sid in result.get("beam_sids", [])
                if sid is not None and tuple(sid) in gold
            }
            high = [adv for reward, adv in zip(rewards, advantages) if reward >= 8.0]
            groups.append({
                "group_id": inputs[start]["recommendation_group_id"],
                "rewards": rewards,
                "exact_candidate_count": sum(reward >= 8.0 for reward in rewards),
                "multi_positive_candidate_count": sum(reward > 8.0 for reward in rewards),
                "high_quality_negative_rate": (
                    sum(value < 0 for value in high) / len(high) if high else 0.0
                ),
                "advantage_abs_mean": sum(abs(value) for value in advantages) / len(advantages),
                "exact_gold_hit_count": sum(result.get("exact", 0) for result in results),
                "ab_hit_count": sum(result.get("ab", 0) for result in results),
                "a_hit_count": sum(result.get("a", 0) for result in results),
                "distinct_exact_gold_sids_covered_across_g4": len(exact_covered),
                "candidates": diagnostics,
            })
        self._monitor._append(
            f"ranks/rank{self.accelerator.process_index}-think-exact-clamp.jsonl",
            {
                "type": "think_exact_clamp",
                "rollout_id": entry["rollout_id"],
                "step": self.state.global_step,
                "route": "think",
                "groups": groups,
                "beam_exact_hit_count": beam_call.get("exact") if beam_call else None,
                "beam_ab_hit_count": beam_call.get("ab") if beam_call else None,
                "beam_a_hit_count": beam_call.get("a") if beam_call else None,
            },
        )
