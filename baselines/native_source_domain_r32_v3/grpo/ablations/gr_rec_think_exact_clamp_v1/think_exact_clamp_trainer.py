"""Minimal Think-only advantage override for GR_REC_v1."""

from __future__ import annotations

from grpo_trl_trainer import RecGRPOTrainer

try:
    from .think_diagnostics import interest_diagnostics
    from .think_exact_clamp import think_exact_clamp_advantages
except ImportError:  # Direct script/PYTHONPATH entry point.
    from think_diagnostics import interest_diagnostics
    from think_exact_clamp import think_exact_clamp_advantages


class ThinkExactClampRecGRPOTrainer(RecGRPOTrainer):
    """Use ExactClamp for Think; return the untouched parent output for NoThink."""

    def _calculate_rewards(self, inputs, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(inputs, *args, **kwargs)
        self._think_exact_clamp_route = inputs[0]["route"]
        if self._think_exact_clamp_route == "think":
            self._think_exact_clamp_global_rewards = (
                rewards_per_func * self.reward_weights.to(self.accelerator.device).unsqueeze(0)
            ).nansum(dim=1)
        return rewards_per_func

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

    def _write_rollout_monitor(
        self, entry, inputs, completion_ids_list, completions_text, local_rewards,
        beam_call, group_ids_all, completion_ids_all, global_rewards,
    ):
        if self._think_exact_clamp_route != "think":
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
