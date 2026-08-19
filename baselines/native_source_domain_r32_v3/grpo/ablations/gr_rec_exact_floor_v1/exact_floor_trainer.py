"""Minimal RecGRPOTrainer override for ExactFloor advantages.

Reward calculation, generation, Beam32, sampling and loss remain owned by the
GR_REC_v1 parent. This subclass only replaces the returned advantage tensor.
"""

from __future__ import annotations

import torch

from grpo_trl_trainer import RecGRPOTrainer

try:
    from .exact_floor import exact_floor_advantages
except ImportError:  # Direct script/PYTHONPATH entry point.
    from exact_floor import exact_floor_advantages


class ExactFloorRecGRPOTrainer(RecGRPOTrainer):
    """GR_REC_v1 trainer with only its group advantage construction replaced."""

    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        device = self.accelerator.device
        self._exact_floor_global_rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        return rewards_per_func

    def _exact_floor_local_advantages(self, local_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self._exact_floor_global_rewards
        global_advantages = exact_floor_advantages(rewards, self.num_generations)
        start = self.accelerator.process_index * local_count
        return global_advantages[start:start + local_count], global_advantages

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        local_advantages, global_advantages = self._exact_floor_local_advantages(
            output["advantages"].numel()
        )
        output["advantages"] = local_advantages

        if self._detailed_monitor:
            count = global_advantages.numel()
            self._logs["advantages"][-count:] = self._exact_floor_advantages_list
        if self._parity_audit and self._parity_log:
            start = self.accelerator.process_index * output["advantages"].numel()
            stop = start + output["advantages"].numel()
            self._parity_log[-1]["advantages"] = self._exact_floor_advantages_list[start:stop]
        del self._exact_floor_global_rewards
        del self._exact_floor_advantages_list
        return output

    def _write_rollout_monitor(
        self, entry, inputs, completion_ids_list, completions_text, local_rewards,
        beam_call, group_ids_all, completion_ids_all, global_rewards,
    ):
        grouped = [
            global_rewards[index:index + self.num_generations]
            for index in range(0, len(global_rewards), self.num_generations)
        ]
        baselines = [min(sum(group) / len(group), 8.0) for group in grouped]
        advantages = [
            (reward - baseline) / 8.0
            for group, baseline in zip(grouped, baselines)
            for reward in group
        ]
        self._exact_floor_advantages_list = advantages
        high_quality = [
            advantage for reward, advantage in zip(global_rewards, advantages) if reward >= 8.0
        ]
        stats = {
            "group_reward_mean": sum(global_rewards) / len(global_rewards),
            "exact_floor_baseline": sum(baselines) / len(baselines),
            "advantage_mean": sum(advantages) / len(advantages),
            "advantage_abs_mean": sum(abs(value) for value in advantages) / len(advantages),
            "high_quality_negative_rate": (
                sum(value < 0 for value in high_quality) / len(high_quality) if high_quality else 0.0
            ),
        }
        entry.update(stats)
        entry["advantage_rule"] = "exact_floor_v1"
        if self._smoke_log:
            self._smoke_log[-1].update(stats)
        return super()._write_rollout_monitor(
            entry, inputs, completion_ids_list, completions_text, local_rewards,
            beam_call, group_ids_all, completion_ids_all, global_rewards,
        )
