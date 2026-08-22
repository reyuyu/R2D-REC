"""Think-only fixed probes enriched with Composite Interest evidence."""
from __future__ import annotations

import random
import statistics
import time
import torch
from transformers import TrainerCallback

from grpo_probe import FixedProbeEvaluator
from .composite_trainer import score_candidate
from .interest_metric import population_advantages


class CompositeThinkProbeEvaluator(FixedProbeEvaluator):
    """Reuse production Think generation/Beam while omitting NoThink probes."""

    def __init__(self, *args, probe_rounds, probe_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_rounds = tuple(tuple(round_ids) for round_ids in probe_rounds)
        self.probe_steps = tuple(int(step) for step in probe_steps)
        flattened = tuple(group_id for round_ids in self.probe_rounds for group_id in round_ids)
        if len(self.probe_rounds) != 3 or any(len(round_ids) != 4 for round_ids in self.probe_rounds):
            raise ValueError("Composite probes require three rounds of four ranks")
        if flattened != tuple(self.group_ids):
            raise ValueError("probe rounds must exactly cover fixed group_ids in order")

    def _think_round(self, round_ids):
        original_group_ids = self.group_ids
        try:
            self.group_ids = list(round_ids)
            return self._think()
        finally:
            self.group_ids = original_group_ids

    def _think(self):
        result = super()._think()
        gold_cot = self.records[result["group_id"]]["gold_cot"]
        scored = [
            score_candidate(item["completion"], gold_cot, result["prompt"], item["reward"])
            for item in result["candidates"]
        ]
        advantages = population_advantages([item["composite_reward"] for item in scored])
        for candidate, score, advantage in zip(result["candidates"], scored, advantages):
            candidate.update(score)
            candidate["final_sequence_advantage"] = advantage
        return result

    def evaluate(self, step, reason):
        if self.last_step == step or self._already_complete(step):
            self.last_step = step
            return
        if self.trainer.accelerator.num_processes != 4:
            raise RuntimeError("fixed Think probe requires four ranks")
        self.trainer.accelerator.wait_for_everyone()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.trainer.accelerator.device)
        python_rng = random.getstate()
        previous_route = (
            self.trainer.num_generations, self.trainer.args.temperature,
            self.trainer.args.top_p, self.trainer.temperature, self.trainer.top_p,
        )
        model = self.trainer.model
        had_beam_stats = hasattr(model, "_beam_stats")
        previous_beam_stats = getattr(model, "_beam_stats", None)
        started = time.perf_counter()
        try:
            self._set_seed()
            think_by_rank = []
            for round_index, round_ids in enumerate(self.probe_rounds):
                parts = self._gather(self._think_round(round_ids))
                for part in parts:
                    part["probe_round"] = round_index
                think_by_rank.extend(parts)
            torch.cuda.synchronize()
            rank_walls = self._gather(time.perf_counter() - started)
            if self.trainer.accelerator.is_main_process:
                for part in think_by_rank:
                    candidates = part["candidates"]
                    beam = [float(item["beam_raw"]) for item in candidates]
                    composite = [float(item["composite_reward"]) for item in candidates]
                    self.monitor.write_probe({
                        "step": int(step), "reason": reason,
                        "probe_round": part["probe_round"],
                        "group_id": part["group_id"],
                        "target_domain": part.get("target_domain"),
                        "gold_sids": part["gold_sids"],
                        "think_prompt": part["prompt"],
                        "probe_route": "think_only",
                        "think": {
                            "beam_reward_mean": statistics.fmean(beam),
                            "composite_reward_mean": statistics.fmean(composite),
                            "composite_reward_std": statistics.pstdev(composite),
                            "candidates": candidates,
                        },
                        "think_generation_wall_sec": part["generation_wall_sec"],
                        "think_wall_sec": part["wall_sec"],
                        "probe_wall_sec": max(rank_walls), "seed": self.seed,
                    })
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.trainer.accelerator.device)
            random.setstate(python_rng)
            (self.trainer.num_generations, self.trainer.args.temperature,
             self.trainer.args.top_p, self.trainer.temperature,
             self.trainer.top_p) = previous_route
            if getattr(self.trainer, "generation_config", None) is not None:
                self.trainer.generation_config.temperature = previous_route[1]
                self.trainer.generation_config.top_p = previous_route[2]
            if had_beam_stats:
                model._beam_stats = previous_beam_stats
            elif hasattr(model, "_beam_stats"):
                delattr(model, "_beam_stats")
            self.trainer.accelerator.wait_for_everyone()
        self.last_step = step


class CompositeProbeCallback(TrainerCallback):
    """Run all 12 probes only at the experiment's explicit milestones."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def _due(self, step):
        return int(step) in self.evaluator.probe_steps

    def on_train_begin(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self._due(step):
            self.evaluator.evaluate(step, "baseline")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self._due(step):
            self.evaluator.evaluate(step, "milestone")

    def on_train_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self._due(step):
            self.evaluator.evaluate(step, "final")
