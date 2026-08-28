"""Think-only G8 trainer with zero-std rerolls and answer-suffix loss."""

from __future__ import annotations

import json
import time

import torch
from torch.utils.data import Sampler

from grpo_trl_trainer import ROUTE_ID, ROUTE_TEMP, ROUTE_TOP_P, RecGRPOTrainer
from .suffix_objective import (
    GROUP_SIZE,
    MAX_RESAMPLE_ROUNDS,
    parse_suffix_completion,
    resample_decision,
    suffix_mask_for_ids,
    suffix_reward,
)


class ThinkG8SingleGroupSampler(Sampler):
    """One unique prompt per global generation batch, repeated G8."""

    def __init__(self, data_source, repeat_count=2, shuffle=False, seed=None):
        if shuffle:
            raise ValueError("Think suffix sampler requires shuffle=False")
        self.data_source = data_source
        self.repeat_count = int(repeat_count)
        self._chunks = [("think", [index]) for index in range(len(data_source))]

    def __iter__(self):
        for _route, indices in self._chunks:
            index = indices[0]
            for _ in range(self.repeat_count):
                for _ in range(GROUP_SIZE):
                    yield index

    def __len__(self):
        return len(self.data_source) * self.repeat_count * GROUP_SIZE


def make_think_suffix_reward_func(tokenizer, beam32_fn=None):
    """NoThink-v1 SID reward applied to complete Think+answer rollouts."""
    del beam32_fn

    def reward_func(prompts, completions, completion_ids, **kwargs):
        del prompts, completions
        routes = kwargs.get("route")
        golds = kwargs["all_gold_sids"]
        if routes is None or any(route != "think" for route in routes):
            raise RuntimeError("Think suffix reward received a non-Think sample")
        rewards = []
        for candidate_ids, candidate_golds in zip(completion_ids, golds):
            text = tokenizer.decode(candidate_ids, skip_special_tokens=False)
            reward, _ = suffix_reward(text, candidate_golds)
            rewards.append(reward)
        return rewards

    reward_func.__name__ = "think_suffix_sid_reward"
    return reward_func


class ThinkSuffixSIDTrainer(RecGRPOTrainer):
    """G8 Think policy whose loss starts strictly after </think>."""

    def __init__(self, *args, **kwargs):
        self._active_generation_batch = None
        self._resample_runtime = None
        # This Think route optimizes the answer suffix, so generation must
        # continue past </think>. The base trainer defaults to stopping there.
        self._stop_think_at_closure = False
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return ThinkG8SingleGroupSampler(
            dataset,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _set_route_config(self, route):
        if route != "think":
            raise RuntimeError(f"Think suffix trainer received route={route!r}")
        self._active_route = route
        self.num_generations = GROUP_SIZE
        self.args.temperature = ROUTE_TEMP["think"]
        self.args.top_p = ROUTE_TOP_P["think"]
        self.temperature = ROUTE_TEMP["think"]
        self.top_p = ROUTE_TOP_P["think"]
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.temperature = self.temperature
            self.generation_config.top_p = self.top_p
            self.generation_config.stop_strings = None

    def _route_generation_contract(self, route):
        if route != "think":
            raise RuntimeError(f"unexpected route contract: {route!r}")
        return {"group_size": GROUP_SIZE, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, generation_batch):
        if isinstance(generation_batch, list):
            if not generation_batch or any(row.get("route") != "think" for row in generation_batch):
                raise RuntimeError("Think suffix trainer requires a homogeneous Think batch")
            self._active_generation_batch = generation_batch
        return super()._prepare_inputs(generation_batch)

    def _generate_single_turn(self, prompts, images=None):
        """Generate complete CoT+answer; reroll the global G8 on zero reward std."""
        if not isinstance(self._active_generation_batch, list):
            raise RuntimeError("missing active generation batch for zero-std rescue")
        rejected = []
        accepted = None
        accepted_reward_vector = None
        final_decision = None
        started = time.monotonic()
        for attempt in range(MAX_RESAMPLE_ROUNDS + 1):
            generated = super()._generate_single_turn(prompts, images)
            _prompt_ids, completion_ids, _logprobs, _forward_kwargs = generated
            local_rewards = []
            for candidate_ids, row in zip(completion_ids, self._active_generation_batch):
                text = self.processing_class.decode(candidate_ids, skip_special_tokens=False)
                reward, _ = suffix_reward(text, row["all_gold_sids"])
                local_rewards.append(reward)
            local_tensor = torch.tensor(local_rewards, dtype=torch.float32, device=self.accelerator.device)
            global_rewards = self.accelerator.gather(local_tensor).tolist()
            decision = resample_decision(global_rewards, attempt)
            if decision["retry"]:
                rejected.append(global_rewards)
                continue
            accepted = generated
            accepted_reward_vector = global_rewards
            final_decision = decision
            break
        if accepted is None or final_decision is None:
            raise RuntimeError("zero-std rescue did not produce a terminal round")
        self._resample_runtime = {
            **final_decision,
            "resample_rounds_used": final_decision["resample_round"],
            "rejected_reward_vectors": rejected,
            "accepted_reward_vector": accepted_reward_vector,
            "zero_std_rescued": bool(rejected and not final_decision["zero_std"]),
            "zero_std_rescue_exhausted": final_decision["exhausted"],
            "resample_wall_sec": time.monotonic() - started,
        }
        return accepted

    def _rollout_monitor_fields(self):
        return dict(self._resample_runtime or {})

    def _capture_global_completion_ids(self):
        return True

    def _monitor_candidate_fields(self, route, candidate_ids, raw_text, reward):
        if route != "think":
            return {}
        text = self.processing_class.decode(candidate_ids, skip_special_tokens=False)
        parsed = parse_suffix_completion(text)
        close_ids = self.processing_class.encode("</think>", add_special_tokens=False)
        suffix_mask = suffix_mask_for_ids(candidate_ids, close_ids)
        return {
            "completion": text,
            "parsed_sid": list(parsed.parsed_sid) if parsed.parsed_sid else None,
            "reward_level": reward,
            "closed": parsed.closed,
            "parser_status": parsed.parser_status,
            "sid_count": parsed.sid_count,
            "all_parsed_sids": [list(sid) for sid in parsed.all_parsed_sids],
            "multi_sid_output": parsed.multi_sid_output,
            "suffix_token_count": sum(suffix_mask),
            "cot_token_count": len(candidate_ids) - sum(suffix_mask),
            "loss_scope": "tokens_after_think_close_only",
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        close_ids = self.processing_class.encode("</think>", add_special_tokens=False)
        masks = []
        for ids, completion_mask in zip(output["completion_ids"], output["completion_mask"]):
            valid_length = int(completion_mask.sum().item())
            masks.append(suffix_mask_for_ids(ids.tolist(), close_ids, valid_length))
        output["suffix_loss_mask"] = torch.tensor(
            masks,
            dtype=output["completion_mask"].dtype,
            device=output["completion_mask"].device,
        )
        return output

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("Think suffix trainer does not return model outputs")
        started = time.monotonic()
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        loss_mask = inputs.get("suffix_loss_mask")
        if loss_mask is None or loss_mask.shape != completion_mask.shape:
            raise RuntimeError("missing batch-aligned suffix_loss_mask")
        if bool((loss_mask > completion_mask).any()):
            raise RuntimeError("suffix_loss_mask includes padded completion tokens")

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Full completion attention is intentional: SID prediction must see its CoT.
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, completion_ids.size(1), compute_entropy=False,
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            ratio = torch.clamp(ratio, max=self.args.delta)
        advantages = inputs["advantages"].unsqueeze(1)
        per_token_loss = -torch.min(ratio * advantages, clipped * advantages)
        per_sample_loss = (
            (per_token_loss * loss_mask).sum(-1)
            / loss_mask.sum(-1).clamp(min=1.0)
        )
        loss = per_sample_loss.mean() / self.current_gradient_accumulation_steps

        with torch.no_grad():
            active = loss_mask.bool()
            flat = log_ratio[active].float()
            ratio_flat = torch.exp(flat) if flat.numel() else flat
            policy_epoch = self._smoke_policy_epoch.get(self._smoke_rollout_id, 0)
            entry = {
                f"policy_epoch_{policy_epoch + 1}": 1,
                f"step_{policy_epoch + 1}": self.state.global_step,
                f"ratio_mean_ep{policy_epoch + 1}": float(ratio_flat.mean()) if flat.numel() else 1.0,
                f"clip_fraction_ep{policy_epoch + 1}": (
                    float(((ratio_flat - 1.0).abs() > self.epsilon_low).float().mean())
                    if flat.numel() else 0.0
                ),
                f"approx_kl_ep{policy_epoch + 1}": (
                    float((ratio_flat - 1.0 - flat).mean()) if flat.numel() else 0.0
                ),
                f"policy_fwb_sec_ep{policy_epoch + 1}": round(time.monotonic() - started, 2),
                f"action_tokens_ep{policy_epoch + 1}": float(loss_mask.sum()),
                f"context_cot_tokens_ep{policy_epoch + 1}": float(
                    (completion_mask - loss_mask).clamp(min=0).sum()
                ),
            }
            self._smoke_policy_epoch[self._smoke_rollout_id] = policy_epoch + 1
            self._smoke_log[-1].update(entry)
        return loss


def audit_think_g8_sampler(dataset, sampler):
    rows = list(dataset)
    if not rows or any(row.get("route") != "think" for row in rows):
        raise RuntimeError("Think-only dataset contains an invalid route")
    group_ids = [row["recommendation_group_id"] for row in rows]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("Think-only dataset must contain one row per group")
    optimizer_steps = len(rows) * sampler.repeat_count
    return {
        "selected_groups": len(rows),
        "trained_groups": len(rows),
        "dropped_groups": 0,
        "think_unique_groups": len(rows),
        "nothink_unique_groups": 0,
        "think_rollouts": len(rows),
        "nothink_rollouts": 0,
        "fresh_rollout_count": len(rows),
        "unique_groups_per_global_rollout": 1,
        "candidates_per_round": GROUP_SIZE,
        "max_resample_rounds": MAX_RESAMPLE_ROUNDS,
        "max_candidates_per_group": GROUP_SIZE * (MAX_RESAMPLE_ROUNDS + 1),
        "optimizer_steps": optimizer_steps,
        "think_optimizer_steps": optimizer_steps,
        "nothink_optimizer_steps": 0,
        "repeat_count": sampler.repeat_count,
        "num_iterations": sampler.repeat_count,
        "route_schedule_preview": ["think"] * min(24, len(rows)),
        "rollout_group_ids_preview": group_ids[:8],
    }
