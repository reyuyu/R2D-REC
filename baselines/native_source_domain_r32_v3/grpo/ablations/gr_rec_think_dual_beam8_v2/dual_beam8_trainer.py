"""Think G4 + per-CoT Beam8 dual-objective GRPO.

One global recommendation group produces four stochastic CoTs (one per rank on
4 GPUs).  Each CoT is evaluated by a deterministic fixed-domain Beam8 ABC3
continuation.  Two independent group-relative advantages are used:

* CoT advantage: the four Beam8 aggregate rewards are normalized as one G4.
  Loss applies only to sampled CoT action tokens.
* SID advantage: the eight hierarchical SID rewards under each CoT are
  normalized inside that CoT's own G8.  Loss applies only to the three generated
  A/B/C tokens; prompt, CoT, and fixed domain prefix are context only.
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler

from grpo_beam_domain import build_fixed_domain_beam_input, parse_fixed_domain_beam_sid
from grpo_model import encode_prompt, generate_batch
from grpo_sid import parse_sid, q_reward, think_reward
from grpo_trl_trainer import RecGRPOTrainer

COT_G = 4
SID_G = 8


def population_advantages(values, eps=1e-4):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("advantage group must be non-empty")
    std = values.std(correction=0)
    if bool(torch.isclose(std, torch.zeros_like(std))):
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


class ThinkG4SingleGroupSampler(Sampler):
    """Repeat one dataset row four times globally, then replay it for PPO epoch 2."""

    def __init__(self, data_source, repeat_count=2, shuffle=False, seed=None):
        if shuffle:
            raise ValueError("dual Beam8 sampler requires shuffle=False")
        self.data_source = data_source
        self.repeat_count = int(repeat_count)

    def __iter__(self):
        for index in range(len(self.data_source)):
            for _ in range(self.repeat_count):
                for _ in range(COT_G):
                    yield index

    def __len__(self):
        return len(self.data_source) * self.repeat_count * COT_G


class DualBeam8Runtime:
    """Per-process side channel populated by the Think reward function."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_local = None

    def score_local_cot(self, prompt, cot_ids, gold_sids, target_domain, group_id=None):
        gold_set = {sid for raw in gold_sids for sid in [parse_sid(raw)] if sid is not None}
        cot_text = self.tokenizer.decode(cot_ids, skip_special_tokens=False)
        close_index = cot_text.find("</think>")
        cot_trim = cot_text[: close_index + len("</think>")] if close_index >= 0 else cot_text
        cot_trim_ids = self.tokenizer.encode(cot_trim, add_special_tokens=False)
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        beam_context_ids, prefix_text, prefix_ids = build_fixed_domain_beam_input(
            self.tokenizer, prompt_ids, cot_trim_ids, target_domain,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                _texts, generated_ids = generate_batch(
                    self.model,
                    self.tokenizer,
                    [beam_context_ids],
                    min_new_tokens=3,
                    max_new_tokens=3,
                    num_beams=SID_G,
                    num_return_sequences=SID_G,
                    return_ids=True,
                )
        finally:
            self.model.train(was_training)

        beam_sids = [
            parse_fixed_domain_beam_sid(self.tokenizer, ids, target_domain)
            for ids in generated_ids
        ]
        sid_rewards = [q_reward(sid, gold_set) for sid in beam_sids]
        cot_reward, exact, ab, a = think_reward(beam_sids, gold_set)
        record = {
            "recommendation_group_id": group_id,
            "target_domain": target_domain,
            "cot_ids": list(cot_ids),
            "cot_text": cot_text,
            "closed": close_index >= 0,
            "beam_context_ids": list(beam_context_ids),
            "domain_prefix": prefix_text,
            "domain_prefix_ids": list(prefix_ids),
            "beam_candidate_ids": [list(ids) for ids in generated_ids],
            "beam_sids": beam_sids,
            "sid_rewards": sid_rewards,
            "cot_reward": float(cot_reward),
            "exact": int(exact),
            "ab": int(ab),
            "a": int(a),
            "invalid": int(sum(sid is None for sid in beam_sids)),
        }
        self.last_local = record
        return record


def make_dual_beam8_reward_func(runtime: DualBeam8Runtime):
    """Return one scalar Beam8 quality reward per sampled CoT."""

    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        routes = kwargs.get("route")
        if routes is None or any(route != "think" for route in routes):
            raise RuntimeError("dual Beam8 reward requires Think-only samples")
        golds = kwargs["all_gold_sids"]
        domains = kwargs["target_domain"]
        group_ids = kwargs.get("recommendation_group_id") or [None] * len(prompts)
        rewards = []
        # Production shape is one local CoT per rank. Keep the loop for CPU tests.
        local_records = []
        for prompt, cot_ids, candidate_golds, domain, gid in zip(
            prompts, completion_ids, golds, domains, group_ids
        ):
            record = runtime.score_local_cot(prompt, cot_ids, candidate_golds, domain, gid)
            local_records.append(record)
            rewards.append(record["cot_reward"])
        runtime.last_local = local_records[0] if len(local_records) == 1 else local_records
        return rewards

    reward_func.__name__ = "dual_beam8_cot_reward"
    reward_func.dual_beam8_runtime = runtime
    return reward_func


class ThinkDualBeam8Trainer(RecGRPOTrainer):
    """Combine the upstream G4 CoT PPO objective with a per-CoT G8 SID objective."""

    def __init__(self, *args, **kwargs):
        reward_funcs = kwargs.get("reward_funcs")
        if reward_funcs is None and len(args) >= 4:
            reward_funcs = args[3]
        runtime = None
        for reward_func in reward_funcs or []:
            runtime = getattr(reward_func, "dual_beam8_runtime", runtime)
        if runtime is None:
            raise RuntimeError("ThinkDualBeam8Trainer requires dual Beam8 reward runtime")
        self._dual_beam8_runtime = runtime
        self._sid_rollout = None
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return ThinkG4SingleGroupSampler(
            dataset,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _route_generation_contract(self, route):
        if route != "think":
            return super()._route_generation_contract(route) if hasattr(super(), "_route_generation_contract") else {
                "group_size": 8, "temperature": 1.0, "top_p": 1.0,
            }
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, generation_batch):
        if isinstance(generation_batch, list) and generation_batch:
            if any(row.get("route") != "think" for row in generation_batch):
                raise RuntimeError("dual Beam8 training dataset must be Think-only")
        return super()._prepare_inputs(generation_batch)

    def _pad_sid_sequences(self, record):
        pad_id = int(self.pad_token_id)
        sequences = [
            list(record["beam_context_ids"]) + list(candidate)
            for candidate in record["beam_candidate_ids"]
        ]
        if len(sequences) != SID_G or any(len(candidate) != 3 for candidate in record["beam_candidate_ids"]):
            raise RuntimeError("Beam8 SID objective requires exactly eight strict 3-token candidates")
        max_len = max(len(sequence) for sequence in sequences)
        rows, masks = [], []
        for sequence in sequences:
            left = max_len - len(sequence)
            rows.append([pad_id] * left + sequence)
            masks.append([0] * left + [1] * len(sequence))
        device = self.accelerator.device
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.long, device=device),
        )

    def _build_sid_rollout(self):
        record = self._dual_beam8_runtime.last_local
        if isinstance(record, list):
            if len(record) != 1:
                raise RuntimeError("formal dual Beam8 shape requires one local CoT per rank")
            record = record[0]
        if not isinstance(record, dict):
            raise RuntimeError("missing local Beam8 record")
        input_ids, attention_mask = self._pad_sid_sequences(record)
        sid_advantages = population_advantages(record["sid_rewards"]).to(self.accelerator.device)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                3,
                compute_entropy=False,
            )
        if old_logps.shape != (SID_G, 3):
            raise RuntimeError(f"unexpected SID old-logp shape: {tuple(old_logps.shape)}")
        self._sid_rollout = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_per_token_logps": old_logps.detach(),
            "advantages": sid_advantages.detach(),
            "sid_rewards": torch.tensor(record["sid_rewards"], dtype=torch.float32, device=self.accelerator.device),
            "record": record,
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self._build_sid_rollout()
        return output

    def _sid_loss(self, model):
        rollout = self._sid_rollout
        if rollout is None:
            raise RuntimeError("SID rollout is unavailable")
        current_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            rollout["input_ids"],
            rollout["attention_mask"],
            3,
            compute_entropy=False,
        )
        old_logps = rollout["old_per_token_logps"]
        advantages = rollout["advantages"].unsqueeze(1)
        log_ratio = current_logps - old_logps
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            ratio = torch.clamp(ratio, max=self.args.delta)
        per_token_loss = -torch.min(ratio * advantages, clipped * advantages)
        # All eight candidates are strict ABC3, so each sample has exactly 3 actions.
        return per_token_loss.mean() / self.current_gradient_accumulation_steps

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cot_loss = super()._compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        sid_loss = self._sid_loss(model)
        return cot_loss + sid_loss


def audit_dual_sampler(dataset, sampler):
    rows = list(dataset)
    if not rows or any(row.get("route") != "think" for row in rows):
        raise RuntimeError("dual Beam8 training requires Think-only rows")
    gids = [row["recommendation_group_id"] for row in rows]
    if len(gids) != len(set(gids)):
        raise RuntimeError("dual Beam8 dataset must contain one Think row per group")
    optimizer_steps = len(rows) * sampler.repeat_count
    return {
        "selected_groups": len(rows),
        "trained_groups": len(rows),
        "think_unique_groups": len(rows),
        "fresh_rollout_count": len(rows),
        "cot_candidates_per_group": COT_G,
        "sid_candidates_per_cot": SID_G,
        "sid_candidates_per_business_group": COT_G * SID_G,
        "repeat_count": sampler.repeat_count,
        "optimizer_steps": optimizer_steps,
    }
