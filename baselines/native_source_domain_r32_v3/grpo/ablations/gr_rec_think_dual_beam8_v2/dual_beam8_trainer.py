"""Formal-ready Think G4 + per-CoT Beam8 dual-objective trainer.

The CoT branch is ordinary group-relative PPO over one global G4. The SID
branch is explicitly Beam-selected PPO-style optimization over four independent
G8 groups. Beam candidates are generated once and cached for both policy
iterations; they are never flattened into a G32 normalization group.
"""
from __future__ import annotations

import hashlib
import json
import time

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import TrainerCallback

from grpo_beam_domain import build_fixed_domain_beam_input, parse_fixed_domain_beam_sid
from grpo_model import encode_prompt, generate_batch
from grpo_sid import parse_sid, q_reward, think_reward
from grpo_trl_trainer import RecGRPOTrainer

COT_G = 4
SID_G = 8
LAMBDA_COT = 1.0
LAMBDA_SID = 1.0


def population_advantages(values, eps=1e-4):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("advantage group must be non-empty")
    std = values.std(correction=0)
    if bool(torch.isclose(std, torch.zeros_like(std))):
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


def independent_sid_advantages(reward_groups):
    """Normalize four G8 groups independently, never as one G32."""
    if len(reward_groups) != COT_G or any(len(group) != SID_G for group in reward_groups):
        raise ValueError("SID reward topology must be exactly 4 independent G8 groups")
    return torch.stack([population_advantages(group) for group in reward_groups])


def assert_one_global_group(group_ids):
    if len(group_ids) != COT_G or len(set(group_ids)) != 1:
        raise RuntimeError(f"DUAL_BEAM8_G4_GROUP_MIX: {group_ids}")


def reward_level(reward):
    return {-1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
            0.5: "A", 2.0: "AB", 8.0: "EXACT"}.get(float(reward), "unknown")


def rollout_fingerprint(records):
    payload = [{"gid": row["recommendation_group_id"], "cot": row["cot_ids"],
                "beam": row["beam_candidate_ids"], "cot_reward": row["cot_reward"],
                "sid_rewards": row["sid_rewards"]} for row in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ThinkG4SingleGroupSampler(Sampler):
    """Yield one row as a rank-aligned global G4, repeated for two PPO passes."""

    def __init__(self, data_source, repeat_count=2, shuffle=False, seed=None):
        del seed
        if shuffle:
            raise ValueError("dual Beam8 sampler requires shuffle=False")
        if int(repeat_count) != 2:
            raise ValueError("dual Beam8 num_iterations=2 is a hard contract")
        self.data_source = data_source
        self.repeat_count = 2

    def __iter__(self):
        for index in range(len(self.data_source)):
            for _ in range(self.repeat_count):
                yield from [index] * COT_G

    def __len__(self):
        return len(self.data_source) * self.repeat_count * COT_G


class DualBeam8Runtime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_local = None
        self.beam_calls = 0
        self.fresh_rollouts = 0

    def score_local_cot(self, prompt, cot_ids, gold_sids, target_domain, group_id=None):
        started = time.perf_counter()
        gold_set = {sid for raw in gold_sids for sid in [parse_sid(raw)] if sid is not None}
        close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(close_ids) != 1 or int(close_ids[0]) not in cot_ids:
            raise RuntimeError("DUAL_BEAM8_COT_NOT_CLOSED")
        close_at = list(cot_ids).index(int(close_ids[0]))
        cot_trim_ids = [int(token) for token in cot_ids[: close_at + 1]]
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        beam_context_ids, prefix_text, prefix_ids = build_fixed_domain_beam_input(
            self.tokenizer, prompt_ids, cot_trim_ids, target_domain)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                _texts, generated_ids = generate_batch(
                    self.model, self.tokenizer, [beam_context_ids], min_new_tokens=3,
                    max_new_tokens=3, num_beams=SID_G, num_return_sequences=SID_G,
                    do_sample=False, return_ids=True)
        finally:
            self.model.train(was_training)
        self.beam_calls += 1
        if len(generated_ids) != SID_G or any(len(ids) != 3 for ids in generated_ids):
            raise RuntimeError("DUAL_BEAM8_NOT_EXACT_8XABC3")
        beam_sids = [parse_fixed_domain_beam_sid(self.tokenizer, ids, target_domain)
                     for ids in generated_ids]
        sid_rewards = [float(q_reward(sid, gold_set)) for sid in beam_sids]
        cot_reward, exact, ab, a = think_reward(beam_sids, gold_set)
        record = {
            "recommendation_group_id": group_id, "target_domain": target_domain,
            "cot_ids": cot_trim_ids,
            "cot_text": self.tokenizer.decode(cot_trim_ids, skip_special_tokens=False),
            "closed": True, "cot_length": len(cot_trim_ids),
            "beam_context_ids": list(beam_context_ids), "domain_prefix": prefix_text,
            "domain_prefix_ids": list(prefix_ids),
            "beam_candidate_ids": [[int(token) for token in ids] for ids in generated_ids],
            "beam_sids": beam_sids, "sid_rewards": sid_rewards,
            "sid_reward_levels": [reward_level(value) for value in sid_rewards],
            "sid_advantages": population_advantages(sid_rewards).tolist(),
            "sid_population_std": float(torch.tensor(sid_rewards).std(correction=0)),
            "sid_zero_std": len(set(sid_rewards)) == 1,
            "cot_reward": float(cot_reward), "exact": int(exact), "ab": int(ab),
            "a": int(a), "invalid": int(sum(sid is None for sid in beam_sids)),
            "beam8_wall_sec": time.perf_counter() - started,
        }
        self.last_local = record
        return record


def make_dual_beam8_reward_func(runtime):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        routes = kwargs.get("route")
        if routes is None or any(route != "think" for route in routes):
            raise RuntimeError("dual Beam8 reward requires Think-only samples")
        records = [runtime.score_local_cot(prompt, ids, golds, domain, gid)
                   for prompt, ids, golds, domain, gid in zip(
                       prompts, completion_ids, kwargs["all_gold_sids"],
                       kwargs["target_domain"], kwargs["recommendation_group_id"])]
        runtime.last_local = records[0] if len(records) == 1 else records
        return [record["cot_reward"] for record in records]
    reward_func.__name__ = "dual_beam8_cot_reward"
    reward_func.dual_beam8_runtime = runtime
    return reward_func


class ThinkDualBeam8Trainer(RecGRPOTrainer):
    def __init__(self, *args, lambda_cot=LAMBDA_COT, lambda_sid=LAMBDA_SID, **kwargs):
        reward_funcs = kwargs.get("reward_funcs") or (args[3] if len(args) >= 4 else [])
        runtime = next((getattr(fn, "dual_beam8_runtime", None) for fn in reward_funcs
                        if getattr(fn, "dual_beam8_runtime", None) is not None), None)
        if runtime is None:
            raise RuntimeError("ThinkDualBeam8Trainer requires dual Beam8 runtime")
        if float(lambda_cot) != 1.0 or float(lambda_sid) != 1.0:
            raise ValueError("first formal dual Beam8 experiment freezes lambdas at 1.0")
        self._dual_beam8_runtime = runtime
        self.lambda_cot, self.lambda_sid = 1.0, 1.0
        self._sid_rollout = None
        self._dual_policy_epoch = 0
        self._dual_rollout_fingerprint = None
        super().__init__(*args, **kwargs)
        self.add_callback(FinalStepSaveCallback())

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return ThinkG4SingleGroupSampler(dataset, repeat_count=self.num_iterations,
                                         shuffle=self.shuffle_dataset, seed=self.args.seed)

    def _route_generation_contract(self, route):
        if route != "think":
            raise RuntimeError("dual Beam8 is Think-only")
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, batch):
        rows = batch if isinstance(batch, list) else []
        if rows and any(row.get("route") != "think" for row in rows):
            raise RuntimeError("dual Beam8 training dataset must be Think-only")
        return super()._prepare_inputs(batch)

    def _pad_sid_sequences(self, record):
        candidates = record["beam_candidate_ids"]
        if len(candidates) != SID_G or any(len(candidate) != 3 for candidate in candidates):
            raise RuntimeError("Beam8 SID objective requires exactly 8xABC3")
        sequences = [list(record["beam_context_ids"]) + list(candidate) for candidate in candidates]
        max_len = max(map(len, sequences))
        rows = [[self.pad_token_id] * (max_len - len(row)) + row for row in sequences]
        masks = [[0] * (max_len - len(row)) + [1] * len(row) for row in sequences]
        device = self.accelerator.device
        return (torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(masks, dtype=torch.long, device=device))

    @staticmethod
    def _gather_records(record):
        if not dist.is_initialized():
            return [record]
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        return records

    def _build_sid_rollout(self, cot_output):
        record = self._dual_beam8_runtime.last_local
        if isinstance(record, list):
            if len(record) != 1:
                raise RuntimeError("formal shape requires one local CoT per rank")
            record = record[0]
        records = self._gather_records(record)
        assert_one_global_group([row["recommendation_group_id"] for row in records])
        cot_rewards = [row["cot_reward"] for row in records]
        cot_advantages = population_advantages(cot_rewards)
        local_rank = self.accelerator.process_index
        expected = cot_advantages[local_rank:local_rank + len(cot_output["advantages"])]
        if not torch.allclose(cot_output["advantages"].float().cpu(), expected, atol=1e-6):
            raise RuntimeError("DUAL_BEAM8_COT_ADVANTAGE_PARITY_FAILED")
        if cot_output.get("old_per_token_logps") is None:
            raise RuntimeError("DUAL_BEAM8_COT_OLD_LOGP_MUST_BE_FULL_FORWARD")
        if cot_output["old_per_token_logps"].requires_grad:
            raise RuntimeError("DUAL_BEAM8_COT_OLD_LOGP_NOT_DETACHED")
        input_ids, attention_mask = self._pad_sid_sequences(record)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention_mask, 3, compute_entropy=False)
        if old_logps.shape != (SID_G, 3) or old_logps.requires_grad:
            raise RuntimeError("DUAL_BEAM8_SID_OLD_LOGP_CONTRACT_FAILED")
        self._sid_rollout = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "old_per_token_logps": old_logps.detach(),
            "advantages": population_advantages(record["sid_rewards"]).to(input_ids.device),
            "record": record, "global_records": records,
        }
        self._dual_rollout_fingerprint = rollout_fingerprint(records)
        self._dual_policy_epoch = 0
        self._dual_beam8_runtime.fresh_rollouts += 1
        if self.accelerator.is_main_process and self._monitor.enabled:
            self._monitor.write_dual_beam8({
                "step": int(self.state.global_step), "rollout_id": int(self._smoke_rollout_id),
                "rollout_fingerprint": self._dual_rollout_fingerprint,
                "recommendation_group_id": records[0]["recommendation_group_id"],
                "target_domain": records[0]["target_domain"], "cots": records,
                "cot_rewards": cot_rewards, "cot_advantages": cot_advantages.tolist(),
                "cot_population_std": float(torch.tensor(cot_rewards).std(correction=0)),
                "cot_zero_std": len(set(cot_rewards)) == 1,
                "normalization_topology": "G4 + 4 independent G8; never G32",
            })

    def _generate_and_score_completions(self, inputs):
        before_calls = self._dual_beam8_runtime.beam_calls
        output = super()._generate_and_score_completions(inputs)
        self._build_sid_rollout(output)
        if self._dual_beam8_runtime.beam_calls - before_calls != len(inputs):
            raise RuntimeError("DUAL_BEAM8_EXPECTED_ONE_BEAM_CALL_PER_LOCAL_COT")
        return output

    def _branch_loss(self, model, input_ids, attention_mask, action_mask,
                     old_logps, advantages, branch):
        current_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, action_mask.size(1), compute_entropy=False)
        log_ratio = current_logps - old_logps
        ratio = log_ratio.exp()
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token = -torch.min(ratio * advantages.unsqueeze(1),
                               clipped * advantages.unsqueeze(1))
        loss = ((per_token * action_mask).sum(-1) /
                action_mask.sum(-1).clamp(min=1)).mean()
        loss = loss / self.current_gradient_accumulation_steps
        with torch.no_grad():
            selected = log_ratio[action_mask.bool()].float()
            ratio_selected = selected.exp()
            stats = {
                f"{branch}_loss": float(loss.detach()),
                f"{branch}_ratio_mean": float(ratio_selected.mean()),
                f"{branch}_clip_fraction": float(((ratio_selected - 1).abs() > self.epsilon_low).float().mean()),
                f"{branch}_approx_kl": float((ratio_selected - 1 - selected).mean()),
                f"{branch}_action_tokens": int(action_mask.sum()),
                f"{branch}_action_logp_grad_norm": float((advantages.unsqueeze(1).expand_as(action_mask) * action_mask).float().norm()),
            }
        return loss, stats

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        if self._sid_rollout is None:
            raise RuntimeError("SID rollout unavailable")
        if self._dual_policy_epoch >= 2:
            raise RuntimeError("DUAL_BEAM8_ROLLOUT_REUSED_MORE_THAN_TWO_ITERATIONS")
        prompt_completion = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        prompt_completion_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        cot_loss, cot_stats = self._branch_loss(
            model, prompt_completion, prompt_completion_mask, inputs["completion_mask"],
            inputs["old_per_token_logps"], inputs["advantages"], "cot")
        sid = self._sid_rollout
        sid_action_mask = torch.ones((SID_G, 3), dtype=torch.long, device=sid["input_ids"].device)
        sid_loss, sid_stats = self._branch_loss(
            model, sid["input_ids"], sid["attention_mask"], sid_action_mask,
            sid["old_per_token_logps"], sid["advantages"], "sid")
        total = self.lambda_cot * cot_loss + self.lambda_sid * sid_loss
        self._dual_policy_epoch += 1
        stats = {**cot_stats, **sid_stats, "total_loss": float(total.detach()),
                 "policy_iteration": self._dual_policy_epoch,
                 "rollout_fingerprint": self._dual_rollout_fingerprint,
                 "beam_calls": self._dual_beam8_runtime.beam_calls}
        if self._smoke_log:
            self._smoke_log[-1].update(stats)
        return total

    def log(self, logs, start_time=None):
        result = super().log(logs, start_time)
        if (self.accelerator.is_main_process and self._monitor.enabled and
                self._smoke_log and "loss" in logs):
            rollout = self._smoke_log[-1]
            self._monitor.write_dual_beam8({
                "type": "optimization", "step": int(self.state.global_step),
                "rollout_id": rollout.get("rollout_id"),
                "rollout_fingerprint": rollout.get("rollout_fingerprint"),
                "policy_iteration": rollout.get("policy_iteration"),
                "cot_loss": rollout.get("cot_loss"), "sid_loss": rollout.get("sid_loss"),
                "total_loss": rollout.get("total_loss"),
                "cot_ratio_mean": rollout.get("cot_ratio_mean"),
                "sid_ratio_mean": rollout.get("sid_ratio_mean"),
                "cot_clip_fraction": rollout.get("cot_clip_fraction"),
                "sid_clip_fraction": rollout.get("sid_clip_fraction"),
                "cot_approx_kl": rollout.get("cot_approx_kl"),
                "sid_approx_kl": rollout.get("sid_approx_kl"),
                "cot_action_tokens": rollout.get("cot_action_tokens"),
                "sid_action_tokens": rollout.get("sid_action_tokens"),
                "cot_action_logp_grad_norm": rollout.get("cot_action_logp_grad_norm"),
                "sid_action_logp_grad_norm": rollout.get("sid_action_logp_grad_norm"),
                "total_lora_grad_norm": logs.get("grad_norm"),
                "generation_wall_sec": rollout.get("gen_wall_sec"),
                "gpu_peak_memory_bytes": (torch.cuda.max_memory_allocated()
                                           if torch.cuda.is_available() else 0),
            })
        return result


class FinalStepSaveCallback(TrainerCallback):
    """Force a full Trainer checkpoint at max_steps (3090 in the formal run)."""

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) == int(args.max_steps):
            control.should_save = True
        return control


def audit_dual_sampler(dataset, sampler):
    rows = list(dataset)
    if not rows or any(row.get("route") != "think" for row in rows):
        raise RuntimeError("dual Beam8 requires Think-only rows")
    gids = [row["recommendation_group_id"] for row in rows]
    if len(gids) != len(set(gids)):
        raise RuntimeError("dataset must contain one Think row per business group")
    optimizer_steps = len(rows) * 2
    return {
        "selected_groups": len(rows), "trained_groups": len(rows), "dropped_groups": 0,
        "think_unique_groups": len(rows), "nothink_unique_groups": 0,
        "think_rollouts": len(rows), "nothink_rollouts": 0,
        "fresh_rollout_count": len(rows), "unique_groups_per_global_rollout": 1,
        "cot_candidates_per_group": COT_G, "sid_candidates_per_cot": SID_G,
        "sid_candidates_per_business_group": COT_G * SID_G,
        "repeat_count": 2, "num_iterations": 2,
        "optimizer_steps": optimizer_steps, "think_optimizer_steps": optimizer_steps,
        "nothink_optimizer_steps": 0, "route_schedule_preview": ["think"] * min(24, len(rows)),
        "rollout_group_ids_preview": gids[:8],
    }
