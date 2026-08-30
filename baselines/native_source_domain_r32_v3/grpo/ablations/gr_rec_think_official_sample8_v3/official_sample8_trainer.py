"""V3 reward math with an Official fixed-domain Sample8 ABC3 answer branch."""
from __future__ import annotations

import hashlib
import json
import time

import torch
import torch.distributed as dist

from grpo_beam_domain import build_fixed_domain_beam_input, parse_strict_abc3_ids
from grpo_model import encode_prompt
from grpo_sid import parse_sid, q_reward
from grpo_trl_trainer import RecGRPOTrainer
from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import (
    MAX_COT_CLOSURE_RETRIES,
    FinalStepSaveCallback,
    ResumeCadenceCallback,
    ThinkG4SingleGroupSampler,
    population_advantages,
)

COT_G = 4
SID_G = 8
OFFICIAL_SID_TOKENS = 3
LAMBDA_COT = 1.0
LAMBDA_SID = 1.0
TARGET_DOMAINS = frozenset(("video", "ad", "prod", "living"))


def reward_level(value):
    return {
        -1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
        0.5: "A", 2.0: "AB", 8.0: "EXACT",
    }.get(float(value), "unknown")


def official_action_mask(candidate_ids):
    if len(candidate_ids) != SID_G:
        raise ValueError("Official branch must contain Sample8")
    if any(len(candidate) != OFFICIAL_SID_TOKENS for candidate in candidate_ids):
        raise ValueError("Official branch must contain exact ABC3 continuations")
    return [[1] * OFFICIAL_SID_TOKENS for _ in candidate_ids]


def combine_losses(cot_loss, sid_loss):
    return LAMBDA_COT * cot_loss + LAMBDA_SID * sid_loss


def independent_official_advantages(groups):
    if len(groups) != COT_G or any(len(group) != SID_G for group in groups):
        raise ValueError("V3-Official requires exactly four independent Official G8 groups")
    return torch.stack([population_advantages(group) for group in groups])


def finalize_global_records(records):
    """Apply unmodified V3 q_reward and independent scalar G8 normalization."""
    if len(records) != COT_G:
        raise ValueError("V3-Official formal topology requires one global G4")
    gids = {row["recommendation_group_id"] for row in records}
    if len(gids) != 1:
        raise RuntimeError(f"V3_OFFICIAL_G4_GROUP_MIX: {sorted(gids)}")
    domains = {row["target_domain"] for row in records}
    if len(domains) != 1 or not domains.issubset(TARGET_DOMAINS):
        raise RuntimeError(f"V3_OFFICIAL_MIXED_G4_DOMAIN_MIX: {sorted(domains)}")
    for row in records:
        gold = set(row["gold_sids"])
        sid_rewards = [float(q_reward(sid, gold)) for sid in row["candidate_sids"]]
        advantages = population_advantages(sid_rewards).tolist()
        details = [
            {
                "candidate_sid": sid,
                "candidate_text": text,
                "q_reward": reward,
                "reward_level": reward_level(reward),
                "sid_advantage": float(advantage),
            }
            for sid, reward, advantage, text in zip(
                row["candidate_sids"], sid_rewards, advantages,
                row["candidate_texts"],
            )
        ]
        row.update({
            "q_rewards": sid_rewards,
            "sid_rewards": sid_rewards,
            "sid_advantages": advantages,
            "candidate_details": details,
            "cot_reward": float(sum(sid_rewards)),
            "exact": sum(value == 8.0 for value in sid_rewards),
            "ab": sum(value == 2.0 for value in sid_rewards),
            "a": sum(value == 0.5 for value in sid_rewards),
            "domain": sum(value == 0.0 for value in sid_rewards),
            "wrong_domain": sum(value == -0.25 for value in sid_rewards),
            "invalid": sum(value == -1.0 for value in sid_rewards),
        })
    cot_advantages = population_advantages([row["cot_reward"] for row in records]).tolist()
    for row, advantage in zip(records, cot_advantages):
        row["cot_advantage"] = float(advantage)
        row["zero_std_g8"] = not any(abs(value) > 0 for value in row["sid_advantages"])
    return records


def rollout_fingerprint(records):
    payload = [{
        "gid": row["recommendation_group_id"],
        "cot": row["cot_ids"],
        "official": row["candidate_ids"],
        "q_rewards": row["q_rewards"],
        "cot_reward": row["cot_reward"],
    } for row in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def official_rollout_summary(records):
    """Aggregate raw V3 reward signal without changing its mathematics."""
    details = [item for row in records for item in row["candidate_details"]]
    return {
        "domain": records[0]["target_domain"],
        "q_reward_mean": (
            sum(item["q_reward"] for item in details) / len(details)
            if details else 0.0
        ),
        "cot_length_mean": sum(row["cot_length"] for row in records) / len(records),
        "cot_rewards": [row["cot_reward"] for row in records],
        "cot_advantages": [row["cot_advantage"] for row in records],
        "zero_std_g8_count": sum(row["zero_std_g8"] for row in records),
        "A": sum(row["a"] for row in records),
        "AB": sum(row["ab"] for row in records),
        "Exact": sum(row["exact"] for row in records),
        "invalid": sum(row["invalid"] for row in records),
    }


def assert_iteration_reuse(expected_fingerprint, actual_fingerprint, policy_iteration,
                           sample_calls_before, sample_calls_after):
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("V3_OFFICIAL iteration2 rollout fingerprint changed")
    if policy_iteration not in (1, 2):
        raise RuntimeError("V3_OFFICIAL rollout may be optimized exactly twice")
    if policy_iteration == 2 and sample_calls_after != sample_calls_before:
        raise RuntimeError("V3_OFFICIAL iteration2 resampled Official candidates")
    return True


class OfficialSample8Runtime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_local = None
        self.last_global_records = None
        self.sample_calls = 0
        self.fresh_rollouts = 0
        self.closure_attempt = 0
        self.last_global_cot_closed = None

    def _close_token_id(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("V3_OFFICIAL_THINK_CLOSE_TOKEN_CONTRACT_FAILED")
        return int(ids[0])

    def cot_is_closed(self, ids):
        return self._close_token_id() in ids

    def _all_cots_closed(self, local_closed):
        flag = torch.tensor([int(local_closed)], dtype=torch.int32, device=self.model.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _sample_official(self, context_ids):
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        output = self.model.generate(
            inputs=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            num_beams=1,
            num_return_sequences=SID_G,
            min_new_tokens=OFFICIAL_SID_TOKENS,
            max_new_tokens=OFFICIAL_SID_TOKENS,
            pad_token_id=pad_id,
        )[:, input_ids.size(1):]
        rows = [[int(token) for token in row] for row in output.tolist()]
        official_action_mask(rows)
        return rows

    @staticmethod
    def _gather(record):
        if not dist.is_initialized():
            return [record]
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, record)
        return rows

    def score_local_cot(self, prompt, cot_ids, gold_sids, target_domain, group_id):
        if target_domain not in TARGET_DOMAINS:
            raise RuntimeError(f"V3-Official unsupported domain: {target_domain}")
        started = time.perf_counter()
        close_id = self._close_token_id()
        close_at = list(cot_ids).index(close_id)
        cot_trim = [int(value) for value in cot_ids[:close_at + 1]]
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        context, prefix_text, prefix_ids = build_fixed_domain_beam_input(
            self.tokenizer, prompt_ids, cot_trim, target_domain
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                candidate_ids = self._sample_official(context)
        finally:
            self.model.train(was_training)
        self.sample_calls += 1
        candidates = [
            parse_strict_abc3_ids(self.tokenizer, ids, target_domain)
            for ids in candidate_ids
        ]
        gold = {
            sid for raw in gold_sids
            if (sid := parse_sid(raw)) is not None
        }
        return {
            "recommendation_group_id": group_id,
            "target_domain": target_domain,
            "cot_ids": cot_trim,
            "cot_text": self.tokenizer.decode(cot_trim, skip_special_tokens=False),
            "cot_length": len(cot_trim),
            "official_context_ids": context,
            "official_domain_prefix": prefix_text,
            "official_domain_prefix_ids": prefix_ids,
            "candidate_ids": candidate_ids,
            "candidate_texts": [
                self.tokenizer.decode(ids, skip_special_tokens=False)
                for ids in candidate_ids
            ],
            "candidate_sids": candidates,
            "gold_sids": sorted(gold),
            "rollout_wall_sec": time.perf_counter() - started,
            "sampling_contract": {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "num_return_sequences": SID_G,
                "min_new_tokens": OFFICIAL_SID_TOKENS,
                "max_new_tokens": OFFICIAL_SID_TOKENS,
                "action_tokens": "ABC3 only; fixed domain begin is context",
            },
        }


def make_official_sample8_reward_func(runtime):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        if any(route != "think" for route in kwargs.get("route", [])):
            raise RuntimeError("V3-Official requires Think-only samples")
        if any(domain not in TARGET_DOMAINS for domain in kwargs.get("target_domain", [])):
            raise RuntimeError("V3-Official requires a supported target domain")
        closed = all(runtime.cot_is_closed(ids) for ids in completion_ids)
        runtime.last_global_cot_closed = runtime._all_cots_closed(closed)
        if not runtime.last_global_cot_closed:
            runtime.last_local = None
            return [0.0] * len(completion_ids)
        local = [
            runtime.score_local_cot(prompt, ids, golds, domain, gid)
            for prompt, ids, golds, domain, gid in zip(
                prompts, completion_ids, kwargs["all_gold_sids"],
                kwargs["target_domain"], kwargs["recommendation_group_id"],
            )
        ]
        if len(local) != 1:
            raise RuntimeError("V3-Official formal shape requires one local CoT per rank")
        records = finalize_global_records(runtime._gather(local[0]))
        rank = dist.get_rank() if dist.is_initialized() else 0
        runtime.last_global_records = records
        runtime.last_local = records[rank]
        return [runtime.last_local["cot_reward"]]

    reward_func.__name__ = "official_sample8_v3_cot_reward"
    reward_func.official_sample8_runtime = runtime
    return reward_func


class ThinkOfficialSample8Trainer(RecGRPOTrainer):
    def __init__(self, *args, lambda_cot=LAMBDA_COT, lambda_sid=LAMBDA_SID, **kwargs):
        funcs = kwargs.get("reward_funcs") or (args[3] if len(args) >= 4 else [])
        runtime = next((
            getattr(fn, "official_sample8_runtime", None)
            for fn in funcs
            if getattr(fn, "official_sample8_runtime", None) is not None
        ), None)
        if runtime is None:
            raise RuntimeError("V3-Official trainer requires OfficialSample8Runtime")
        if (float(lambda_cot), float(lambda_sid)) != (1.0, 1.0):
            raise ValueError("V3-Official loss weights are frozen at 1:1")
        self.runtime = runtime
        self._official_rollout = None
        self._official_policy_epoch = 0
        self._official_fingerprint = None
        self._official_sample_calls_at_rollout = None
        super().__init__(*args, **kwargs)
        self.add_callback(ResumeCadenceCallback())
        self.add_callback(FinalStepSaveCallback())

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return ThinkG4SingleGroupSampler(
            dataset, repeat_count=self.num_iterations,
            shuffle=self.shuffle_dataset, seed=self.args.seed,
        )

    def _route_generation_contract(self, route):
        if route != "think":
            raise RuntimeError("V3-Official is Think-only")
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, batch):
        rows = batch if isinstance(batch, list) else []
        if rows and any(
            row.get("route") != "think" or row.get("target_domain") not in TARGET_DOMAINS
            for row in rows
        ):
            raise RuntimeError("V3-Official dataset must be Think-only and all-domain")
        return super()._prepare_inputs(batch)

    def _pad_official(self, record):
        candidates = record["candidate_ids"]
        context = record["official_context_ids"]
        sequences = [list(context) + list(candidate) for candidate in candidates]
        max_len = max(map(len, sequences))
        rows = [[self.pad_token_id] * (max_len - len(row)) + row for row in sequences]
        attention = [[0] * (max_len - len(row)) + [1] * len(row) for row in sequences]
        action = official_action_mask(candidates)
        device = self.accelerator.device
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(attention, dtype=torch.long, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
        )

    def _build_official_rollout(self, cot_output):
        records = self.runtime.last_global_records
        record = self.runtime.last_local
        cot_advantages = population_advantages([row["cot_reward"] for row in records])
        rank = self.accelerator.process_index
        if not torch.allclose(
            cot_output["advantages"].float().cpu(),
            cot_advantages[rank:rank + 1],
            atol=1e-6,
        ):
            raise RuntimeError("V3_OFFICIAL_COT_ADVANTAGE_PARITY_FAILED")
        cot_old = cot_output.get("old_per_token_logps")
        if cot_old is None or cot_old.requires_grad:
            raise RuntimeError("V3_OFFICIAL_COT_OLD_LOGP_FAILED")
        input_ids, attention, action = self._pad_official(record)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention, action.size(1), compute_entropy=False
            )
        if old_logps.shape != action.shape or old_logps.requires_grad:
            raise RuntimeError("V3_OFFICIAL_OFFICIAL_OLD_LOGP_CONTRACT_FAILED")
        self._official_rollout = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "action_mask": action,
            "old_per_token_logps": old_logps.detach(),
            "advantages": population_advantages(record["sid_rewards"]).to(input_ids.device),
            "record": record,
            "global_records": records,
        }
        self._official_fingerprint = rollout_fingerprint(records)
        self._official_policy_epoch = 0
        self._official_sample_calls_at_rollout = self.runtime.sample_calls
        self.runtime.fresh_rollouts += 1
        if self.accelerator.is_main_process and self._monitor.enabled:
            summary = official_rollout_summary(records)
            self._monitor.write_official_sample8_v3({
                "step": int(self.state.global_step),
                "rollout_id": int(self._smoke_rollout_id),
                "rollout_fingerprint": self._official_fingerprint,
                "recommendation_group_id": records[0]["recommendation_group_id"],
                "target_domain": records[0]["target_domain"],
                "cots": records,
                "reward_summary": summary,
                "normalization_topology": "G4 + 4 independent Official G8; never G32",
            })

    def _generate_and_score_completions(self, inputs):
        before = self.runtime.sample_calls
        for attempt in range(MAX_COT_CLOSURE_RETRIES + 1):
            self.runtime.closure_attempt = attempt
            metric_lengths = {key: len(values) for key, values in self._metrics["train"].items()}
            output = super()._generate_and_score_completions(inputs)
            if self.runtime.last_global_cot_closed:
                self._build_official_rollout(output)
                if self.runtime.sample_calls - before != len(inputs):
                    raise RuntimeError("V3_OFFICIAL_EXPECTED_ONE_OFFICIAL_SAMPLE8_PER_COT")
                return output
            for key in list(self._metrics["train"]):
                del self._metrics["train"][key][metric_lengths.get(key, 0):]
            if self.runtime.sample_calls != before:
                raise RuntimeError("V3_OFFICIAL_SID_SAMPLED_BEFORE_COT_ACCEPTANCE")
        raise RuntimeError("V3_OFFICIAL_COT_NOT_CLOSED_AFTER_RETRIES")

    def _branch_loss(self, model, input_ids, attention, action, old_logps,
                     advantages, name):
        current, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention, action.size(1), compute_entropy=False
        )
        log_ratio = current - old_logps
        ratio = log_ratio.exp()
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token = -torch.min(
            ratio * advantages.unsqueeze(1),
            clipped * advantages.unsqueeze(1),
        )
        loss = (
            (per_token * action).sum(-1) / action.sum(-1).clamp(min=1)
        ).mean() / self.current_gradient_accumulation_steps
        selected = log_ratio[action.bool()].detach().float()
        return loss, {
            f"{name}_loss": float(loss.detach()),
            f"{name}_ratio_mean": float(selected.exp().mean()) if selected.numel() else 1.0,
            f"{name}_clip_fraction": (
                float(((selected.exp() - 1).abs() > self.epsilon_low).float().mean())
                if selected.numel() else 0.0
            ),
            f"{name}_approx_kl": (
                float((selected.exp() - 1 - selected).mean())
                if selected.numel() else 0.0
            ),
            f"{name}_action_tokens": int(action.sum()),
        }

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        if self._official_rollout is None or self._official_policy_epoch >= 2:
            raise RuntimeError("V3_OFFICIAL rollout unavailable or reused more than twice")
        prompt_completion = torch.cat(
            [inputs["prompt_ids"], inputs["completion_ids"]], dim=1
        )
        prompt_completion_mask = torch.cat(
            [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
        )
        cot_loss, cot_stats = self._branch_loss(
            model, prompt_completion, prompt_completion_mask,
            inputs["completion_mask"], inputs["old_per_token_logps"],
            inputs["advantages"], "cot",
        )
        sid = self._official_rollout
        sid_loss, sid_stats = self._branch_loss(
            model, sid["input_ids"], sid["attention_mask"], sid["action_mask"],
            sid["old_per_token_logps"], sid["advantages"], "sid",
        )
        total = combine_losses(cot_loss, sid_loss)
        self._official_policy_epoch += 1
        assert_iteration_reuse(
            self._official_fingerprint, rollout_fingerprint(sid["global_records"]),
            self._official_policy_epoch, self._official_sample_calls_at_rollout,
            self.runtime.sample_calls,
        )
        stats = {
            **cot_stats, **sid_stats,
            "total_loss": float(total.detach()),
            "policy_iteration": self._official_policy_epoch,
            "rollout_fingerprint": self._official_fingerprint,
            "official_sample_calls": self.runtime.sample_calls,
        }
        if self._smoke_log:
            self._smoke_log[-1].update(stats)
        return total

    def log(self, logs, start_time=None):
        result = super().log(logs, start_time)
        if (
            self.accelerator.is_main_process and self._monitor.enabled
            and self._smoke_log and "loss" in logs
        ):
            rollout = self._smoke_log[-1]
            record = self._official_rollout["record"] if self._official_rollout else {}
            grad_norm = logs.get("grad_norm")
            self._monitor.write_official_sample8_v3({
                "type": "optimization",
                "step": int(self.state.global_step),
                "rollout_id": rollout.get("rollout_id"),
                "policy_iteration": rollout.get("policy_iteration"),
                "rollout_fingerprint": rollout.get("rollout_fingerprint"),
                "target_domain": record.get("target_domain"),
                "cot_loss": rollout.get("cot_loss"),
                "sid_loss": rollout.get("sid_loss"),
                "total_loss": rollout.get("total_loss"),
                "cot_ratio_mean": rollout.get("cot_ratio_mean"),
                "sid_ratio_mean": rollout.get("sid_ratio_mean"),
                "cot_clip_fraction": rollout.get("cot_clip_fraction"),
                "sid_clip_fraction": rollout.get("sid_clip_fraction"),
                "cot_approx_kl": rollout.get("cot_approx_kl"),
                "sid_approx_kl": rollout.get("sid_approx_kl"),
                "cot_action_tokens": rollout.get("cot_action_tokens"),
                "sid_action_tokens": rollout.get("sid_action_tokens"),
                "total_lora_grad_norm": grad_norm,
                "logs_grad_norm": grad_norm,
                "zero_gradient_group": grad_norm is not None and float(grad_norm) == 0.0,
            })
        return result


def audit_official_sampler(dataset, sampler):
    del sampler
    rows = list(dataset)
    gids = [row.get("recommendation_group_id") for row in rows]
    if not rows or len(gids) != len(set(gids)):
        raise RuntimeError("V3-Official requires one row per unique business group")
    if any(
        row.get("route") != "think" or row.get("target_domain") not in TARGET_DOMAINS
        for row in rows
    ):
        raise RuntimeError("V3-Official dataset must be Think-only and all-domain")
    count = len(rows)
    return {
        "selected_groups": count,
        "trained_groups": count,
        "dropped_groups": 0,
        "think_unique_groups": count,
        "nothink_unique_groups": 0,
        "think_rollouts": count,
        "nothink_rollouts": 0,
        "fresh_rollout_count": count,
        "unique_groups_per_global_rollout": 1,
        "cot_candidates_per_group": COT_G,
        "official_candidates_per_group": COT_G * SID_G,
        "official_g8_groups": COT_G,
        "repeat_count": 2,
        "num_iterations": 2,
        "optimizer_steps": count * 2,
        "think_optimizer_steps": count * 2,
        "nothink_optimizer_steps": 0,
        "domain_group_counts": {
            domain: sum(row["target_domain"] == domain for row in rows)
            for domain in sorted(TARGET_DOMAINS)
        },
        "route_schedule_preview": ["think"] * min(24, count),
        "rollout_group_ids_preview": gids[:8],
    }
