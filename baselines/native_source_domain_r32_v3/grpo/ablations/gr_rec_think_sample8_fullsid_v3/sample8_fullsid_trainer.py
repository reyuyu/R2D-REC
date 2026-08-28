"""Formal-ready Think G4 + per-CoT Sample8 FullSID dual-objective trainer.

The CoT branch is ordinary group-relative PPO over one global G4. The SID
branch is on-policy stochastic optimization over four independent G8 groups.
Full SID candidates are sampled once and cached for both policy
iterations; they are never flattened into a G32 normalization group.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import TrainerCallback

from grpo_model import encode_prompt
from grpo_sid import parse_sid, q_reward
from grpo_trl_trainer import RecGRPOTrainer

COT_G = 4
SID_G = 8
LAMBDA_COT = 1.0
LAMBDA_SID = 1.0
FULL_SID_TOKENS = 4
SAMPLE_MAX_NEW_TOKENS = 128
_DOMAIN_PATTERN = re.compile(r"<\|(ad|video|prod|living)_begin\|>").fullmatch
_ABC_PATTERNS = (
    re.compile(r"<s_a_(\d+)>").fullmatch,
    re.compile(r"<s_b_(\d+)>").fullmatch,
    re.compile(r"<s_c_(\d+)>").fullmatch,
)


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
        raise RuntimeError(f"SAMPLE8_FULLSID_G4_GROUP_MIX: {group_ids}")


def reward_level(reward):
    return {-1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
            0.5: "A", 2.0: "AB", 8.0: "EXACT"}.get(float(reward), "unknown")


def scan_full_sid_ids(tokenizer, generated_ids):
    """Scan a continuation for complete raw-ID SIDs and select the first one."""
    ids = list(generated_ids or [])
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    if isinstance(tokens, str):
        tokens = [tokens]
    occurrences = []
    for start in range(max(0, len(tokens) - FULL_SID_TOKENS + 1)):
        domain = _DOMAIN_PATTERN(tokens[start])
        abc = [
            pattern(token)
            for pattern, token in zip(_ABC_PATTERNS, tokens[start + 1:start + 4])
        ]
        if domain is None or any(match is None for match in abc):
            continue
        sid = (domain.group(1), *(int(match.group(1)) for match in abc))
        occurrences.append((sid, (start, start + FULL_SID_TOKENS)))
    parsed = occurrences[0][0] if occurrences else None
    return {
        "parsed_sid": parsed,
        "all_parsed_sids": [item[0] for item in occurrences],
        "sid_spans": [item[1] for item in occurrences],
        "first_sid_span": occurrences[0][1] if occurrences else None,
        "sid_count": len(occurrences),
        "multi_sid_output": len(occurrences) > 1,
        "parser_status": (
            "multiple_sids" if len(occurrences) > 1
            else "ok" if occurrences
            else "missing_sid"
        ),
    }


def parse_full_sid_ids(tokenizer, generated_ids):
    """Return the first complete raw-ID SID found anywhere in a continuation."""
    return scan_full_sid_ids(tokenizer, generated_ids)["parsed_sid"]


def cot_reward_from_sid_rewards(sid_rewards):
    if len(sid_rewards) != SID_G:
        raise ValueError("CoT reward requires exactly eight sampled SID rewards")
    return float(sum(float(value) for value in sid_rewards))


def rollout_fingerprint(records):
    payload = [{"gid": row["recommendation_group_id"], "cot": row["cot_ids"],
                "sample": row["sample_candidate_ids"], "cot_reward": row["cot_reward"],
                "sid_rewards": row["sid_rewards"]} for row in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ThinkG4SingleGroupSampler(Sampler):
    """Yield one row as a rank-aligned global G4, repeated for two PPO passes."""

    def __init__(self, data_source, repeat_count=2, shuffle=False, seed=None):
        del seed
        if shuffle:
            raise ValueError("Sample8 FullSID sampler requires shuffle=False")
        if int(repeat_count) != 2:
            raise ValueError("Sample8 FullSID num_iterations=2 is a hard contract")
        self.data_source = data_source
        self.repeat_count = 2

    def __iter__(self):
        for index in range(len(self.data_source)):
            for _ in range(self.repeat_count):
                yield from [index] * COT_G

    def __len__(self):
        return len(self.data_source) * self.repeat_count * COT_G


class Sample8FullSIDRuntime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_local = None
        self.sample_calls = 0
        self.fresh_rollouts = 0

    def _sample_full_sids(self, context_ids):
        device = self.model.device
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            num_beams=1,
            num_return_sequences=SID_G,
            min_new_tokens=FULL_SID_TOKENS,
            max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
            pad_token_id=pad_id,
        )
        generated = outputs[:, input_ids.size(1):]
        rows = []
        for row in generated.tolist():
            ids = [int(token) for token in row]
            if self.tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(self.tokenizer.eos_token_id)]
            rows.append(ids)
        return rows

    def score_local_cot(self, prompt, cot_ids, gold_sids, target_domain, group_id=None):
        started = time.perf_counter()
        gold_set = {sid for raw in gold_sids for sid in [parse_sid(raw)] if sid is not None}
        close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(close_ids) != 1 or int(close_ids[0]) not in cot_ids:
            raise RuntimeError("SAMPLE8_FULLSID_COT_NOT_CLOSED")
        close_at = list(cot_ids).index(int(close_ids[0]))
        cot_trim_ids = [int(token) for token in cot_ids[: close_at + 1]]
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        sample_context_ids = list(prompt_ids) + cot_trim_ids
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                generated_ids = self._sample_full_sids(sample_context_ids)
        finally:
            self.model.train(was_training)
        self.sample_calls += 1
        if len(generated_ids) != SID_G or any(len(ids) < FULL_SID_TOKENS for ids in generated_ids):
            raise RuntimeError("SAMPLE8_FULLSID_NOT_EXACT_SAMPLE8")
        scans = [scan_full_sid_ids(self.tokenizer, ids) for ids in generated_ids]
        sample_sids = [scan["parsed_sid"] for scan in scans]
        sid_rewards = [float(q_reward(sid, gold_set)) for sid in sample_sids]
        cot_reward = cot_reward_from_sid_rewards(sid_rewards)
        record = {
            "recommendation_group_id": group_id, "target_domain": target_domain,
            "cot_ids": cot_trim_ids,
            "cot_text": self.tokenizer.decode(cot_trim_ids, skip_special_tokens=False),
            "closed": True, "cot_length": len(cot_trim_ids),
            "sample_context_ids": sample_context_ids,
            "fixed_domain_prefix": False,
            "natural_language_bridge": False,
            "sample_candidate_ids": generated_ids,
            "sample_candidate_texts": [
                self.tokenizer.decode(ids, skip_special_tokens=False)
                for ids in generated_ids
            ],
            "sample_sids": sample_sids, "sid_rewards": sid_rewards,
            "all_parsed_sids": [scan["all_parsed_sids"] for scan in scans],
            "sid_action_spans": [scan["first_sid_span"] for scan in scans],
            "sid_counts": [scan["sid_count"] for scan in scans],
            "multi_sid_outputs": [scan["multi_sid_output"] for scan in scans],
            "parser_statuses": [scan["parser_status"] for scan in scans],
            "sid_reward_levels": [reward_level(value) for value in sid_rewards],
            "sid_advantages": population_advantages(sid_rewards).tolist(),
            "sid_population_std": float(torch.tensor(sid_rewards).std(correction=0)),
            "sid_zero_std": len(set(sid_rewards)) == 1,
            "cot_reward": cot_reward,
            "exact": int(sum(value == 8.0 for value in sid_rewards)),
            "ab": int(sum(value == 2.0 for value in sid_rewards)),
            "a": int(sum(value == 0.5 for value in sid_rewards)),
            "domain": int(sum(value == 0.0 for value in sid_rewards)),
            "wrong_domain": int(sum(value == -0.25 for value in sid_rewards)),
            "invalid": int(sum(value == -1.0 for value in sid_rewards)),
            "sample8_wall_sec": time.perf_counter() - started,
            "sampling_contract": {
                "do_sample": True, "temperature": 1.0, "top_p": 1.0,
                "top_k": 0, "repetition_penalty": 1.0,
                "max_new_tokens": SAMPLE_MAX_NEW_TOKENS,
                "sid_selection": "first complete raw-ID SID in continuation",
            },
        }
        self.last_local = record
        return record


def make_sample8_fullsid_reward_func(runtime):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        routes = kwargs.get("route")
        if routes is None or any(route != "think" for route in routes):
            raise RuntimeError("Sample8 FullSID reward requires Think-only samples")
        records = [runtime.score_local_cot(prompt, ids, golds, domain, gid)
                   for prompt, ids, golds, domain, gid in zip(
                       prompts, completion_ids, kwargs["all_gold_sids"],
                       kwargs["target_domain"], kwargs["recommendation_group_id"])]
        runtime.last_local = records[0] if len(records) == 1 else records
        return [record["cot_reward"] for record in records]
    reward_func.__name__ = "sample8_fullsid_cot_reward"
    reward_func.sample8_fullsid_runtime = runtime
    return reward_func


class ThinkSample8FullSIDTrainer(RecGRPOTrainer):
    def __init__(self, *args, lambda_cot=LAMBDA_COT, lambda_sid=LAMBDA_SID, **kwargs):
        reward_funcs = kwargs.get("reward_funcs") or (args[3] if len(args) >= 4 else [])
        runtime = next((getattr(fn, "sample8_fullsid_runtime", None) for fn in reward_funcs
                        if getattr(fn, "sample8_fullsid_runtime", None) is not None), None)
        if runtime is None:
            raise RuntimeError("ThinkSample8FullSIDTrainer requires Sample8 FullSID runtime")
        if float(lambda_cot) != 1.0 or float(lambda_sid) != 1.0:
            raise ValueError("first formal Sample8 FullSID experiment freezes lambdas at 1.0")
        self._sample8_fullsid_runtime = runtime
        self.lambda_cot, self.lambda_sid = 1.0, 1.0
        self._sid_rollout = None
        self._sample_policy_epoch = 0
        self._sample_rollout_fingerprint = None
        super().__init__(*args, **kwargs)
        self.add_callback(ResumeCadenceCallback())
        self.add_callback(FinalStepSaveCallback())

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return ThinkG4SingleGroupSampler(dataset, repeat_count=self.num_iterations,
                                         shuffle=self.shuffle_dataset, seed=self.args.seed)

    def _route_generation_contract(self, route):
        if route != "think":
            raise RuntimeError("Sample8 FullSID is Think-only")
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, batch):
        rows = batch if isinstance(batch, list) else []
        if rows and any(row.get("route") != "think" for row in rows):
            raise RuntimeError("Sample8 FullSID training dataset must be Think-only")
        return super()._prepare_inputs(batch)

    def _pad_sid_sequences(self, record):
        candidates = record["sample_candidate_ids"]
        spans = record["sid_action_spans"]
        if len(candidates) != SID_G or len(spans) != SID_G:
            raise RuntimeError("Sample8 SID objective requires exactly eight continuations")
        continuation_width = max(map(len, candidates))
        sequences = [
            list(record["sample_context_ids"]) + list(candidate)
            for candidate in candidates
        ]
        max_len = max(map(len, sequences))
        rows = [[self.pad_token_id] * (max_len - len(row)) + row for row in sequences]
        masks = [[0] * (max_len - len(row)) + [1] * len(row) for row in sequences]
        action_masks = []
        for candidate, span in zip(candidates, spans):
            row = [0] * continuation_width
            if span is not None:
                start, end = span
                offset = continuation_width - len(candidate)
                row[offset + start:offset + end] = [1] * FULL_SID_TOKENS
            action_masks.append(row)
        device = self.accelerator.device
        return (torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(masks, dtype=torch.long, device=device),
                torch.tensor(action_masks, dtype=torch.long, device=device))

    @staticmethod
    def _gather_records(record):
        if not dist.is_initialized():
            return [record]
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        return records

    def _build_sid_rollout(self, cot_output):
        record = self._sample8_fullsid_runtime.last_local
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
            raise RuntimeError("SAMPLE8_FULLSID_COT_ADVANTAGE_PARITY_FAILED")
        if cot_output.get("old_per_token_logps") is None:
            raise RuntimeError("SAMPLE8_FULLSID_COT_OLD_LOGP_MUST_BE_FULL_FORWARD")
        if cot_output["old_per_token_logps"].requires_grad:
            raise RuntimeError("SAMPLE8_FULLSID_COT_OLD_LOGP_NOT_DETACHED")
        input_ids, attention_mask, action_mask = self._pad_sid_sequences(record)
        continuation_width = action_mask.size(1)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention_mask, continuation_width,
                compute_entropy=False)
        if old_logps.shape != action_mask.shape or old_logps.requires_grad:
            raise RuntimeError("SAMPLE8_FULLSID_SID_OLD_LOGP_CONTRACT_FAILED")
        active_counts = action_mask.sum(1)
        if any(int(value) not in (0, FULL_SID_TOKENS) for value in active_counts):
            raise RuntimeError("SAMPLE8_FULLSID_ACTION_MASK_NOT_FIRST_COMPLETE_SID4")
        self._sid_rollout = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "old_per_token_logps": old_logps.detach(),
            "action_mask": action_mask,
            "advantages": population_advantages(record["sid_rewards"]).to(input_ids.device),
            "record": record, "global_records": records,
        }
        self._sample_rollout_fingerprint = rollout_fingerprint(records)
        self._sample_policy_epoch = 0
        self._sample8_fullsid_runtime.fresh_rollouts += 1
        if self.accelerator.is_main_process and self._monitor.enabled:
            self._monitor.write_sample8_fullsid({
                "step": int(self.state.global_step), "rollout_id": int(self._smoke_rollout_id),
                "rollout_fingerprint": self._sample_rollout_fingerprint,
                "recommendation_group_id": records[0]["recommendation_group_id"],
                "target_domain": records[0]["target_domain"], "cots": records,
                "cot_rewards": cot_rewards, "cot_advantages": cot_advantages.tolist(),
                "cot_population_std": float(torch.tensor(cot_rewards).std(correction=0)),
                "cot_zero_std": len(set(cot_rewards)) == 1,
                "normalization_topology": "G4 + 4 independent G8; never G32",
            })

    def _generate_and_score_completions(self, inputs):
        before_calls = self._sample8_fullsid_runtime.sample_calls
        output = super()._generate_and_score_completions(inputs)
        self._build_sid_rollout(output)
        if self._sample8_fullsid_runtime.sample_calls - before_calls != len(inputs):
            raise RuntimeError("SAMPLE8_FULLSID_EXPECTED_ONE_SAMPLE8_CALL_PER_LOCAL_COT")
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
            ratio_mean = float(ratio_selected.mean()) if selected.numel() else 1.0
            clip_fraction = (float(((ratio_selected - 1).abs() > self.epsilon_low).float().mean())
                             if selected.numel() else 0.0)
            approx_kl = (float((ratio_selected - 1 - selected).mean())
                         if selected.numel() else 0.0)
            stats = {
                f"{branch}_loss": float(loss.detach()),
                f"{branch}_ratio_mean": ratio_mean,
                f"{branch}_clip_fraction": clip_fraction,
                f"{branch}_approx_kl": approx_kl,
                f"{branch}_action_tokens": int(action_mask.sum()),
                f"{branch}_action_logp_grad_norm": float((advantages.unsqueeze(1).expand_as(action_mask) * action_mask).float().norm()),
            }
        return loss, stats

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        if self._sid_rollout is None:
            raise RuntimeError("SID rollout unavailable")
        if self._sample_policy_epoch >= 2:
            raise RuntimeError("SAMPLE8_FULLSID_ROLLOUT_REUSED_MORE_THAN_TWO_ITERATIONS")
        prompt_completion = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        prompt_completion_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        cot_loss, cot_stats = self._branch_loss(
            model, prompt_completion, prompt_completion_mask, inputs["completion_mask"],
            inputs["old_per_token_logps"], inputs["advantages"], "cot")
        sid = self._sid_rollout
        sid_loss, sid_stats = self._branch_loss(
            model, sid["input_ids"], sid["attention_mask"], sid["action_mask"],
            sid["old_per_token_logps"], sid["advantages"], "sid")
        total = self.lambda_cot * cot_loss + self.lambda_sid * sid_loss
        self._sample_policy_epoch += 1
        stats = {**cot_stats, **sid_stats, "total_loss": float(total.detach()),
                 "policy_iteration": self._sample_policy_epoch,
                 "rollout_fingerprint": self._sample_rollout_fingerprint,
                 "sample_calls": self._sample8_fullsid_runtime.sample_calls}
        if self._smoke_log:
            self._smoke_log[-1].update(stats)
        return total

    def log(self, logs, start_time=None):
        result = super().log(logs, start_time)
        if (self.accelerator.is_main_process and self._monitor.enabled and
                self._smoke_log and "loss" in logs):
            rollout = self._smoke_log[-1]
            self._monitor.write_sample8_fullsid({
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


class ResumeCadenceCallback(TrainerCallback):
    """Recompute interval state from current args after TrainerState restore."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.compute_steps(args, int(state.max_steps))
        return control


class FinalStepSaveCallback(TrainerCallback):
    """Force a full Trainer checkpoint at max_steps (3090 in the formal run)."""

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) == int(args.max_steps):
            control.should_save = True
        return control


def audit_sample8_sampler(dataset, sampler):
    rows = list(dataset)
    if not rows or any(row.get("route") != "think" for row in rows):
        raise RuntimeError("Sample8 FullSID requires Think-only rows")
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
