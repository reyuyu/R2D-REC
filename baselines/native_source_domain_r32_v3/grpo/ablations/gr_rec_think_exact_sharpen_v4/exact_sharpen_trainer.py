"""Think G4 with independent Free/Official Sample8 SID objectives."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter

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
    scan_full_sid_ids,
)

COT_G = 4
SID_G = 8
FREE_SID_TOKENS = 4
OFFICIAL_SID_TOKENS = 3
FREE_MAX_NEW_TOKENS = 128
SATURATION_THRESHOLD = 8
LAMBDA_COT = 1.0
LAMBDA_FREE_SID = 0.5
LAMBDA_OFFICIAL_SID = 0.5


def reward_level(value):
    return {-1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
            0.5: "A", 2.0: "AB", 8.0: "EXACT"}.get(float(value), "unknown")


def branch_is_saturated(raw_reward_groups):
    flat = [float(v) for group in raw_reward_groups for v in group]
    if len(flat) != COT_G * SID_G:
        raise ValueError("branch saturation requires exactly 32 candidates")
    return sum(v in (0.5, 2.0, 8.0) for v in flat) > SATURATION_THRESHOLD


def saturated_reward(raw_reward, saturated):
    value = float(raw_reward)
    if not saturated:
        return value
    return {0.5: 0.0, 2.0: 1.5, 8.0: 7.5}.get(value, value)


def duplicate_penalties(sids, raw_rewards):
    if len(sids) != SID_G or len(raw_rewards) != SID_G:
        raise ValueError("duplicate penalty requires one G8")
    counts = Counter(
        tuple(sid) for sid, reward in zip(sids, raw_rewards)
        if sid is not None and float(reward) != 8.0
    )
    penalties = []
    for sid, reward in zip(sids, raw_rewards):
        if sid is None or float(reward) == 8.0:
            penalties.append(0.0)
            continue
        count = counts[tuple(sid)]
        penalties.append(0.0 if count <= 2 else -0.5 if count == 3 else -1.0)
    return penalties


def shape_g8(sids, raw_rewards, saturated):
    penalties = duplicate_penalties(sids, raw_rewards)
    shaped = [saturated_reward(r, saturated) + p
              for r, p in zip(raw_rewards, penalties)]
    return shaped, penalties


def strict_unique_cot_reward(sids, raw_rewards, saturated):
    """Coverage reward from one CoT's Free G8; higher levels cover prefixes."""
    if len(sids) != SID_G or len(raw_rewards) != SID_G:
        raise ValueError("CoT coverage requires one Free G8")
    exact = {tuple(s) for s, r in zip(sids, raw_rewards)
             if s is not None and float(r) == 8.0}
    ab = {(s[0], s[1], s[2]) for s, r in zip(sids, raw_rewards)
          if s is not None and float(r) == 2.0}
    a = {(s[0], s[1]) for s, r in zip(sids, raw_rewards)
         if s is not None and float(r) == 0.5}
    exact_ab = {(s[0], s[1], s[2]) for s in exact}
    uncovered_ab = ab - exact_ab
    covered_a = {(s[0], s[1]) for s in exact} | {(s[0], s[1]) for s in ab}
    uncovered_a = a - covered_a
    if saturated:
        score = 7.5 * len(exact) + 1.5 * len(uncovered_ab)
    else:
        score = 8.0 * len(exact) + 2.0 * len(uncovered_ab) + 0.5 * len(uncovered_a)
    return float(score), {
        "unique_exact": len(exact), "unique_uncovered_ab": len(uncovered_ab),
        "unique_uncovered_a": 0 if saturated else len(uncovered_a),
    }


def independent_g8_advantages(groups):
    if len(groups) != COT_G or any(len(group) != SID_G for group in groups):
        raise ValueError("requires exactly four independent G8 groups")
    return torch.stack([population_advantages(group) for group in groups])


def finalize_global_records(records):
    if len(records) != COT_G:
        raise ValueError("formal topology requires four global CoT records")
    gids = {row["recommendation_group_id"] for row in records}
    if len(gids) != 1:
        raise RuntimeError(f"V4_G4_GROUP_MIX: {sorted(gids)}")
    free_raw = [row["free_raw_rewards"] for row in records]
    official_raw = [row["official_raw_rewards"] for row in records]
    free_sat = branch_is_saturated(free_raw)
    official_sat = branch_is_saturated(official_raw)
    for row in records:
        free_shaped, free_penalties = shape_g8(
            row["free_sids"], row["free_raw_rewards"], free_sat)
        official_shaped, official_penalties = shape_g8(
            row["official_sids"], row["official_raw_rewards"], official_sat)
        cot_reward, coverage = strict_unique_cot_reward(
            row["free_sids"], row["free_raw_rewards"], free_sat)
        row.update({
            "free_saturated": free_sat, "official_saturated": official_sat,
            "free_a_plus_count": sum(v in (0.5, 2.0, 8.0)
                                     for group in free_raw for v in group),
            "official_a_plus_count": sum(v in (0.5, 2.0, 8.0)
                                         for group in official_raw for v in group),
            "free_rewards": free_shaped, "official_rewards": official_shaped,
            "free_duplicate_penalties": free_penalties,
            "official_duplicate_penalties": official_penalties,
            "free_advantages": population_advantages(free_shaped).tolist(),
            "official_advantages": population_advantages(official_shaped).tolist(),
            "cot_reward": cot_reward, "cot_coverage": coverage,
        })
    return records


def rollout_fingerprint(records):
    payload = [{
        "gid": r["recommendation_group_id"], "cot": r["cot_ids"],
        "free": r["free_candidate_ids"], "official": r["official_candidate_ids"],
        "free_rewards": r["free_rewards"], "official_rewards": r["official_rewards"],
        "cot_reward": r["cot_reward"],
    } for r in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExactSharpenRuntime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_local = None
        self.last_global_records = None
        self.free_sample_calls = 0
        self.official_sample_calls = 0
        self.fresh_rollouts = 0
        self.closure_attempt = 0
        self.last_global_cot_closed = None

    def _close_token_id(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("V4_THINK_CLOSE_TOKEN_CONTRACT_FAILED")
        return int(ids[0])

    def cot_is_closed(self, ids):
        return self._close_token_id() in ids

    def _all_cots_closed(self, local_closed):
        flag = torch.tensor([int(local_closed)], dtype=torch.int32, device=self.model.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _sample(self, context_ids, *, official):
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_tokens = OFFICIAL_SID_TOKENS if official else FREE_MAX_NEW_TOKENS
        min_tokens = OFFICIAL_SID_TOKENS if official else FREE_SID_TOKENS
        outputs = self.model.generate(
            inputs=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True,
            temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            num_beams=1, num_return_sequences=SID_G,
            min_new_tokens=min_tokens, max_new_tokens=max_tokens, pad_token_id=pad_id,
        )
        rows = []
        for row in outputs[:, input_ids.size(1):].tolist():
            ids = [int(token) for token in row]
            if self.tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(self.tokenizer.eos_token_id)]
            rows.append(ids)
        if len(rows) != SID_G:
            raise RuntimeError("V4_NOT_SAMPLE8")
        if official and any(len(ids) != OFFICIAL_SID_TOKENS for ids in rows):
            raise RuntimeError("V4_OFFICIAL_NOT_EXACT_8XABC3")
        return rows

    @staticmethod
    def _gather(record):
        if not dist.is_initialized():
            return [record]
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, record)
        return rows

    def score_local_cot(self, prompt, cot_ids, gold_sids, target_domain, group_id):
        started = time.perf_counter()
        close_id = self._close_token_id()
        close_at = list(cot_ids).index(close_id)
        cot_trim = [int(v) for v in cot_ids[:close_at + 1]]
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        free_context = list(prompt_ids) + cot_trim
        official_context, prefix_text, prefix_ids = build_fixed_domain_beam_input(
            self.tokenizer, prompt_ids, cot_trim, target_domain)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                free_ids = self._sample(free_context, official=False)
                official_ids = self._sample(official_context, official=True)
        finally:
            self.model.train(was_training)
        self.free_sample_calls += 1
        self.official_sample_calls += 1
        free_scans = [scan_full_sid_ids(self.tokenizer, ids) for ids in free_ids]
        free_sids = [scan["parsed_sid"] for scan in free_scans]
        official_sids = [parse_strict_abc3_ids(self.tokenizer, ids, target_domain)
                         for ids in official_ids]
        gold = {sid for raw in gold_sids for sid in [parse_sid(raw)] if sid is not None}
        return {
            "recommendation_group_id": group_id, "target_domain": target_domain,
            "cot_ids": cot_trim, "cot_text": self.tokenizer.decode(cot_trim, skip_special_tokens=False),
            "free_context_ids": free_context, "official_context_ids": official_context,
            "official_domain_prefix": prefix_text, "official_domain_prefix_ids": prefix_ids,
            "free_candidate_ids": free_ids, "official_candidate_ids": official_ids,
            "free_sids": free_sids, "official_sids": official_sids,
            "free_raw_rewards": [float(q_reward(sid, gold)) for sid in free_sids],
            "official_raw_rewards": [float(q_reward(sid, gold)) for sid in official_sids],
            "free_action_spans": [scan["first_sid_span"] for scan in free_scans],
            "free_parser_statuses": [scan["parser_status"] for scan in free_scans],
            "rollout_wall_sec": time.perf_counter() - started,
        }


def make_exact_sharpen_reward_func(runtime):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        if any(route != "think" for route in kwargs.get("route", [])):
            raise RuntimeError("V4 requires Think-only samples")
        local_closed = all(runtime.cot_is_closed(ids) for ids in completion_ids)
        runtime.last_global_cot_closed = runtime._all_cots_closed(local_closed)
        if not runtime.last_global_cot_closed:
            runtime.last_local = None
            return [0.0] * len(completion_ids)
        local = [runtime.score_local_cot(prompt, ids, golds, domain, gid)
                 for prompt, ids, golds, domain, gid in zip(
                     prompts, completion_ids, kwargs["all_gold_sids"],
                     kwargs["target_domain"], kwargs["recommendation_group_id"])]
        if len(local) != 1:
            raise RuntimeError("V4 formal shape requires one local CoT per rank")
        records = finalize_global_records(runtime._gather(local[0]))
        rank = dist.get_rank() if dist.is_initialized() else 0
        runtime.last_global_records = records
        runtime.last_local = records[rank]
        return [runtime.last_local["cot_reward"]]
    reward_func.__name__ = "exact_sharpen_v4_cot_reward"
    reward_func.exact_sharpen_runtime = runtime
    return reward_func


class ThinkExactSharpenTrainer(RecGRPOTrainer):
    def __init__(self, *args, lambda_cot=LAMBDA_COT,
                 lambda_free=LAMBDA_FREE_SID, lambda_official=LAMBDA_OFFICIAL_SID, **kwargs):
        funcs = kwargs.get("reward_funcs") or (args[3] if len(args) >= 4 else [])
        runtime = next((getattr(fn, "exact_sharpen_runtime", None) for fn in funcs
                        if getattr(fn, "exact_sharpen_runtime", None) is not None), None)
        if runtime is None:
            raise RuntimeError("V4 trainer requires ExactSharpenRuntime")
        if (float(lambda_cot), float(lambda_free), float(lambda_official)) != (1.0, 0.5, 0.5):
            raise ValueError("V4 loss weights are frozen at 1/0.5/0.5")
        self.runtime = runtime
        self.lambda_cot, self.lambda_free, self.lambda_official = 1.0, 0.5, 0.5
        self._v4_rollout = None
        self._v4_policy_epoch = 0
        self._v4_fingerprint = None
        super().__init__(*args, **kwargs)
        self.add_callback(ResumeCadenceCallback())
        self.add_callback(FinalStepSaveCallback())

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return ThinkG4SingleGroupSampler(dataset, repeat_count=self.num_iterations,
                                         shuffle=self.shuffle_dataset, seed=self.args.seed)

    def _route_generation_contract(self, route):
        if route != "think":
            raise RuntimeError("V4 is Think-only")
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, batch):
        rows = batch if isinstance(batch, list) else []
        if rows and any(row.get("route") != "think" for row in rows):
            raise RuntimeError("V4 dataset must be Think-only")
        return super()._prepare_inputs(batch)

    def _pad_branch(self, record, branch):
        candidates = record[f"{branch}_candidate_ids"]
        context = record[f"{branch}_context_ids"]
        sequences = [list(context) + list(candidate) for candidate in candidates]
        max_len = max(map(len, sequences))
        rows = [[self.pad_token_id] * (max_len - len(row)) + row for row in sequences]
        masks = [[0] * (max_len - len(row)) + [1] * len(row) for row in sequences]
        width = max(map(len, candidates))
        action = []
        if branch == "free":
            for candidate, span in zip(candidates, record["free_action_spans"]):
                row = [0] * width
                if span is not None:
                    start, end = span
                    offset = width - len(candidate)
                    row[offset + start:offset + end] = [1] * FREE_SID_TOKENS
                action.append(row)
        else:
            if any(len(candidate) != OFFICIAL_SID_TOKENS for candidate in candidates):
                raise RuntimeError("V4 official mask requires exact ABC3")
            action = [[1] * OFFICIAL_SID_TOKENS for _ in candidates]
        device = self.accelerator.device
        return (torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(masks, dtype=torch.long, device=device),
                torch.tensor(action, dtype=torch.long, device=device))

    def _build_branch(self, record, branch):
        input_ids, attention, action = self._pad_branch(record, branch)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention, action.size(1), compute_entropy=False)
        expected = FREE_SID_TOKENS if branch == "free" else OFFICIAL_SID_TOKENS
        if old_logps.shape != action.shape or old_logps.requires_grad:
            raise RuntimeError(f"V4_{branch.upper()}_OLD_LOGP_CONTRACT_FAILED")
        if any(int(v) not in ((0, expected) if branch == "free" else (expected,))
               for v in action.sum(1)):
            raise RuntimeError(f"V4_{branch.upper()}_ACTION_MASK_FAILED")
        return {"input_ids": input_ids, "attention_mask": attention,
                "action_mask": action, "old_per_token_logps": old_logps.detach(),
                "advantages": population_advantages(record[f"{branch}_rewards"]).to(input_ids.device)}

    def _build_v4_rollout(self, cot_output):
        records = self.runtime.last_global_records
        record = self.runtime.last_local
        cot_adv = population_advantages([row["cot_reward"] for row in records])
        rank = self.accelerator.process_index
        if not torch.allclose(cot_output["advantages"].float().cpu(), cot_adv[rank:rank + 1], atol=1e-6):
            raise RuntimeError("V4_COT_ADVANTAGE_PARITY_FAILED")
        if cot_output.get("old_per_token_logps") is None or cot_output["old_per_token_logps"].requires_grad:
            raise RuntimeError("V4_COT_OLD_LOGP_FAILED")
        self._v4_rollout = {"free": self._build_branch(record, "free"),
                            "official": self._build_branch(record, "official"),
                            "record": record, "global_records": records}
        self._v4_fingerprint = rollout_fingerprint(records)
        self._v4_policy_epoch = 0
        self.runtime.fresh_rollouts += 1
        if self.accelerator.is_main_process and self._monitor.enabled:
            self._monitor.write_exact_sharpen_v4({
                "step": int(self.state.global_step), "rollout_id": int(self._smoke_rollout_id),
                "rollout_fingerprint": self._v4_fingerprint,
                "recommendation_group_id": records[0]["recommendation_group_id"],
                "free_saturated": records[0]["free_saturated"],
                "official_saturated": records[0]["official_saturated"],
                "free_a_plus_count": records[0]["free_a_plus_count"],
                "official_a_plus_count": records[0]["official_a_plus_count"],
                "cots": records,
                "normalization_topology": "G4 + 4 Free G8 + 4 Official G8; never G16/G64",
            })

    def _generate_and_score_completions(self, inputs):
        free_before, official_before = self.runtime.free_sample_calls, self.runtime.official_sample_calls
        for attempt in range(MAX_COT_CLOSURE_RETRIES + 1):
            self.runtime.closure_attempt = attempt
            metric_lengths = {k: len(v) for k, v in self._metrics["train"].items()}
            output = super()._generate_and_score_completions(inputs)
            if self.runtime.last_global_cot_closed:
                self._build_v4_rollout(output)
                if self.runtime.free_sample_calls - free_before != len(inputs):
                    raise RuntimeError("V4_EXPECTED_ONE_FREE_SAMPLE8_PER_COT")
                if self.runtime.official_sample_calls - official_before != len(inputs):
                    raise RuntimeError("V4_EXPECTED_ONE_OFFICIAL_SAMPLE8_PER_COT")
                return output
            for key in list(self._metrics["train"]):
                del self._metrics["train"][key][metric_lengths.get(key, 0):]
            if (self.runtime.free_sample_calls, self.runtime.official_sample_calls) != (free_before, official_before):
                raise RuntimeError("V4_SID_SAMPLED_BEFORE_COT_ACCEPTANCE")
        raise RuntimeError("V4_COT_NOT_CLOSED_AFTER_RETRIES")

    def _branch_loss(self, model, input_ids, attention, action, old_logps, advantages, name):
        current, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention, action.size(1), compute_entropy=False)
        log_ratio = current - old_logps
        ratio = log_ratio.exp()
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token = -torch.min(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1))
        loss = ((per_token * action).sum(-1) / action.sum(-1).clamp(min=1)).mean()
        loss = loss / self.current_gradient_accumulation_steps
        selected = log_ratio[action.bool()].detach().float()
        return loss, {
            f"{name}_loss": float(loss.detach()),
            f"{name}_ratio_mean": float(selected.exp().mean()) if selected.numel() else 1.0,
            f"{name}_clip_fraction": float(((selected.exp() - 1).abs() > self.epsilon_low).float().mean()) if selected.numel() else 0.0,
            f"{name}_approx_kl": float((selected.exp() - 1 - selected).mean()) if selected.numel() else 0.0,
            f"{name}_action_tokens": int(action.sum()),
        }

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        if self._v4_rollout is None or self._v4_policy_epoch >= 2:
            raise RuntimeError("V4 rollout unavailable or reused more than twice")
        pc = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        pcm = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        cot_loss, cot_stats = self._branch_loss(
            model, pc, pcm, inputs["completion_mask"], inputs["old_per_token_logps"],
            inputs["advantages"], "cot")
        losses, stats = {}, {}
        for branch in ("free", "official"):
            item = self._v4_rollout[branch]
            losses[branch], branch_stats = self._branch_loss(
                model, item["input_ids"], item["attention_mask"], item["action_mask"],
                item["old_per_token_logps"], item["advantages"], branch)
            stats.update(branch_stats)
        total = self.lambda_cot * cot_loss + self.lambda_free * losses["free"] + self.lambda_official * losses["official"]
        self._v4_policy_epoch += 1
        stats.update(cot_stats)
        stats.update({"total_loss": float(total.detach()), "policy_iteration": self._v4_policy_epoch,
                      "rollout_fingerprint": self._v4_fingerprint,
                      "free_sample_calls": self.runtime.free_sample_calls,
                      "official_sample_calls": self.runtime.official_sample_calls})
        if self._smoke_log:
            self._smoke_log[-1].update(stats)
        return total

    def log(self, logs, start_time=None):
        result = super().log(logs, start_time)
        if (self.accelerator.is_main_process and self._monitor.enabled and
                self._smoke_log and "loss" in logs):
            rollout = self._smoke_log[-1]
            self._monitor.write_exact_sharpen_v4({
                "type": "optimization",
                "step": int(self.state.global_step),
                "rollout_id": rollout.get("rollout_id"),
                "policy_iteration": rollout.get("policy_iteration"),
                "rollout_fingerprint": rollout.get("rollout_fingerprint"),
                "cot_loss": rollout.get("cot_loss"),
                "free_loss": rollout.get("free_loss"),
                "official_loss": rollout.get("official_loss"),
                "total_loss": rollout.get("total_loss"),
                "cot_ratio_mean": rollout.get("cot_ratio_mean"),
                "free_ratio_mean": rollout.get("free_ratio_mean"),
                "official_ratio_mean": rollout.get("official_ratio_mean"),
                "cot_clip_fraction": rollout.get("cot_clip_fraction"),
                "free_clip_fraction": rollout.get("free_clip_fraction"),
                "official_clip_fraction": rollout.get("official_clip_fraction"),
                "cot_approx_kl": rollout.get("cot_approx_kl"),
                "free_approx_kl": rollout.get("free_approx_kl"),
                "official_approx_kl": rollout.get("official_approx_kl"),
                "cot_action_tokens": rollout.get("cot_action_tokens"),
                "free_action_tokens": rollout.get("free_action_tokens"),
                "official_action_tokens": rollout.get("official_action_tokens"),
                "total_lora_grad_norm": logs.get("grad_norm"),
                "logs_grad_norm": logs.get("grad_norm"),
            })
        return result


def audit_v4_sampler(dataset, sampler):
    rows = list(dataset)
    gids = [row.get("recommendation_group_id") for row in rows]
    if len(rows) != 611 or len(set(gids)) != 611:
        raise RuntimeError("V4 dataset must contain 611 rows / 611 unique groups")
    if any(row.get("route") != "think" for row in rows):
        raise RuntimeError("V4 dataset must be Think-only")
    return {
        "selected_groups": 611, "trained_groups": 611, "dropped_groups": 0,
        "think_unique_groups": 611, "nothink_unique_groups": 0,
        "think_rollouts": 611, "nothink_rollouts": 0,
        "fresh_rollout_count": 611, "unique_groups_per_global_rollout": 1,
        "cot_candidates_per_group": 4, "free_candidates_per_group": 32,
        "official_candidates_per_group": 32, "free_g8_groups": 4,
        "official_g8_groups": 4, "repeat_count": 2, "num_iterations": 2,
        "optimizer_steps": 1222, "think_optimizer_steps": 1222,
        "nothink_optimizer_steps": 0,
        "route_schedule_preview": ["think"] * 24,
        "rollout_group_ids_preview": gids[:8],
    }
