"""V6-A: all-domain Official GRPO with hierarchical token advantages."""
from __future__ import annotations

import hashlib
import json
import re
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
HISTORY_SID_RE = re.compile(
    r"<\|(video|ad|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)
TARGET_DOMAINS = frozenset(("video", "ad", "prod", "living"))


def fresh_rollout_index_from_step(global_step):
    step = int(global_step)
    if step < 0 or step % 2:
        raise RuntimeError(f"V6-A fresh rollout requires an even global step: {step}")
    return step // 2


def extract_history_region(prompt):
    """Return only the user-history body from an original V3 Think prompt."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    first_newline = prompt.find("\n")
    if first_newline < 0:
        return ""
    region = prompt[first_newline + 1:]
    boundaries = [
        position for marker in ("/think", "</think>", "<|im_start|>assistant")
        if (position := region.find(marker)) >= 0
    ]
    if boundaries:
        region = region[:min(boundaries)]
    return region


def extract_history_sids(prompt, target_domain=None):
    """Parse complete history SIDs, optionally restricted to the target domain."""
    if target_domain is not None and target_domain not in TARGET_DOMAINS:
        raise ValueError(f"unsupported target domain: {target_domain}")
    region = extract_history_region(prompt)
    return {
        (domain, int(a), int(b), int(c))
        for domain, a, b, c in HISTORY_SID_RE.findall(region)
        if target_domain is None or domain == target_domain
    }


def reward_level(value):
    return {
        -1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
        0.5: "A", 2.0: "AB", 8.0: "EXACT",
    }.get(float(value), "unknown")


def reward_hits(raw_q_reward):
    """Map the untouched hierarchical q_reward to cumulative A/B/C hits."""
    raw = float(raw_q_reward)
    return (
        float(raw in (0.5, 2.0, 8.0)),
        float(raw in (2.0, 8.0)),
        float(raw == 8.0),
    )


def finegrained_g8_signal(raw_rewards):
    """Build frozen [G8, ABC] advantages and frontier gate masks."""
    if len(raw_rewards) != SID_G:
        raise ValueError("V6-A fine-grained signal requires one complete G8")
    hits = torch.tensor(
        [reward_hits(value) for value in raw_rewards], dtype=torch.float32
    )
    advantages = torch.stack(
        [population_advantages(hits[:, level]) for level in range(3)], dim=1
    )
    masks = torch.stack(
        [torch.ones(SID_G), hits[:, 0], hits[:, 1]], dim=1
    )
    return {
        "hits": hits,
        "advantages": advantages,
        "masks": masks,
        "active_counts": masks.sum(dim=0).to(torch.int64),
        "zero_std": torch.tensor([
            not bool(torch.any(advantages[:, level] != 0))
            for level in range(3)
        ]),
    }


def official_action_mask(candidate_ids):
    if len(candidate_ids) != SID_G:
        raise ValueError("Official branch must contain Sample8")
    if any(len(candidate) != OFFICIAL_SID_TOKENS for candidate in candidate_ids):
        raise ValueError("Official branch must contain exact ABC3 continuations")
    return [[1] * OFFICIAL_SID_TOKENS for _ in candidate_ids]


def combine_losses(cot_loss, sid_loss):
    return LAMBDA_COT * cot_loss + LAMBDA_SID * sid_loss


def independent_finegrained_signals(groups):
    if len(groups) != COT_G or any(len(group) != SID_G for group in groups):
        raise ValueError("V6-A requires exactly four independent Official G8 groups")
    signals = [finegrained_g8_signal(group) for group in groups]
    return {
        key: torch.stack([signal[key] for signal in signals])
        for key in ("hits", "advantages", "masks", "active_counts", "zero_std")
    }


def _count_levels(record, prefix, copied):
    details = record["candidate_details"]
    for level in ("A", "AB", "EXACT"):
        record[f"{prefix}_{level if level != 'EXACT' else 'Exact'}"] = sum(
            item["reward_level"] == level and item["is_history_copy"] is copied
            for item in details
        )


def finalize_global_records(records, fresh_rollout_index):
    if len(records) != COT_G:
        raise ValueError("V6-A formal topology requires one global G4")
    gids = {row["recommendation_group_id"] for row in records}
    if len(gids) != 1:
        raise RuntimeError(f"V6_G4_GROUP_MIX: {sorted(gids)}")
    domains = {row["target_domain"] for row in records}
    if len(domains) != 1 or not domains.issubset(TARGET_DOMAINS):
        raise RuntimeError(f"V6_MIXED_G4_DOMAIN_MIX: {sorted(domains)}")
    target_domain = next(iter(domains))
    for row in records:
        history = set(row["history_sids"])
        gold = set(row["gold_sids"])
        raw_rewards, copies = [], []
        for sid in row["candidate_sids"]:
            raw = float(q_reward(sid, gold))
            copied = sid is not None and tuple(sid) in history
            raw_rewards.append(raw)
            copies.append(copied)
        signal = finegrained_g8_signal(raw_rewards)
        hits = signal["hits"].tolist()
        advantages = signal["advantages"].tolist()
        masks = signal["masks"].tolist()
        details = []
        for sid, raw, copied, hit, advantage, mask, text in zip(
            row["candidate_sids"], raw_rewards, copies, hits, advantages,
            masks, row["candidate_texts"],
        ):
            details.append({
                "candidate_sid": sid,
                "candidate_text": text,
                "raw_q_reward": raw,
                "reward_level": reward_level(raw),
                "is_history_copy": copied,
                "hit_A": int(hit[0]),
                "hit_B": int(hit[1]),
                "hit_C": int(hit[2]),
                "adv_A": float(advantage[0]),
                "adv_B": float(advantage[1]),
                "adv_C": float(advantage[2]),
                "mask_A": int(mask[0]),
                "mask_B": int(mask[1]),
                "mask_C": int(mask[2]),
            })
        copy_advantages = [
            advantage
            for item in details if item["is_history_copy"]
            for advantage, mask in zip(
                (item["adv_A"], item["adv_B"], item["adv_C"]),
                (item["mask_A"], item["mask_B"], item["mask_C"]),
            ) if mask
        ]
        row.update({
            "raw_q_rewards": raw_rewards,
            "is_history_copy": copies,
            "sid_rewards": raw_rewards,
            "hit_A": [int(value[0]) for value in hits],
            "hit_B": [int(value[1]) for value in hits],
            "hit_C": [int(value[2]) for value in hits],
            "token_advantages": advantages,
            "token_masks": [[int(value) for value in mask] for mask in masks],
            "candidate_details": details,
            "cot_contributions": raw_rewards,
            "cot_reward": float(sum(raw_rewards)),
            "fresh_rollout_index": int(fresh_rollout_index),
            "reward_stage": "finegrained_frontier",
            "history_sid_count": len(history),
            "gold_history_exact_overlap": len(gold & history),
            "copy_count": sum(copies),
            "copy_rate": sum(copies) / SID_G,
            "copy_positive_advantage_count": sum(
                item["is_history_copy"] and any(
                    advantage > 0 and mask
                    for advantage, mask in zip(
                        (item["adv_A"], item["adv_B"], item["adv_C"]),
                        (item["mask_A"], item["mask_B"], item["mask_C"]),
                    )
                ) for item in details
            ),
            "copy_mean_advantage": (
                sum(copy_advantages) / len(copy_advantages) if copy_advantages else 0.0
            ),
            "active_A_tokens": int(signal["active_counts"][0]),
            "active_B_tokens": int(signal["active_counts"][1]),
            "active_C_tokens": int(signal["active_counts"][2]),
            "zero_std_A": bool(signal["zero_std"][0]),
            "zero_std_B": bool(signal["zero_std"][1]),
            "zero_std_C": bool(signal["zero_std"][2]),
        })
        _count_levels(row, "copy", True)
        _count_levels(row, "noncopy", False)
    cot_advantages = population_advantages([row["cot_reward"] for row in records]).tolist()
    for row, advantage in zip(records, cot_advantages):
        row["cot_advantage"] = float(advantage)
        row["zero_std_g8"] = all(
            row[key] for key in ("zero_std_A", "zero_std_B", "zero_std_C")
        )
    return records


def rollout_fingerprint(records):
    payload = [{
        "gid": row["recommendation_group_id"],
        "cot": row["cot_ids"],
        "official": row["candidate_ids"],
        "raw": row["raw_q_rewards"],
        "hits": [row["hit_A"], row["hit_B"], row["hit_C"]],
        "token_advantages": row["token_advantages"],
        "token_masks": row["token_masks"],
        "cot_reward": row["cot_reward"],
        "fresh_rollout_index": row["fresh_rollout_index"],
    } for row in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def domain_rollout_summary(records):
    """Aggregate the frozen G4/G8 signal without changing any reward math."""
    details = [item for row in records for item in row["candidate_details"]]
    copies = [item for item in details if item["is_history_copy"]]
    copy_advantages = [
        advantage
        for item in copies
        for advantage, mask in zip(
            (item["adv_A"], item["adv_B"], item["adv_C"]),
            (item["mask_A"], item["mask_B"], item["mask_C"]),
        ) if mask
    ]
    summary = {
        "domain": records[0]["target_domain"],
        "fresh_rollout_index": records[0]["fresh_rollout_index"],
        "reward_stage": records[0]["reward_stage"],
        "history_copy_rate": len(copies) / len(details) if details else 0.0,
        "raw_reward_mean": (
            sum(item["raw_q_reward"] for item in details) / len(details)
            if details else 0.0
        ),
        "shaped_reward_mean": (
            sum(item["raw_q_reward"] for item in details) / len(details)
            if details else 0.0
        ),
        "copy_positive_advantage_count": sum(
            item["is_history_copy"] and any(
                advantage > 0 and mask
                for advantage, mask in zip(
                    (item["adv_A"], item["adv_B"], item["adv_C"]),
                    (item["mask_A"], item["mask_B"], item["mask_C"]),
                )
            )
            for item in details
        ),
        "copy_mean_advantage": (
            sum(copy_advantages) / len(copy_advantages) if copy_advantages else 0.0
        ),
        "cot_length_mean": sum(row["cot_length"] for row in records) / len(records),
        "cot_rewards": [row["cot_reward"] for row in records],
        "cot_advantages": [row["cot_advantage"] for row in records],
        "zero_std_g8_count": sum(row["zero_std_g8"] for row in records),
        "active_A_tokens": sum(row["active_A_tokens"] for row in records),
        "active_B_tokens": sum(row["active_B_tokens"] for row in records),
        "active_C_tokens": sum(row["active_C_tokens"] for row in records),
        "zero_std_A_count": sum(row["zero_std_A"] for row in records),
        "zero_std_B_count": sum(row["zero_std_B"] for row in records),
        "zero_std_C_count": sum(row["zero_std_C"] for row in records),
    }
    for prefix in ("copy", "noncopy"):
        for level in ("A", "AB", "Exact"):
            summary[f"{prefix}_{level}"] = sum(
                row[f"{prefix}_{level}"] for row in records
            )
    return summary


def assert_iteration_reuse(expected_fingerprint, actual_fingerprint, policy_iteration,
                           sample_calls_before, sample_calls_after):
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("V6-A iteration2 rollout fingerprint changed")
    if policy_iteration not in (1, 2):
        raise RuntimeError("V6-A rollout may be optimized exactly twice")
    if policy_iteration == 2 and sample_calls_after != sample_calls_before:
        raise RuntimeError("V6-A iteration2 resampled Official candidates")
    return True


class OfficialFineGrainedRuntime:
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
            raise RuntimeError("V6_THINK_CLOSE_TOKEN_CONTRACT_FAILED")
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
            raise RuntimeError(f"V6-A unsupported domain: {target_domain}")
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
        history = extract_history_sids(prompt, target_domain)
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
            "history_sids": sorted(history),
            "rollout_wall_sec": time.perf_counter() - started,
        }


def make_official_finegrained_reward_func(runtime):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        del completions
        if any(route != "think" for route in kwargs.get("route", [])):
            raise RuntimeError("V6-A requires Think-only samples")
        if any(domain not in TARGET_DOMAINS for domain in kwargs.get("target_domain", [])):
            raise RuntimeError("V6-A requires a supported target domain")
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
            raise RuntimeError("V6-A formal shape requires one local CoT per rank")
        records = finalize_global_records(
            runtime._gather(local[0]), runtime.fresh_rollouts
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        runtime.last_global_records = records
        runtime.last_local = records[rank]
        return [runtime.last_local["cot_reward"]]

    reward_func.__name__ = "official_finegrained_v6_cot_reward"
    reward_func.official_finegrained_runtime = runtime
    return reward_func


class ThinkOfficialFineGrainedTrainer(RecGRPOTrainer):
    def __init__(self, *args, lambda_cot=LAMBDA_COT, lambda_sid=LAMBDA_SID, **kwargs):
        funcs = kwargs.get("reward_funcs") or (args[3] if len(args) >= 4 else [])
        runtime = next((
            getattr(fn, "official_finegrained_runtime", None)
            for fn in funcs
            if getattr(fn, "official_finegrained_runtime", None) is not None
        ), None)
        if runtime is None:
            raise RuntimeError("V6-A trainer requires OfficialFineGrainedRuntime")
        if (float(lambda_cot), float(lambda_sid)) != (1.0, 1.0):
            raise ValueError("V6-A loss weights are frozen at 1:1")
        self.runtime = runtime
        self._v6_rollout = None
        self._v6_policy_epoch = 0
        self._v6_fingerprint = None
        self._v6_sample_calls_at_rollout = None
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
            raise RuntimeError("V6-A is Think-only")
        return {"group_size": COT_G, "temperature": 0.9, "top_p": 0.95}

    def _prepare_inputs(self, batch):
        rows = batch if isinstance(batch, list) else []
        if rows and any(
            row.get("route") != "think" or row.get("target_domain") not in TARGET_DOMAINS
            for row in rows
        ):
            raise RuntimeError("V6-A dataset must be Think-only and all-domain")
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

    def _build_v6_rollout(self, cot_output):
        records = self.runtime.last_global_records
        record = self.runtime.last_local
        cot_advantages = population_advantages([row["cot_reward"] for row in records])
        rank = self.accelerator.process_index
        if not torch.allclose(
            cot_output["advantages"].float().cpu(),
            cot_advantages[rank:rank + 1],
            atol=1e-6,
        ):
            raise RuntimeError("V6_COT_ADVANTAGE_PARITY_FAILED")
        cot_old = cot_output.get("old_per_token_logps")
        if cot_old is None or cot_old.requires_grad:
            raise RuntimeError("V6_COT_OLD_LOGP_FAILED")
        input_ids, attention, full_action = self._pad_official(record)
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention, full_action.size(1), compute_entropy=False
            )
        if old_logps.shape != full_action.shape or old_logps.requires_grad:
            raise RuntimeError("V6_OFFICIAL_OLD_LOGP_CONTRACT_FAILED")
        token_advantages = torch.tensor(
            record["token_advantages"], dtype=torch.float32, device=input_ids.device
        )
        token_masks = torch.tensor(
            record["token_masks"], dtype=torch.long, device=input_ids.device
        )
        if token_advantages.shape != old_logps.shape or token_masks.shape != old_logps.shape:
            raise RuntimeError("V6_TOKEN_SIGNAL_SHAPE_FAILED")
        self._v6_rollout = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "action_mask": token_masks,
            "old_per_token_logps": old_logps.detach(),
            "advantages": token_advantages,
            "record": record,
            "global_records": records,
        }
        self._v6_fingerprint = rollout_fingerprint(records)
        self._v6_policy_epoch = 0
        self._v6_sample_calls_at_rollout = self.runtime.sample_calls
        self.runtime.fresh_rollouts += 1
        if self.accelerator.is_main_process and self._monitor.enabled:
            summary = domain_rollout_summary(records)
            self._monitor.write_official_finegrained_v6({
                "step": int(self.state.global_step),
                "rollout_id": int(self._smoke_rollout_id),
                "rollout_fingerprint": self._v6_fingerprint,
                "recommendation_group_id": records[0]["recommendation_group_id"],
                "target_domain": records[0]["target_domain"],
                "fresh_rollout_index": records[0]["fresh_rollout_index"],
                "reward_stage": "finegrained_frontier",
                "cots": records,
                "domain_summary": summary,
                "normalization_topology": "G4 + 4 independent Official G8; never G32",
            })

    def _generate_and_score_completions(self, inputs):
        self.runtime.fresh_rollouts = fresh_rollout_index_from_step(
            self.state.global_step
        )
        before = self.runtime.sample_calls
        for attempt in range(MAX_COT_CLOSURE_RETRIES + 1):
            self.runtime.closure_attempt = attempt
            metric_lengths = {key: len(values) for key, values in self._metrics["train"].items()}
            output = super()._generate_and_score_completions(inputs)
            if self.runtime.last_global_cot_closed:
                self._build_v6_rollout(output)
                if self.runtime.sample_calls - before != len(inputs):
                    raise RuntimeError("V6_EXPECTED_ONE_OFFICIAL_SAMPLE8_PER_COT")
                return output
            for key in list(self._metrics["train"]):
                del self._metrics["train"][key][metric_lengths.get(key, 0):]
            if self.runtime.sample_calls != before:
                raise RuntimeError("V6_SID_SAMPLED_BEFORE_COT_ACCEPTANCE")
        raise RuntimeError("V6_COT_NOT_CLOSED_AFTER_RETRIES")

    def _branch_loss(self, model, input_ids, attention, action, old_logps,
                     advantages, name):
        current, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention, action.size(1), compute_entropy=False
        )
        log_ratio = current - old_logps
        ratio = log_ratio.exp()
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if advantages.ndim == 1:
            objective_advantages = advantages.unsqueeze(1)
            token_level = False
        elif advantages.shape == current.shape:
            objective_advantages = advantages
            token_level = True
        else:
            raise RuntimeError(
                f"V6_{name.upper()}_ADVANTAGE_SHAPE_FAILED: "
                f"advantages={tuple(advantages.shape)} logps={tuple(current.shape)}"
            )
        per_token = -torch.min(
            ratio * objective_advantages,
            clipped * objective_advantages,
        )
        if token_level:
            loss = (per_token * action).sum() / action.sum().clamp(min=1)
        else:
            loss = (
                (per_token * action).sum(-1) / action.sum(-1).clamp(min=1)
            ).mean()
        loss = loss / self.current_gradient_accumulation_steps
        selected = log_ratio[action.bool()].detach().float()
        stats = {
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
        if token_level and name == "sid":
            for index, level in enumerate(("A", "B", "C")):
                level_selected = log_ratio[:, index][action[:, index].bool()].detach().float()
                level_ratio = level_selected.exp()
                stats.update({
                    f"sid_{level}_ratio_mean": (
                        float(level_ratio.mean()) if level_selected.numel() else 1.0
                    ),
                    f"sid_{level}_clip_fraction": (
                        float(((level_ratio - 1).abs() > self.epsilon_low).float().mean())
                        if level_selected.numel() else 0.0
                    ),
                    f"sid_{level}_approx_kl": (
                        float((level_ratio - 1 - level_selected).mean())
                        if level_selected.numel() else 0.0
                    ),
                    f"sid_{level}_action_tokens": int(action[:, index].sum()),
                })
        return loss, stats

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        if self._v6_rollout is None or self._v6_policy_epoch >= 2:
            raise RuntimeError("V6-A rollout unavailable or reused more than twice")
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
        sid = self._v6_rollout
        sid_loss, sid_stats = self._branch_loss(
            model, sid["input_ids"], sid["attention_mask"], sid["action_mask"],
            sid["old_per_token_logps"], sid["advantages"], "sid",
        )
        total = combine_losses(cot_loss, sid_loss)
        self._v6_policy_epoch += 1
        assert_iteration_reuse(
            self._v6_fingerprint, rollout_fingerprint(sid["global_records"]),
            self._v6_policy_epoch, self._v6_sample_calls_at_rollout,
            self.runtime.sample_calls,
        )
        stats = {
            **cot_stats, **sid_stats,
            "total_loss": float(total.detach()),
            "policy_iteration": self._v6_policy_epoch,
            "rollout_fingerprint": self._v6_fingerprint,
            "official_sample_calls": self.runtime.sample_calls,
            "sid_active_A_tokens": sid["record"]["active_A_tokens"],
            "sid_active_B_tokens": sid["record"]["active_B_tokens"],
            "sid_active_C_tokens": sid["record"]["active_C_tokens"],
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
            record = self._v6_rollout["record"] if self._v6_rollout else {}
            grad_norm = logs.get("grad_norm")
            self._monitor.write_official_finegrained_v6({
                "type": "optimization",
                "step": int(self.state.global_step),
                "rollout_id": rollout.get("rollout_id"),
                "policy_iteration": rollout.get("policy_iteration"),
                "rollout_fingerprint": rollout.get("rollout_fingerprint"),
                "target_domain": record.get("target_domain"),
                "fresh_rollout_index": record.get("fresh_rollout_index"),
                "reward_stage": record.get("reward_stage"),
                "cot_loss": rollout.get("cot_loss"),
                "sid_loss": rollout.get("sid_loss"),
                "total_loss": rollout.get("total_loss"),
                "cot_ratio_mean": rollout.get("cot_ratio_mean"),
                "sid_ratio_mean": rollout.get("sid_ratio_mean"),
                "cot_clip_fraction": rollout.get("cot_clip_fraction"),
                "sid_clip_fraction": rollout.get("sid_clip_fraction"),
                "cot_approx_kl": rollout.get("cot_approx_kl"),
                "sid_approx_kl": rollout.get("sid_approx_kl"),
                "sid_A_ratio_mean": rollout.get("sid_A_ratio_mean"),
                "sid_A_clip_fraction": rollout.get("sid_A_clip_fraction"),
                "sid_A_approx_kl": rollout.get("sid_A_approx_kl"),
                "sid_B_ratio_mean": rollout.get("sid_B_ratio_mean"),
                "sid_B_clip_fraction": rollout.get("sid_B_clip_fraction"),
                "sid_B_approx_kl": rollout.get("sid_B_approx_kl"),
                "sid_C_ratio_mean": rollout.get("sid_C_ratio_mean"),
                "sid_C_clip_fraction": rollout.get("sid_C_clip_fraction"),
                "sid_C_approx_kl": rollout.get("sid_C_approx_kl"),
                "cot_action_tokens": rollout.get("cot_action_tokens"),
                "sid_action_tokens": rollout.get("sid_action_tokens"),
                "sid_A_action_tokens": rollout.get("sid_A_action_tokens"),
                "sid_B_action_tokens": rollout.get("sid_B_action_tokens"),
                "sid_C_action_tokens": rollout.get("sid_C_action_tokens"),
                "sid_active_A_tokens": rollout.get("sid_active_A_tokens"),
                "sid_active_B_tokens": rollout.get("sid_active_B_tokens"),
                "sid_active_C_tokens": rollout.get("sid_active_C_tokens"),
                "total_lora_grad_norm": grad_norm,
                "logs_grad_norm": grad_norm,
                "zero_gradient_group": grad_norm is not None and float(grad_norm) == 0.0,
            })
        return result


def audit_v6_sampler(dataset, sampler):
    del sampler
    rows = list(dataset)
    gids = [row.get("recommendation_group_id") for row in rows]
    if not rows or len(gids) != len(set(gids)):
        raise RuntimeError("V6-A requires one row per unique business group")
    if any(
        row.get("route") != "think" or row.get("target_domain") not in TARGET_DOMAINS
        for row in rows
    ):
        raise RuntimeError("V6-A dataset must be Think-only and all-domain")
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
