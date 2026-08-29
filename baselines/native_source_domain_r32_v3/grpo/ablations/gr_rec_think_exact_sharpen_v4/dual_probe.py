"""Checkpoint Probe4 evaluator with separate Free and Official contracts."""
from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict

import torch
import torch.distributed as dist

from grpo_model import encode_prompt
from grpo_probe import FixedProbeEvaluator, _completion_sha256, _gold_set
from grpo_sid import final_sid, q_reward

from .exact_sharpen_trainer import FREE_MAX_NEW_TOKENS, SID_G, scan_full_sid_ids


def probe_summary(candidates):
    rewards = [float(item["reward"]) for item in candidates]
    return {
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "a_count": sum(int(item.get("a") or 0) for item in candidates)
                   if any("a" in item for item in candidates) else sum(float(v) == 0.5 for v in rewards),
        "ab_count": sum(int(item.get("ab") or 0) for item in candidates)
                    if any("ab" in item for item in candidates) else sum(float(v) == 2.0 for v in rewards),
        "exact_count": sum(int(item.get("exact") or 0) for item in candidates)
                       if any("exact" in item for item in candidates) else sum(float(v) == 8.0 for v in rewards),
        "invalid_count": sum(int(item.get("invalid") or 0) for item in candidates)
                         if any("invalid" in item for item in candidates) else sum(float(v) == -1.0 for v in rewards),
        "candidates": candidates,
    }


class DualContractProbeEvaluator(FixedProbeEvaluator):
    """Reuse one fixed CoT batch, then route it to Free Sample8 and Beam32."""

    def _free_sample8(self, prompt, cot_ids, gold):
        close_ids = self.trainer.processing_class.encode("</think>", add_special_tokens=False)
        if len(close_ids) != 1 or close_ids[0] not in cot_ids:
            return [{"parsed_sid": None, "reward": -1.0, "closed": False}] * SID_G
        close_at = list(cot_ids).index(close_ids[0])
        cot_trim = list(cot_ids[:close_at + 1])
        context = encode_prompt(self.trainer.processing_class, prompt) + cot_trim
        input_ids = torch.tensor([context], dtype=torch.long, device=self.trainer.model.device)
        tokenizer = self.trainer.processing_class
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = self.trainer.model
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                output = model.generate(
                    inputs=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    num_beams=1, num_return_sequences=SID_G,
                    min_new_tokens=4, max_new_tokens=FREE_MAX_NEW_TOKENS, pad_token_id=pad_id,
                )[:, input_ids.size(1):]
        finally:
            model.train(was_training)
        candidates = []
        for generated in output.tolist():
            ids = [int(v) for v in generated]
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            scan = scan_full_sid_ids(tokenizer, ids)
            sid = scan["parsed_sid"]
            candidates.append({
                "completion": tokenizer.decode(ids, skip_special_tokens=False),
                "completion_length": len(ids), "completion_sha256": _completion_sha256(ids),
                "closed": True, "parsed_sid": sid, "reward": q_reward(sid, gold),
                "parser_status": scan["parser_status"], "fixed_domain_prefix": False,
            })
        if len(candidates) != SID_G:
            raise RuntimeError("V4 free Probe must return Sample8")
        return candidates

    def _think_dual(self):
        rank = self.trainer.accelerator.process_index
        gid = self.group_ids[rank]
        row = self.records[gid]["think"]
        prompts = [row["prompt"]] * 4
        self.trainer._set_route_config("think")
        torch.cuda.synchronize()
        started = time.perf_counter()
        completion_ids = self._generate(prompts)
        generation_wall = time.perf_counter() - started
        texts = [self.trainer.processing_class.decode(ids, skip_special_tokens=False)
                 for ids in completion_ids]
        gold = _gold_set(row["all_gold_sids"])
        free = [self._free_sample8(row["prompt"], ids, gold) for ids in completion_ids]
        official_rewards = self.beam32_fn(
            prompts, texts, completion_ids, [gold] * 4, [row["target_domain"]] * 4,
            recommendation_group_ids=[gid] * 4)
        beam_call = self.trainer._current_beam_call() or {}
        results = beam_call.get("local_results", [{} for _ in completion_ids])
        official = []
        for ids, text, reward, result in zip(completion_ids, texts, official_rewards, results):
            official.append({
                "completion": text, "completion_length": len(ids),
                "completion_sha256": _completion_sha256(ids), "closed": "</think>" in text,
                "reward": reward, "exact": result.get("exact"), "ab": result.get("ab"),
                "a": result.get("a"), "invalid": result.get("invalid"),
                "beam_sids": result.get("beam_sids"), "fixed_domain_prefix": True,
                "beam_search_space": "ABC_CONTINUATION_AFTER_FIXED_DOMAIN",
            })
        return {
            "group_id": gid, "prompt": row["prompt"], "gold_sids": row["all_gold_sids"],
            "target_domain": row["target_domain"],
            "probe_free": probe_summary([item for group in free for item in group]),
            "probe_official": probe_summary(official),
            "generation_wall_sec": generation_wall,
            "wall_sec": time.perf_counter() - started,
        }

    def evaluate(self, step, reason):
        if self.last_step == step or self._already_complete(step):
            self.last_step = step
            return
        if self.trainer.accelerator.num_processes != 4:
            raise RuntimeError("dual Probe4 requires four ranks")
        self.trainer.accelerator.wait_for_everyone()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.trainer.accelerator.device)
        python_rng = random.getstate()
        previous_route = (self.trainer.num_generations, self.trainer.args.temperature,
                          self.trainer.args.top_p, self.trainer.temperature, self.trainer.top_p)
        started = time.perf_counter()
        try:
            self._set_seed()
            think_by_rank = self._gather(self._think_dual())
            nothink_parts = []
            for batch_index in range(2):
                nothink_parts.extend(self._gather(self._nothink_batch(batch_index)))
            rank_walls = self._gather(time.perf_counter() - started)
            if self.trainer.accelerator.is_main_process:
                no_by_group = defaultdict(list)
                no_meta = {}
                for part in nothink_parts:
                    no_by_group[part["group_id"]].extend(part["candidates"])
                    no_meta[part["group_id"]] = part
                for part in think_by_rank:
                    gid = part["group_id"]
                    free, official = part["probe_free"], part["probe_official"]
                    self.monitor.write_probe({
                        "step": int(step), "reason": reason, "group_id": gid,
                        "target_domain": part["target_domain"], "gold_sids": part["gold_sids"],
                        "think_prompt": part["prompt"],
                        "nothink_prompt": no_meta[gid]["prompt"],
                        "probe_free": free, "probe_official": official,
                        "probe_free_reward_mean": free["reward_mean"],
                        "probe_free_reward_std": free["reward_std"],
                        "probe_free_a_count": free["a_count"],
                        "probe_free_ab_count": free["ab_count"],
                        "probe_free_exact_count": free["exact_count"],
                        "probe_official_reward_mean": official["reward_mean"],
                        "probe_official_reward_std": official["reward_std"],
                        "probe_official_a_count": official["a_count"],
                        "probe_official_ab_count": official["ab_count"],
                        "probe_official_exact_count": official["exact_count"],
                        "think": official,
                        "nothink": self._summary(no_by_group[gid], think=False),
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
            self.trainer.accelerator.wait_for_everyone()
        self.last_step = step
