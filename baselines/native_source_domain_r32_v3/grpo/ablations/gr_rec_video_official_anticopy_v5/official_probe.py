"""Official Beam32 probes with anti-copy anatomy for V5."""
from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict

import torch
import torch.distributed as dist

from grpo_probe import FixedProbeEvaluator, _completion_sha256, _gold_set
from grpo_sid import parse_sid, q_reward

from .video_official_anticopy_trainer import extract_history_sids, reward_level


def _normalize_sid(value):
    if value is None:
        return None
    if isinstance(value, str):
        return parse_sid(value)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return (str(value[0]), int(value[1]), int(value[2]), int(value[3]))
    return None


def beam_copy_details(beam_sids, gold_sids, history_sids):
    gold, history = set(gold_sids), set(history_sids)
    details = []
    for value in beam_sids or []:
        sid = _normalize_sid(value)
        raw = float(q_reward(sid, gold))
        copied = sid is not None and sid in history
        details.append({
            "candidate_sid": sid,
            "raw_q_reward": raw,
            "reward_level": reward_level(raw),
            "is_history_copy": copied,
        })
    return details


def official_probe_summary(candidates):
    rewards = [float(item["reward"]) for item in candidates]
    beam_details = [
        detail for item in candidates for detail in item.get("beam_candidate_details", [])
    ]
    copied = [item for item in beam_details if item["is_history_copy"]]
    noncopied = [item for item in beam_details if not item["is_history_copy"]]
    result = {
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
        "reward_semantics": "production hierarchical Beam32 reward per sampled CoT",
        "reward_denominator": f"{len(candidates)} sampled CoTs",
        "candidate_count": len(candidates),
        "beam_candidate_count": len(beam_details),
        "closure_rate": (
            sum(bool(item.get("closed")) for item in candidates) / len(candidates)
            if candidates else 0.0
        ),
        "a_count": sum(int(item.get("a") or 0) for item in candidates),
        "ab_count": sum(int(item.get("ab") or 0) for item in candidates),
        "exact_count": sum(int(item.get("exact") or 0) for item in candidates),
        "invalid_count": sum(int(item.get("invalid") or 0) for item in candidates),
        "history_copy_count": len(copied),
        "history_copy_rate": len(copied) / len(beam_details) if beam_details else 0.0,
        "copy_A": sum(item["reward_level"] == "A" for item in copied),
        "copy_AB": sum(item["reward_level"] == "AB" for item in copied),
        "copy_Exact": sum(item["reward_level"] == "EXACT" for item in copied),
        "noncopy_A": sum(item["reward_level"] == "A" for item in noncopied),
        "noncopy_AB": sum(item["reward_level"] == "AB" for item in noncopied),
        "noncopy_Exact": sum(item["reward_level"] == "EXACT" for item in noncopied),
        "cot_length_mean": (
            statistics.fmean(float(item["completion_length"]) for item in candidates)
            if candidates else 0.0
        ),
        "candidates": candidates,
    }
    return result


def restore_beam_stats(model, had_beam_stats, previous_beam_stats):
    if had_beam_stats:
        model._beam_stats = previous_beam_stats
    elif hasattr(model, "_beam_stats"):
        delattr(model, "_beam_stats")


class VideoOfficialProbeEvaluator(FixedProbeEvaluator):
    """Keep original all-domain Probe4 and add fixed held-out Video groups."""

    def __init__(self, *args, heldout_records=None, heldout_group_ids=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.heldout_records = dict(heldout_records or {})
        self.heldout_group_ids = list(heldout_group_ids)
        if set(self.group_ids) & set(self.heldout_group_ids):
            raise RuntimeError("V5 held-out Video probe overlaps Probe4")

    def _enrich_candidates(self, prompt, gold, candidates):
        history = extract_history_sids(prompt)
        for candidate in candidates:
            candidate["history_sid_count"] = len(history)
            candidate["gold_history_exact_overlap"] = len(set(gold) & history)
            candidate["beam_candidate_details"] = beam_copy_details(
                candidate.get("beam_sids"), gold, history
            )
        return candidates

    def _think(self):
        part = super()._think()
        gold = _gold_set(part["gold_sids"])
        part["candidates"] = self._enrich_candidates(
            part["prompt"], gold, part["candidates"]
        )
        return part

    @staticmethod
    def _summary(candidates, *, think):
        if think:
            return official_probe_summary(candidates)
        return FixedProbeEvaluator._summary(candidates, think=False)

    def _heldout_complete(self, step):
        complete = False
        if self.trainer.accelerator.is_main_process:
            path = self.monitor.run_dir / "probes.jsonl"
            seen = set()
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("step") == step
                        and row.get("probe_suite") == "heldout_video_official"
                    ):
                        seen.add(row.get("group_id"))
            complete = set(self.heldout_group_ids).issubset(seen)
        decision = [complete]
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        return decision[0]

    def _official_group(self, gid):
        row = self.heldout_records[gid]["think"]
        prompts = [row["prompt"]] * 4
        self.trainer._set_route_config("think")
        started = time.perf_counter()
        completion_ids = self._generate(prompts)
        texts = [
            self.trainer.processing_class.decode(ids, skip_special_tokens=False)
            for ids in completion_ids
        ]
        gold = _gold_set(row["all_gold_sids"])
        rewards = self.beam32_fn(
            prompts, texts, completion_ids, [gold] * 4,
            [row["target_domain"]] * 4,
            recommendation_group_ids=[gid] * 4,
        )
        beam_call = self.trainer._current_beam_call() or {}
        results = beam_call.get("local_results", [{} for _ in completion_ids])
        close_ids = self.trainer.processing_class.encode(
            "</think>", add_special_tokens=False
        )
        if len(close_ids) != 1:
            raise RuntimeError("V5 probe requires one </think> token")
        candidates = []
        for ids, text, reward, result in zip(
            completion_ids, texts, rewards, results
        ):
            candidates.append({
                "completion": text,
                "completion_length": len(ids),
                "completion_sha256": _completion_sha256(ids),
                "closed": close_ids[0] in ids,
                "reward": reward,
                "exact": result.get("exact"),
                "ab": result.get("ab"),
                "a": result.get("a"),
                "invalid": result.get("invalid"),
                "beam_sids": result.get("beam_sids"),
                "fixed_domain_prefix": True,
                "beam_search_space": "ABC_CONTINUATION_AFTER_FIXED_VIDEO_DOMAIN",
            })
        self._enrich_candidates(row["prompt"], gold, candidates)
        return {
            "group_id": gid,
            "prompt": row["prompt"],
            "gold_sids": row["all_gold_sids"],
            "target_domain": row["target_domain"],
            "think": official_probe_summary(candidates),
            "wall_sec": time.perf_counter() - started,
        }

    def _evaluate_heldout(self, step, reason):
        if not self.heldout_group_ids or self._heldout_complete(step):
            return
        self.trainer.accelerator.wait_for_everyone()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.trainer.accelerator.device)
        python_rng = random.getstate()
        previous_route = (
            self.trainer.num_generations,
            self.trainer.args.temperature,
            self.trainer.args.top_p,
            self.trainer.temperature,
            self.trainer.top_p,
        )
        model = self.trainer.model
        had_beam_stats = hasattr(model, "_beam_stats")
        previous_beam_stats = getattr(model, "_beam_stats", None)
        try:
            self._set_seed()
            rank = self.trainer.accelerator.process_index
            local = [
                self._official_group(gid)
                for gid in self.heldout_group_ids[rank::4]
            ]
            gathered = self._gather(local)
            if self.trainer.accelerator.is_main_process:
                for part in [item for rank_items in gathered for item in rank_items]:
                    self.monitor.write_probe({
                        "step": int(step),
                        "reason": reason,
                        "probe_suite": "heldout_video_official",
                        "group_id": part["group_id"],
                        "target_domain": "video",
                        "gold_sids": part["gold_sids"],
                        "think_prompt": part["prompt"],
                        "think": part["think"],
                        "probe_wall_sec": part["wall_sec"],
                        "seed": self.seed,
                    })
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.trainer.accelerator.device)
            random.setstate(python_rng)
            (
                self.trainer.num_generations,
                self.trainer.args.temperature,
                self.trainer.args.top_p,
                self.trainer.temperature,
                self.trainer.top_p,
            ) = previous_route
            if getattr(self.trainer, "generation_config", None) is not None:
                self.trainer.generation_config.temperature = previous_route[1]
                self.trainer.generation_config.top_p = previous_route[2]
            restore_beam_stats(model, had_beam_stats, previous_beam_stats)
            self.trainer.accelerator.wait_for_everyone()

    def evaluate(self, step, reason):
        super().evaluate(step, reason)
        self._evaluate_heldout(step, reason)
