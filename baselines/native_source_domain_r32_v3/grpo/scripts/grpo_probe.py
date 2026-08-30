"""Deterministic held-out probes for low-frequency GRPO evaluation."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from grpo_sid import final_sid, parse_sid, q_reward


PROBE_DOMAINS = ("video", "living", "prod", "ad")


def probe_group_batches(group_ids):
    group_ids = list(group_ids)
    if len(group_ids) % len(PROBE_DOMAINS):
        raise ValueError("fixed probe group count must be a multiple of 4")
    return [group_ids[start:start + len(PROBE_DOMAINS)]
            for start in range(0, len(group_ids), len(PROBE_DOMAINS))]


def select_probe_group_ids(data_path, n_groups, seed, count=0, explicit_ids=None):
    """Select an equal number of held-out groups for each target domain."""
    rows = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    gids = sorted({row["recommendation_group_id"] for row in rows})
    domains_by_gid = defaultdict(set)
    for row in rows:
        domains_by_gid[row["recommendation_group_id"]].add(row["target_domain"])
    inconsistent = {gid: values for gid, values in domains_by_gid.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"probe groups have inconsistent target domains: {inconsistent}")
    domain_by_gid = {gid: next(iter(values)) for gid, values in domains_by_gid.items()}
    rng = random.Random(seed)
    rng.shuffle(gids)
    selected = gids[:n_groups]
    explicit = list(dict.fromkeys(explicit_ids or ()))
    effective_count = len(explicit) if explicit and count == 0 else count
    if effective_count and effective_count % len(PROBE_DOMAINS):
        raise ValueError("--probe-groups must be a multiple of 4")
    per_domain = effective_count // len(PROBE_DOMAINS) if effective_count else 0
    if explicit:
        if count not in (0, len(explicit)):
            raise ValueError("--probe-groups must match the number of --probe-group-id values")
        probes = explicit
    else:
        probes_by_domain = defaultdict(list)
        for gid in selected:
            domain = domain_by_gid[gid]
            if domain in PROBE_DOMAINS and len(probes_by_domain[domain]) < per_domain:
                probes_by_domain[domain].append(gid)
        probes = [probes_by_domain[domain][index]
                  for index in range(per_domain)
                  for domain in PROBE_DOMAINS
                  if index < len(probes_by_domain[domain])]
    if probes and (len(probes) % len(PROBE_DOMAINS) or not per_domain):
        missing_domains = [domain for domain in PROBE_DOMAINS
                           if domain not in {domain_by_gid.get(gid) for gid in probes}]
        raise ValueError(
            "fixed probes require an equal number of groups covering video/living/prod/ad; "
            f"missing domains: {missing_domains}"
        )
    missing = set(probes).difference(selected)
    if missing:
        raise ValueError(f"probe group IDs are outside the selected dataset: {sorted(missing)}")
    probe_domains = [domain_by_gid[gid] for gid in probes]
    expected_domains = list(PROBE_DOMAINS) * per_domain
    if probes and probe_domains != expected_domains:
        raise ValueError(
            "fixed probe groups must repeat the video/living/prod/ad order; "
            f"got {probe_domains}"
        )
    return probes


def load_probe_records(data_path, group_ids):
    wanted = set(group_ids)
    by_group = defaultdict(dict)
    for line in open(data_path, encoding="utf-8"):
        row = json.loads(line)
        gid = row["recommendation_group_id"]
        if gid in wanted:
            by_group[gid][row["route"]] = row
    for gid in group_ids:
        if set(by_group[gid]) != {"think", "no_think"}:
            raise ValueError(f"probe group {gid!r} does not contain both routes")
    return {gid: by_group[gid] for gid in group_ids}


def validate_probe_schedule(group_ids, every_steps):
    if not group_ids:
        return
    if every_steps < 2 or every_steps % 2:
        raise ValueError("--probe-every-steps must be a positive even number")


def probe_due(step, every_steps):
    return step == 0 or (step > 0 and step % every_steps == 0)


def _completion_sha256(ids):
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _gold_set(values):
    return {sid for value in values if (sid := parse_sid(value)) is not None}


class FixedProbeEvaluator:
    """Run production-shaped Think/NoThink sampling without optimizer updates."""

    def __init__(self, trainer, records, group_ids, beam32_fn, monitor, seed, every_steps,
                 probe_suite=None):
        self.trainer = trainer
        self.records = records
        self.group_ids = list(group_ids)
        self.beam32_fn = beam32_fn
        self.monitor = monitor
        self.seed = int(seed)
        self.every_steps = int(every_steps)
        self.probe_suite = probe_suite
        self.last_step = None
        self._group_offset = 0

    def _already_complete(self, step):
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
                    if row.get("step") == step and row.get("probe_suite") == self.probe_suite:
                        seen.add(row.get("group_id"))
            complete = set(self.group_ids).issubset(seen)
        decision = [complete]
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        return decision[0]

    def _pending_group_ids(self, step):
        pending = None
        if self.trainer.accelerator.is_main_process:
            path = self.monitor.run_dir / "probes.jsonl"
            seen = set()
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("step") == step and row.get("probe_suite") == self.probe_suite:
                        seen.add(row.get("group_id"))
            pending = [gid for gid in self.group_ids if gid not in seen]
        decision = [pending]
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        return decision[0]

    def _set_seed(self):
        rank = self.trainer.accelerator.process_index
        random.seed(self.seed + rank)
        torch.manual_seed(self.seed + rank)
        torch.cuda.manual_seed(self.seed + rank)

    @staticmethod
    def _gather(value):
        if not dist.is_initialized():
            return [value]
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, value)
        return gathered

    def _generate(self, prompts):
        _, completion_ids, _, _ = self.trainer._generate_single_turn(prompts)
        return completion_ids

    def _think(self):
        rank = self.trainer.accelerator.process_index
        gid = self.group_ids[self._group_offset + rank]
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
        target_domain = row["target_domain"]
        rewards = self.beam32_fn(
            prompts, texts, completion_ids, [gold] * 4, [target_domain] * 4,
            recommendation_group_ids=[gid] * 4,
        )
        torch.cuda.synchronize()
        total_wall = time.perf_counter() - started
        beam_call = self.trainer._current_beam_call() or {}
        results = beam_call.get("local_results", [{} for _ in completion_ids])
        close_ids = self.trainer.processing_class.encode("</think>", add_special_tokens=False)
        if len(close_ids) != 1:
            raise RuntimeError(f"expected one </think> token, got {close_ids}")
        close_id = close_ids[0]
        candidates = []
        for ids, text, reward, result in zip(completion_ids, texts, rewards, results):
            candidates.append({
                "completion": text,
                "completion_length": len(ids),
                "completion_sha256": _completion_sha256(ids),
                "closed": close_id in ids,
                "reward": reward,
                "exact": result.get("exact"),
                "ab": result.get("ab"),
                "a": result.get("a"),
                "invalid": result.get("invalid"),
                "beam_sids": result.get("beam_sids"),
                "beam_fixed_domain_prefix": result.get("beam_fixed_domain_prefix"),
                "target_domain": result.get("target_domain"),
                "domain_prefix": result.get("domain_prefix"),
                "beam_search_space": result.get("beam_search_space"),
            })
        return {
            "group_id": gid,
            "prompt": row["prompt"],
            "gold_sids": row["all_gold_sids"],
            "target_domain": row.get("target_domain"),
            "candidates": candidates,
            "generation_wall_sec": generation_wall,
            "wall_sec": total_wall,
        }

    def _nothink_batch(self, batch_index):
        rank = self.trainer.accelerator.process_index
        group_index = self._group_offset + batch_index * 2 + rank // 2
        gid = self.group_ids[group_index]
        row = self.records[gid]["no_think"]
        prompts = [row["prompt"]] * 4
        self.trainer._set_route_config("no_think")
        torch.cuda.synchronize()
        started = time.perf_counter()
        completion_ids = self._generate(prompts)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        gold = _gold_set(row["all_gold_sids"])
        candidates = []
        for ids in completion_ids:
            text = self.trainer.processing_class.decode(ids, skip_special_tokens=False)
            sid = final_sid(text)
            candidates.append({
                "completion": text,
                "completion_length": len(ids),
                "completion_sha256": _completion_sha256(ids),
                "parsed_sid": sid,
                "reward": q_reward(sid, gold),
            })
        return {
            "group_id": gid,
            "prompt": row["prompt"],
            "gold_sids": row["all_gold_sids"],
            "target_domain": row.get("target_domain"),
            "candidates": candidates,
            "wall_sec": wall,
        }

    @staticmethod
    def _summary(candidates, *, think):
        rewards = [float(item["reward"]) for item in candidates]
        result = {
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards),
            "candidates": candidates,
        }
        if think:
            result.update({
                "closure_rate": sum(bool(item["closed"]) for item in candidates) / len(candidates),
                "exact_count": sum(int(item.get("exact") or 0) for item in candidates),
                "ab_count": sum(int(item.get("ab") or 0) for item in candidates),
                "a_count": sum(int(item.get("a") or 0) for item in candidates),
                "invalid_count": sum(int(item.get("invalid") or 0) for item in candidates),
            })
        return result

    def evaluate(self, step, reason):
        if self.last_step == step:
            self.last_step = step
            return
        pending_group_ids = self._pending_group_ids(step)
        if not pending_group_ids:
            self.last_step = step
            return
        evaluation_group_ids = (
            pending_group_ids if len(pending_group_ids) % len(PROBE_DOMAINS) == 0
            else self.group_ids
        )
        if self.trainer.accelerator.num_processes != 4:
            raise RuntimeError("fixed probe evaluation requires the production 4-rank shape")
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
        configured_group_ids = self.group_ids
        configured_group_offset = self._group_offset
        started = time.perf_counter()
        try:
            self.group_ids = list(evaluation_group_ids)
            self._set_seed()
            think_by_rank = []
            nothink_parts = []
            for group_offset in range(0, len(self.group_ids), len(PROBE_DOMAINS)):
                self._group_offset = group_offset
                think_by_rank.extend(self._gather(self._think()))
                for batch_index in range(2):
                    nothink_parts.extend(self._gather(self._nothink_batch(batch_index)))
            torch.cuda.synchronize()
            local_wall = time.perf_counter() - started
            rank_walls = self._gather(local_wall)
            if self.trainer.accelerator.is_main_process:
                no_by_group = defaultdict(list)
                no_meta = {}
                for part in nothink_parts:
                    no_by_group[part["group_id"]].extend(part["candidates"])
                    no_meta[part["group_id"]] = part
                for think_part in think_by_rank:
                    gid = think_part["group_id"]
                    no_part = no_meta[gid]
                    self.monitor.write_probe({
                        "probe_suite": self.probe_suite,
                        "step": int(step),
                        "reason": reason,
                        "group_id": gid,
                        "target_domain": think_part.get("target_domain"),
                        "gold_sids": think_part["gold_sids"],
                        "think_prompt": think_part["prompt"],
                        "nothink_prompt": no_part["prompt"],
                        "think": self._summary(think_part["candidates"], think=True),
                        "nothink": self._summary(no_by_group[gid], think=False),
                        "think_generation_wall_sec": think_part["generation_wall_sec"],
                        "think_wall_sec": think_part["wall_sec"],
                        "nothink_rank_wall_sec": no_part["wall_sec"],
                        "probe_wall_sec": max(rank_walls),
                        "seed": self.seed,
                    })
        finally:
            self.group_ids = configured_group_ids
            self._group_offset = configured_group_offset
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


class FixedProbeCallback(TrainerCallback):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def on_train_begin(self, args, state, control, **kwargs):
        self.evaluator.evaluate(int(state.global_step), "baseline")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if probe_due(step, self.evaluator.every_steps):
            self.evaluator.evaluate(step, "interval")

    def on_train_end(self, args, state, control, **kwargs):
        self.evaluator.evaluate(int(state.global_step), "final")
