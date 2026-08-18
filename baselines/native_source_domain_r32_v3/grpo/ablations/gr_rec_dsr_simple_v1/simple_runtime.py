# -*- coding: utf-8 -*-
"""DSR-Simple runtime adapters; frozen primary rewards remain unchanged."""
from __future__ import annotations

from gr_rec_dsr_v1.dsr_objectives import cot_score, exploration_score, prefix_support
from gr_rec_dsr_v1.dsr_parser import parse_interest_section
from gr_rec_dsr_v1.dsr_runtime import (
    get_capture,
    make_dsr_beam32_fn,
    make_dsr_nothink_reward_func,
    reset_global_capture,
)

from .simple_objectives import beam_a_diversity, interest_count_score


def _gold_set(values):
    from grpo_sid import parse_sid

    return {parsed for value in values if (parsed := parse_sid(value)) is not None}


def make_simple_beam32_fn(baseline_factory, model, tokenizer, monitor_writer=None):
    return make_dsr_beam32_fn(baseline_factory, model, tokenizer, monitor_writer)


def make_simple_think_reward_func(beam32_fn=None):
    """Capture Simple training fields and diagnostic-only legacy fields."""
    def reward_func(prompts, completions, completion_ids, **kwargs):
        routes = kwargs.get("route")
        think_idx = [index for index, route in enumerate(routes or ()) if route == "think"]
        if not think_idx:
            return [None] * len(prompts)
        if beam32_fn is None:
            raise RuntimeError("DSR-Simple Think reward requires Beam32")
        gold_sets = [_gold_set(values) for values in kwargs["all_gold_sids"]]
        sub_prompts = [prompts[index] for index in think_idx]
        sub_ids = [completion_ids[index] for index in think_idx]
        sub_golds = [gold_sets[index] for index in think_idx]
        primary = beam32_fn(
            sub_prompts,
            [completions[index] for index in think_idx],
            sub_ids,
            sub_golds,
        )
        capture = get_capture()
        results = capture.pending_beam_results
        if results is None or len(results) != len(primary):
            raise RuntimeError("DSR-Simple Think did not receive aligned Beam results")
        target_domains = kwargs.get("target_domain", [None] * len(prompts))
        for local_index, source_index in enumerate(think_idx):
            cot = capture.tokenizer.decode(sub_ids[local_index], skip_special_tokens=False)
            parsed = parse_interest_section(cot, sub_prompts[local_index])
            raw_interest_n = int(parsed.bullet_count)
            beam_sids = results[local_index]["beam_sids"]
            target_domain = target_domains[source_index]
            simple = beam_a_diversity(beam_sids, target_domain)

            # These legacy computations are persisted for observation only.
            diagnostic_cot = cot_score(parsed)
            diagnostic_prefix = prefix_support(beam_sids, sub_golds[local_index])
            diagnostic_explore = exploration_score(beam_sids, target_domain)
            capture.records.append({
                "route": "think",
                "group_id": kwargs["recommendation_group_id"][source_index],
                "target_domain": target_domain,
                "gold_count": len(sub_golds[local_index]),
                "unique_gold_a": len({sid[1] for sid in sub_golds[local_index]}),
                "primary_reward": float(primary[local_index]),
                "cot": cot,
                "completion_length": len(sub_ids[local_index]),
                "closed": "</think>" in cot,
                "parsed": parsed.to_dict(),
                "raw_interest_n": raw_interest_n,
                "s_n": interest_count_score(raw_interest_n),
                **simple,
                "beam_invalid": int(results[local_index].get("invalid", 0)),
                "diagnostic_only": {
                    "grounded_n": int(parsed.grounded_count),
                    "grounding_coverage": (
                        parsed.grounded_count / raw_interest_n if raw_interest_n > 0 else None
                    ),
                    "d_cot": float(diagnostic_cot["evidence_diversity"]),
                    "s_cot": float(diagnostic_cot["s_cot"]),
                    "s_prefix": float(diagnostic_prefix["s_prefix"]),
                    "entropy_s_explore": float(diagnostic_explore["s_explore"]),
                    "fake_or_ungrounded_sid_count": sum(
                        len(item.evidence) - len(item.grounded_evidence)
                        for item in parsed.bullets
                    ),
                    "exact": int(results[local_index].get("exact", 0)),
                    "ab": int(results[local_index].get("ab", 0)),
                    "a": int(results[local_index].get("a", 0)),
                },
            })
        capture.pending_beam_results = None
        output = [None] * len(prompts)
        for index, reward in zip(think_idx, primary):
            output[index] = reward
        return output

    reward_func.__name__ = "think_reward"
    return reward_func


def make_simple_nothink_reward_func(tokenizer=None):
    """Call the old DSR reward unchanged, then attach input metadata only."""
    baseline = make_dsr_nothink_reward_func(tokenizer=tokenizer)

    def reward_func(prompts, completions, completion_ids, **kwargs):
        capture = get_capture()
        before = len(capture.records)
        output = baseline(prompts, completions, completion_ids, **kwargs)
        indexes = [
            index for index, route in enumerate(kwargs.get("route") or ())
            if route == "no_think"
        ]
        added = capture.records[before:]
        if len(added) != len(indexes):
            raise RuntimeError("DSR-Simple NoThink metadata capture is not aligned")
        domains = kwargs.get("target_domain", [None] * len(prompts))
        for item, index in zip(added, indexes):
            item["target_domain"] = domains[index]
            item["gold_count"] = len(kwargs["all_gold_sids"][index])
            item["unique_gold_a"] = len(item.get("gold_as", ()))
        return output

    reward_func.__name__ = "nothink_reward"
    return reward_func


__all__ = [
    "get_capture",
    "make_dsr_nothink_reward_func",
    "make_simple_nothink_reward_func",
    "make_simple_beam32_fn",
    "make_simple_think_reward_func",
    "reset_global_capture",
]
