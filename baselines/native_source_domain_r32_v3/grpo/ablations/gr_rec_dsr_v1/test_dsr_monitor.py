"""CPU-only tests for DSR monitor aggregation and probe enrichment."""
from __future__ import annotations

import math

from .dsr_monitor import summarize_nothink_records, summarize_think_records
from .dsr_probe import DsrProbeMonitorProxy, enrich_probe_event, mean_completion_similarity


def _think_record(group_id, reward, s_cot, s_explore=0.5, raw_count=4, grounded_count=3):
    return {
        "group_id": group_id,
        "primary_reward": reward,
        "parsed": {"parser_success": True},
        "bullet_count": raw_count,
        "grounded_count": grounded_count,
        "s_cot": s_cot,
        "evidence_diversity": 0.8,
        "s_prefix": 0.2,
        "s_explore": s_explore,
        "s_dead": s_cot * s_explore,
        "unique_a": 5,
        "a_entropy_norm": 0.7,
        "s_a": 0.3,
        "s_ab": 0.1,
        "has_exact": reward == 8.0,
    }


def _nothink_record(group_id, reward, predicted_a, gold_as, position=3):
    return {
        "group_id": group_id,
        "primary_reward": reward,
        "predicted_a": predicted_a,
        "sa_position": position if predicted_a is not None else -1,
        "gold_as": gold_as,
    }


def test_think_signal_rescue_and_branch_distribution():
    records = [
        _think_record("dead", 0.0, score)
        for score in (1.0, 0.8, 0.6, 0.4)
    ] + [
        _think_record("normal", reward, score)
        for reward, score in zip((0.0, 0.5, 2.0, 8.0), (0.3, 0.4, 0.5, 0.6))
    ]
    summary, scores, advantages = summarize_think_records(records)
    assert summary["group_count"] == 2
    assert summary["primary_zero_std_rate"] == 0.5
    assert summary["signal_rescue_rate"] == 0.5
    assert summary["aux_zero_std_rate"] == 0.0
    assert summary["branches"]["dead_zero"] == 1
    assert summary["branches"]["primary_variance"] == 1
    assert set(summary["branches"]) == {
        "primary_variance", "dead_zero", "prefix_rescue", "cot_only_saturated_or_other"
    }
    assert len(scores) == len(advantages) == 8


def test_think_raw_grounded_coverage_excludes_undefined_candidates():
    records = [
        _think_record("coverage", 0.0, 0.8, raw_count=raw, grounded_count=grounded)
        for raw, grounded in ((4, 3), (1, 1), (4, 0), (0, 0))
    ]
    summary, _, _ = summarize_think_records(records)
    assert summary["raw_interest_count_mean"] == 2.25
    assert summary["raw_interest_count_distribution"] == {
        "0": 1, "1": 1, "2": 0, "3": 0, "4": 2, "5+": 0
    }
    assert summary["grounded_interest_count_mean"] == 1.0
    assert summary["raw_grounded_gap_mean"] == 1.25
    assert math.isclose(summary["grounding_coverage_mean"], (0.75 + 1.0 + 0.0) / 3)
    assert summary["grounding_coverage_defined_rate"] == 0.75


def test_nothink_complete_monitor_statistics():
    all_zero = [
        _nothink_record("zero", 0.0, value, [99])
        for value in (1, 1, 1, 2, 3, 4, 5, 6)
    ]
    signaled_rewards = (-1.0, -0.25, 0.0, 0.5, 2.0, 8.0, 0.0, 0.5)
    signaled = [
        _nothink_record("signal", reward, 10 + index, [10, 11, 12])
        for index, reward in enumerate(signaled_rewards)
    ]
    summary, plans = summarize_nothink_records(all_zero + signaled)
    assert summary["group_count"] == 2
    assert summary["primary_zero_std_rate"] == 0.5
    assert summary["all_zero_group_rate"] == 0.5
    assert summary["rescue_active_group_rate"] == 0.5
    assert summary["group_a_concentration_max"] == 0.375
    assert summary["any_gold_a_hit_rate"] == 0.5
    assert summary["any_gold_ab_hit_rate"] == 0.5
    assert summary["any_exact_hit_rate"] == 0.5
    assert summary["candidate_reward_distribution"] == {
        "-1": 1, "-0.25": 1, "0": 10, "0.5": 2, "2": 1, "8": 1
    }
    assert summary["valid_sid_rate"] == 15 / 16
    assert summary["wrong_domain_rate"] == 1 / 16
    assert summary["gold_a_strata"]["sparse"]["group_count"] == 1
    assert summary["gold_a_strata"]["dense"]["group_count"] == 1
    assert plans[0].active and not plans[1].active


def _probe_event():
    prompt = "History <|prod_begin|><s_a_1><s_b_2><s_c_3>"
    think_candidates = []
    for index in range(4):
        sid = "<|prod_begin|><s_a_1><s_b_2><s_c_3>"
        think_candidates.append({
            "completion": f"<think>\n【兴趣归纳】\n1. Item {index}: {sid}\n2. Other: {sid}</think>",
            "completion_length": 20 + index,
            "closed": True,
            "reward": float(index),
            "exact": int(index == 3),
            "ab": 0,
            "a": 1,
            "invalid": 0,
            "beam_sids": [["prod", 1, 2, 3]] + [["prod", 7 + index, 8, 9]] * 31,
        })
    no_candidates = [{
        "completion": f"<|prod_begin|><s_a_{7 if index < 3 else 8 + index}><s_b_9><s_c_10>",
        "parsed_sid": ["prod", 7 if index < 3 else 8 + index, 9, 10],
        "reward": 0.0,
    } for index in range(8)]
    return {
        "step": 0,
        "group_id": "probe-prod",
        "target_domain": "prod",
        "gold_sids": [["prod", 1, 2, 3]],
        "think_prompt": prompt,
        "think": {"candidates": think_candidates},
        "nothink": {"candidates": no_candidates},
    }


def test_probe_enrichment_is_cpu_only_and_complete():
    event = enrich_probe_event(_probe_event())
    think = event["dsr"]["think"]
    no_think = event["dsr"]["nothink"]
    assert len(think["candidates"]) == 4
    assert think["branch"] == "primary_variance"
    assert all("s_aux" in candidate and "a_aux" in candidate for candidate in think["candidates"])
    assert all(candidate["raw_interest_n"] == 2 for candidate in think["candidates"])
    assert all(candidate["grounded_n"] == 2 for candidate in think["candidates"])
    assert all(candidate["grounding_coverage"] == 1.0 for candidate in think["candidates"])
    assert think["raw_n_mean"] == think["grounded_n_mean"] == 2.0
    assert think["raw_grounded_gap_mean"] == 0.0
    assert think["grounding_coverage_mean"] == 1.0
    assert no_think["would_rescue"] is True
    assert no_think["concentration"] == 0.375
    assert math.isclose(no_think["coefficient"], 0.0375)
    assert len(no_think["candidates"]) == 8
    assert mean_completion_similarity(["abcdef", "abcdef", "uvwxyz"]) < 1.0


def test_probe_zero_raw_interest_has_null_coverage_and_does_not_pollute_mean():
    event = _probe_event()
    event["think"]["candidates"][0]["completion"] = "<think>No interest list.</think>"
    think = enrich_probe_event(event)["dsr"]["think"]
    assert think["candidates"][0]["raw_interest_n"] == 0
    assert think["candidates"][0]["grounding_coverage"] is None
    assert think["grounding_coverage_mean"] == 1.0


def test_probe_monitor_proxy_writes_one_enriched_event():
    class Monitor:
        def __init__(self):
            self.events = []

        def write_probe(self, event):
            self.events.append(event)
            return True

    monitor = Monitor()
    proxy = DsrProbeMonitorProxy(monitor)
    assert proxy.write_probe(_probe_event())
    assert len(monitor.events) == 1
    assert "dsr" in monitor.events[0]
