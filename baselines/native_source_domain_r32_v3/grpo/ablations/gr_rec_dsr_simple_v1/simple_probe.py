"""Fixed-probe enrichment for DSR-Simple; generation remains baseline-owned."""
from __future__ import annotations

import statistics

from grpo_probe import FixedProbeEvaluator
from grpo_sid import parse_sid
from gr_rec_dsr_v1.dsr_probe import enrich_probe_event as enrich_dsr_probe_event

from .simple_objectives import (
    beam_a_diversity,
    choose_simple_think_aux_scores,
    interest_count_score,
    simple_group_advantages,
)


def _sid(value):
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return (str(value[0]), int(value[1]), int(value[2]), int(value[3]))
    return parse_sid(value) if isinstance(value, str) else None


def enrich_simple_probe_event(event: dict) -> dict:
    enriched = enrich_dsr_probe_event(event)
    think = event.get("think") or {}
    old_think = enriched.get("dsr", {}).get("think", {})
    old_candidates = old_think.get("candidates", [])
    score_inputs = []
    simple_candidates = []
    for candidate, diagnostic in zip(think.get("candidates", ()), old_candidates):
        raw_n = int(diagnostic["raw_interest_n"])
        diversity = beam_a_diversity(
            [_sid(value) for value in candidate.get("beam_sids") or ()],
            event.get("target_domain") or "",
        )
        score_inputs.append({
            "primary_reward": float(candidate.get("reward") or 0.0),
            "s_n": interest_count_score(raw_n),
            **diversity,
        })
        simple_candidates.append({
            "raw_interest_n": raw_n,
            "s_n": interest_count_score(raw_n),
            **diversity,
            "diagnostic_only": {**diagnostic, "diagnostic_only": True},
        })
    scores, branch = choose_simple_think_aux_scores(score_inputs) if score_inputs else ([], "primary_only")
    advantages = simple_group_advantages(scores).tolist() if scores else []
    for candidate, score, advantage in zip(simple_candidates, scores, advantages):
        candidate.update({
            "simple_s_aux": score,
            "simple_a_aux": advantage,
            "simple_branch": branch,
        })
    rewards = [float(item.get("reward") or 0.0) for item in think.get("candidates", ())]
    enriched.setdefault("dsr", {}).setdefault("think", {})["diagnostic_only"] = True
    enriched["simple_dsr"] = {
        "think": {
            "training_signal": True,
            "raw_interest_n_mean": statistics.fmean(
                item["raw_interest_n"] for item in simple_candidates
            ) if simple_candidates else 0.0,
            "s_n_mean": statistics.fmean(item["s_n"] for item in simple_candidates) if simple_candidates else 0.0,
            "unique_valid_target_a_mean": statistics.fmean(
                item["unique_valid_target_a"] for item in simple_candidates
            ) if simple_candidates else 0.0,
            "d_a_mean": statistics.fmean(item["d_a"] for item in simple_candidates) if simple_candidates else 0.0,
            "simple_branch": branch,
            "simple_aux_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "primary_zero_std": statistics.pstdev(rewards) == 0.0 if len(rewards) > 1 else True,
            "candidates": simple_candidates,
        },
        "nothink": {
            **enriched.get("dsr", {}).get("nothink", {}),
            "training_signal": True,
            "implementation": "gr_rec_dsr_v1 (reused)",
        },
    }
    return enriched


class SimpleProbeMonitorProxy:
    def __init__(self, monitor):
        self._monitor = monitor

    def __getattr__(self, name):
        return getattr(self._monitor, name)

    def write_probe(self, event):
        return self._monitor.write_probe(enrich_simple_probe_event(event))


class SimpleFixedProbeEvaluator(FixedProbeEvaluator):
    def __init__(self, *args, monitor, **kwargs):
        self.simple_monitor_proxy = SimpleProbeMonitorProxy(monitor)
        super().__init__(*args, monitor=self.simple_monitor_proxy, **kwargs)
