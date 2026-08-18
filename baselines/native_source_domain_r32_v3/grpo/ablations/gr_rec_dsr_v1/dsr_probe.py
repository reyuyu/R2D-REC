"""Pure CPU enrichment for baseline FixedProbeEvaluator events."""
from __future__ import annotations

import statistics
from collections import Counter
from itertools import combinations

from grpo_probe import FixedProbeEvaluator
from grpo_sid import parse_sid

from .dsr_objectives import (
    build_nothink_rescue_plan,
    choose_think_aux_scores,
    cot_score,
    exploration_score,
    group_aux_advantages,
    prefix_support,
)
from .dsr_parser import parse_interest_section


def _sid(value):
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return (str(value[0]), int(value[1]), int(value[2]), int(value[3]))
    return parse_sid(value) if isinstance(value, str) else None


def _char_ngrams(text: str, size: int = 4) -> set[str]:
    compact = "".join(str(text).split())
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def mean_completion_similarity(completions) -> float:
    sets = [_char_ngrams(value) for value in completions]
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return statistics.fmean(values) if values else 1.0


def enrich_probe_event(event: dict) -> dict:
    """Return a copy of an existing probe event with monitor-only DSR fields."""
    enriched = dict(event)
    golds = {sid for value in event.get("gold_sids", ()) if (sid := _sid(value)) is not None}
    gold_as = {sid[1] for sid in golds}
    think = event.get("think") or {}
    think_candidates = []
    score_inputs = []
    for candidate in think.get("candidates", ()):
        completion = candidate.get("completion") or ""
        parsed = parse_interest_section(completion, event.get("think_prompt") or "")
        cot = cot_score(parsed)
        beam_sids = [_sid(value) for value in candidate.get("beam_sids") or ()]
        prefix = prefix_support(beam_sids, golds)
        explore = exploration_score(beam_sids, event.get("target_domain") or "")
        s_dead = float(cot["s_cot"]) * float(explore["s_explore"])
        score_inputs.append({
            "primary_reward": float(candidate.get("reward") or 0.0),
            "has_exact": bool(candidate.get("exact")),
            **cot,
            **prefix,
            **explore,
            "s_dead": s_dead,
        })
        parsed_data = parsed.to_dict()
        grounded = []
        ungrounded = []
        for bullet in parsed_data["bullets"]:
            grounded.append({
                "title": bullet["title"],
                "verified_sids": bullet["grounded_evidence"],
            })
            verified = set(bullet["grounded_evidence"])
            ungrounded.extend(value for value in bullet["evidence"] if value not in verified)
        think_candidates.append({
            "raw_interest_n": parsed_data["bullet_count"],
            "grounded_n": parsed_data["grounded_count"],
            "parser_success": parsed_data["parser_success"],
            "grounded_interests": grounded,
            "ungrounded_or_fake_sids": ungrounded,
            "d_cot": cot["evidence_diversity"],
            "s_cot": cot["s_cot"],
            "s_prefix": prefix["s_prefix"],
            "s_explore": explore["s_explore"],
            "s_dead": s_dead,
        })
    scores, branch = choose_think_aux_scores(score_inputs)
    advantages = group_aux_advantages(scores, 4).tolist()
    for details, score, advantage in zip(think_candidates, scores, advantages):
        details.update({"s_aux": score, "a_aux": advantage})

    think_rewards = [float(item.get("reward") or 0.0) for item in think.get("candidates", ())]
    think_dsr = {
        "parser_success_rate": sum(item["parser_success"] for item in think_candidates) / len(think_candidates) if think_candidates else 0.0,
        "grounded_n_mean": statistics.fmean(item["grounded_n"] for item in think_candidates) if think_candidates else 0.0,
        "cot_group_similarity": mean_completion_similarity(
            item.get("completion") or "" for item in think.get("candidates", ())
        ),
        "s_cot_mean": statistics.fmean(item["s_cot"] for item in think_candidates) if think_candidates else 0.0,
        "s_prefix_mean": statistics.fmean(item["s_prefix"] for item in think_candidates) if think_candidates else 0.0,
        "s_explore_mean": statistics.fmean(item["s_explore"] for item in think_candidates) if think_candidates else 0.0,
        "s_dead_mean": statistics.fmean(item["s_dead"] for item in think_candidates) if think_candidates else 0.0,
        "branch": branch,
        "aux_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "primary_zero_std": statistics.pstdev(think_rewards) == 0.0 if len(think_rewards) > 1 else True,
        "candidates": think_candidates,
    }

    no_think = event.get("nothink") or {}
    no_candidates = no_think.get("candidates") or []
    rewards = [float(item.get("reward") or 0.0) for item in no_candidates]
    predicted = [(_sid(item.get("parsed_sid")) or (None, None, None, None))[1] for item in no_candidates]
    frequencies = Counter(value for value in predicted if value is not None)
    positions = [0 if value is not None else -1 for value in predicted]
    plan = build_nothink_rescue_plan(rewards, predicted, positions, gold_as)
    candidate_details = []
    for candidate, sid, value, weight in zip(no_candidates, map(lambda item: _sid(item.get("parsed_sid")), no_candidates), predicted, plan.frequency_weights):
        candidate_details.append({
            "predicted_a": value,
            "a_frequency": frequencies.get(value, 0),
            "frequency_weight": weight,
            "gold_a": bool(sid and sid[:2] in {gold[:2] for gold in golds}),
            "gold_ab": bool(sid and sid[:3] in {gold[:3] for gold in golds}),
            "exact": bool(sid and sid in golds),
            "rescue_target": bool(plan.active and value is not None),
        })
    no_dsr = {
        "primary_zero_std": statistics.pstdev(rewards) == 0.0 if len(rewards) > 1 else True,
        "all_zero": all(value == 0.0 for value in rewards),
        "predicted_as": predicted,
        "a_frequencies": {str(key): value for key, value in sorted(frequencies.items())},
        "concentration": plan.concentration,
        "gold_unique_a": plan.gold_unique_a,
        "would_rescue": plan.active,
        "lambda_a": plan.lambda_a,
        "coefficient": plan.coefficient,
        "any_gold_a": any(value >= 0.5 for value in rewards),
        "any_gold_ab": any(value >= 2.0 for value in rewards),
        "any_exact": any(value == 8.0 for value in rewards),
        "candidates": candidate_details,
    }
    enriched["dsr"] = {"think": think_dsr, "nothink": no_dsr}
    return enriched


class DsrProbeMonitorProxy:
    """Intercept baseline probe writes and add CPU diagnostics only."""

    def __init__(self, monitor):
        self._monitor = monitor

    def __getattr__(self, name):
        return getattr(self._monitor, name)

    def write_probe(self, event):
        return self._monitor.write_probe(enrich_probe_event(event))


class DsrFixedProbeEvaluator(FixedProbeEvaluator):
    """Baseline evaluator with an enriched writer; generation is unchanged."""

    def __init__(self, *args, monitor, **kwargs):
        self.dsr_monitor_proxy = DsrProbeMonitorProxy(monitor)
        super().__init__(*args, monitor=self.dsr_monitor_proxy, **kwargs)
