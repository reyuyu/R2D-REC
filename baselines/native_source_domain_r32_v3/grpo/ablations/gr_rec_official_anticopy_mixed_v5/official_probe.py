"""All-domain production Official Beam32 probe with copy anatomy."""
from __future__ import annotations

import statistics

from grpo_probe import FixedProbeEvaluator, _gold_set
from grpo_sid import parse_sid, q_reward

from .official_anticopy_mixed_trainer import extract_history_sids, reward_level


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
        details.append({
            "candidate_sid": sid,
            "raw_q_reward": raw,
            "reward_level": reward_level(raw),
            "is_history_copy": sid is not None and sid in history,
        })
    return details


def official_probe_summary(candidates):
    rewards = [float(item["reward"]) for item in candidates]
    beam_details = [
        detail for item in candidates for detail in item.get("beam_candidate_details", [])
    ]
    copied = [item for item in beam_details if item["is_history_copy"]]
    noncopied = [item for item in beam_details if not item["is_history_copy"]]
    first = candidates[0] if candidates else {}
    return {
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
        "reward_semantics": "production hierarchical Official Beam32 reward per sampled CoT; no training copy discount",
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
        "history_sid_count": int(first.get("history_sid_count") or 0),
        "gold_history_exact_overlap": int(first.get("gold_history_exact_overlap") or 0),
        "gold_history_exact_sids": first.get("gold_history_exact_sids") or [],
        "candidates": candidates,
    }


class MixedOfficialProbeEvaluator(FixedProbeEvaluator):
    """Enrich the unchanged fixed Probe4 with domain-aware copy diagnostics."""

    def _think(self):
        part = super()._think()
        target_domain = part["target_domain"]
        history = extract_history_sids(part["prompt"], target_domain)
        gold = _gold_set(part["gold_sids"])
        overlap = sorted(set(gold) & history)
        for candidate in part["candidates"]:
            candidate["history_sid_count"] = len(history)
            candidate["gold_history_exact_overlap"] = len(set(gold) & history)
            candidate["gold_history_exact_sids"] = [list(sid) for sid in overlap]
            candidate["beam_candidate_details"] = beam_copy_details(
                candidate.get("beam_sids"), gold, history
            )
        return part

    @staticmethod
    def _summary(candidates, *, think):
        if think:
            return official_probe_summary(candidates)
        return FixedProbeEvaluator._summary(candidates, think=False)
