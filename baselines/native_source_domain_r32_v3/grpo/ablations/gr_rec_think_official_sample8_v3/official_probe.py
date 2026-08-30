"""Production Official Beam32 Probe4 schema for V3-Official."""
from __future__ import annotations

import statistics

from grpo_probe import FixedProbeEvaluator


class OfficialProbeEvaluator(FixedProbeEvaluator):
    """Keep generation/scoring unchanged and expose an explicit Official schema."""

    @staticmethod
    def _summary(candidates, *, think):
        summary = FixedProbeEvaluator._summary(candidates, think=think)
        summary["candidate_count"] = len(candidates)
        if think:
            summary.update({
                "reward_semantics": (
                    "production hierarchical reward per sampled CoT over fixed-domain "
                    "Official Beam32 ABC3 candidates"
                ),
                "reward_denominator": f"{len(candidates)} sampled CoTs",
                "cot_length_mean": (
                    statistics.fmean(item["completion_length"] for item in candidates)
                    if candidates else 0.0
                ),
                "official_fixed_domain": True,
                "beam_width": 32,
                "answer_action": "ABC3",
            })
        return summary
