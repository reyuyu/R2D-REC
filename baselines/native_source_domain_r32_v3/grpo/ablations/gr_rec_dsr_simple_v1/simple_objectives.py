"""Pure DSR-Simple Think objective math; no model or tokenizer calls."""
from __future__ import annotations

import statistics
from typing import Sequence

from gr_rec_dsr_v1.dsr_objectives import group_aux_advantages
from gr_rec_dsr_v1.dsr_parser import Sid


SIMPLE_BRANCHES = (
    "primary_only",
    "zero_std_count_rescue",
    "dead_zero_count_diversity_rescue",
)


def interest_count_score(raw_interest_n: int) -> float:
    """Binary anti-collapse score; 2, 3, and 4 interests are equivalent."""
    return 1.0 if 2 <= int(raw_interest_n) <= 4 else 0.0


def beam_a_diversity(
    beam_sids: Sequence[Sid | None], target_domain: str
) -> dict[str, float | int]:
    """Count unique valid target-domain A values in existing Beam results."""
    valid_target_as = {
        int(sid[1])
        for sid in beam_sids
        if sid is not None and sid[0] == target_domain
    }
    unique_a = len(valid_target_as)
    return {
        "unique_valid_target_a": unique_a,
        "d_a": min(unique_a / 8.0, 1.0),
    }


def choose_simple_think_aux_scores(
    candidates: Sequence[dict],
) -> tuple[list[float], str]:
    """Apply the three mutually-exclusive Simple Think branches."""
    if len(candidates) != 4:
        raise ValueError("DSR-Simple Think requires one complete G=4 group")
    rewards = [float(item["primary_reward"]) for item in candidates]
    primary_std = statistics.pstdev(rewards)
    if primary_std > 0.0:
        return [0.0] * 4, "primary_only"
    count_scores = [float(item["s_n"]) for item in candidates]
    if all(reward == 0.0 for reward in rewards):
        return [
            count_score + float(item["d_a"])
            for count_score, item in zip(count_scores, candidates)
        ], "dead_zero_count_diversity_rescue"
    return count_scores, "zero_std_count_rescue"


def simple_group_advantages(scores: Sequence[float]):
    """Population-normalized G=4 advantages with exact zero for ties."""
    return group_aux_advantages(scores, group_size=4)
