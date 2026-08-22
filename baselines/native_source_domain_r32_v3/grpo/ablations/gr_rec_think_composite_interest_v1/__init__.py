"""Think-only CompositeInterest v1 CPU primitives and provenance gate."""

from .interest_metric import (
    MATCH_THRESHOLD,
    MATCH_QUALITY_FLOOR,
    beam_utility,
    composite_reward,
    population_advantages,
    score_interest_cot,
)

__all__ = [
    "MATCH_THRESHOLD",
    "MATCH_QUALITY_FLOOR",
    "beam_utility",
    "composite_reward",
    "population_advantages",
    "score_interest_cot",
]
