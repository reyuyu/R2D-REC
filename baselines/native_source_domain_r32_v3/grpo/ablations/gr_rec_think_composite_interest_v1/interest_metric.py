"""Deterministic CompositeInterest reward primitives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from ..gr_rec_think_exact_clamp_v1.think_diagnostics import (
    InterestParse,
    InterestUnit,
    extract_interest_units,
)

MATCH_THRESHOLD = 0.30
MATCH_QUALITY_FLOOR = 0.60


@dataclass(frozen=True)
class PairMatch:
    candidate_index: int
    gold_index: int
    similarity: float


@dataclass(frozen=True)
class InterestScore:
    parser_success: bool
    gold_parser_success: bool
    n_gold: int
    n_pred: int
    matched_interest_count: int
    interest_coverage: float
    interest_precision: float
    mean_match_similarity: float
    coverage_tier: float
    match_quality: float
    cot_utility: float
    matches: tuple[PairMatch, ...]


def _char_ngrams(text: str) -> Counter[str]:
    compact = "".join((text or "").split())
    if not compact:
        return Counter()
    if len(compact) == 1:
        return Counter((compact,))
    return Counter(compact[i:i + 2] for i in range(len(compact) - 1))


def multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def text_similarity(left: str, right: str) -> float:
    return multiset_f1(_char_ngrams(left), _char_ngrams(right))


def evidence_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    return 2.0 * overlap / (len(left_set) + len(right_set))


def pair_similarity(candidate: InterestUnit, gold: InterestUnit) -> float:
    text_score = text_similarity(candidate.normalized_text, gold.normalized_text)
    if not gold.grounded_evidence_sids:
        return text_score
    return 0.7 * text_score + 0.3 * evidence_similarity(
        candidate.grounded_evidence_sids,
        gold.grounded_evidence_sids,
    )


def maximum_weight_matching(
    candidates: Sequence[InterestUnit],
    gold: Sequence[InterestUnit],
    threshold: float = MATCH_THRESHOLD,
    similarity_fn=pair_similarity,
) -> tuple[PairMatch, ...]:
    """Maximum-cardinality thresholded one-to-one matching.

    Total similarity is the deterministic tie-break; interest lists are small.
    """
    if not candidates or not gold:
        return ()
    weights = [[float(similarity_fn(candidate, target)) for target in gold] for candidate in candidates]
    states: dict[int, tuple[float, tuple[PairMatch, ...]]] = {0: (0.0, ())}
    for candidate_index, row in enumerate(weights):
        updated = dict(states)
        for mask, (total, pairs) in states.items():
            for gold_index, weight in enumerate(row):
                if mask & (1 << gold_index) or weight < threshold:
                    continue
                next_mask = mask | (1 << gold_index)
                match = PairMatch(candidate_index, gold_index, weight)
                proposal = (total + weight, pairs + (match,))
                current = updated.get(next_mask)
                if current is None or proposal[0] > current[0] + 1e-15 or (
                    abs(proposal[0] - current[0]) <= 1e-15
                    and tuple((p.candidate_index, p.gold_index) for p in proposal[1])
                    < tuple((p.candidate_index, p.gold_index) for p in current[1])
                ):
                    updated[next_mask] = proposal
        states = updated
    return max(
        states.values(),
        key=lambda item: (
            len(item[1]),
            item[0],
            tuple((-p.candidate_index, -p.gold_index) for p in item[1]),
        ),
    )[1]


def coverage_tier(match_count: int, gold_count: int) -> float:
    if match_count <= 0 or gold_count <= 0:
        return 0.0
    coverage = match_count / gold_count
    if coverage <= 0.25:
        return 0.20
    if coverage <= 0.50:
        return 0.45
    if coverage < 1.0:
        return 0.70
    return 1.0


def score_parsed_interests(candidate: InterestParse, gold: InterestParse) -> InterestScore:
    matches = maximum_weight_matching(candidate.units, gold.units)
    n_gold, n_pred, matched = len(gold.units), len(candidate.units), len(matches)
    coverage = matched / n_gold if n_gold else 0.0
    precision = matched / n_pred if n_pred else 0.0
    mean_similarity = sum(pair.similarity for pair in matches) / matched if matched else 0.0
    quality = min(1.0, max(0.0, (mean_similarity - MATCH_QUALITY_FLOOR) / 0.40)) if matched else 0.0
    tier = coverage_tier(matched, n_gold)
    return InterestScore(
        parser_success=candidate.parser_success,
        gold_parser_success=gold.parser_success,
        n_gold=n_gold,
        n_pred=n_pred,
        matched_interest_count=matched,
        interest_coverage=coverage,
        interest_precision=precision,
        mean_match_similarity=mean_similarity,
        coverage_tier=tier,
        match_quality=quality,
        cot_utility=0.8 * tier + 0.2 * quality,
        matches=matches,
    )


def score_interest_cot(candidate_cot: str, gold_cot: str, prompt: str) -> InterestScore:
    return score_parsed_interests(
        extract_interest_units(candidate_cot, prompt),
        extract_interest_units(gold_cot, prompt),
    )


def beam_utility(raw_reward: float | int | None) -> float:
    try:
        value = float(raw_reward)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log(17.0))


def composite_reward(raw_beam_reward: float | int | None, cot_utility: float) -> float:
    cot = min(1.0, max(0.0, float(cot_utility)))
    return min(1.0, max(0.0, 0.60 * beam_utility(raw_beam_reward) + 0.40 * cot))


def population_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    if not rewards:
        return []
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0.0:
        return [0.0] * len(values)
    denominator = math.sqrt(variance) + epsilon
    return [(value - mean) / denominator for value in values]


def diversity_monitor(parsed_candidates: Sequence[InterestParse]) -> dict[str, float | int]:
    sets = [frozenset(unit.normalized_text for unit in parsed.units) for parsed in parsed_candidates]
    similarities = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left, right = sets[left_index], sets[right_index]
            similarities.append(len(left & right) / len(left | right) if left or right else 1.0)
    return {
        "pairwise_interest_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "unique_interest_set_count": len(set(sets)),
    }


def monitor_record(raw_beam_reward: float, score: InterestScore, raw_n: int, grounded_n: int) -> dict:
    total = composite_reward(raw_beam_reward, score.cot_utility)
    return {
        "beam_raw": raw_beam_reward,
        "beam_utility": beam_utility(raw_beam_reward),
        "gold_interest_count": score.n_gold,
        "pred_interest_count": score.n_pred,
        "matched_interest_count": score.matched_interest_count,
        "interest_coverage": score.interest_coverage,
        "interest_precision": score.interest_precision,
        "mean_match_similarity": score.mean_match_similarity,
        "coverage_tier": score.coverage_tier,
        "match_quality": score.match_quality,
        "cot_utility": score.cot_utility,
        "composite_reward": total,
        "raw_n": raw_n,
        "grounded_n": grounded_n,
        "grounding_coverage": grounded_n / raw_n if raw_n else None,
    }
