import math

from .interest_metric import InterestScore, beam_utility, composite_reward, monitor_record, population_advantages


def test_display_fields_do_not_change_reward_or_advantage_math():
    score = InterestScore(
        parser_success=True, gold_parser_success=True, n_gold=2, n_pred=2,
        matched_interest_count=1, interest_coverage=0.5, interest_precision=0.5,
        mean_match_similarity=0.8, coverage_tier=0.5, match_quality=0.2,
        cot_utility=0.7, matches=(),
    )
    before_reward = composite_reward(4.0, score.cot_utility)
    before_advantages = population_advantages([before_reward, 0.1, 0.2, 0.3])
    record = monitor_record(4.0, score, raw_n=2, grounded_n=1)
    after_advantages = population_advantages([record['composite_reward'], 0.1, 0.2, 0.3])
    assert record['beam_utility'] == beam_utility(4.0)
    assert record['cot_utility'] == score.cot_utility
    assert record['composite_reward'] == before_reward
    assert before_advantages == after_advantages
    assert math.isclose(record['beam_contribution'] + record['cot_contribution'], before_reward,
                        abs_tol=1e-12)
