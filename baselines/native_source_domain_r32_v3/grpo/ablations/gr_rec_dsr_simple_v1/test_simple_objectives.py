"""CPU tests for the DSR-Simple mathematical contract."""
import math

import torch

from gr_rec_dsr_v1.dsr_objectives import choose_think_aux_scores

from .simple_objectives import (
    beam_a_diversity,
    choose_simple_think_aux_scores,
    interest_count_score,
    simple_group_advantages,
)


def _candidates(rewards, counts, unique_as=None):
    unique_as = unique_as or [0] * 4
    return [
        {
            "primary_reward": reward,
            "s_n": interest_count_score(count),
            "d_a": min(unique_a / 8.0, 1.0),
        }
        for reward, count, unique_a in zip(rewards, counts, unique_as)
    ]


def test_a_primary_variance_disables_aux_exactly():
    scores, branch = choose_simple_think_aux_scores(
        _candidates([8, 2, 0.5, 0], [4, 1, 3, 2], [8, 8, 8, 8])
    )
    advantages = simple_group_advantages(scores)
    assert branch == "primary_only"
    assert scores == [0.0] * 4
    assert torch.equal(advantages, torch.zeros(4))


def test_b_positive_zero_std_uses_count_only():
    scores, branch = choose_simple_think_aux_scores(
        _candidates([2, 2, 2, 2], [4, 1, 3, 2], [8, 8, 8, 8])
    )
    assert branch == "zero_std_count_rescue"
    assert scores == [1.0, 0.0, 1.0, 1.0]


def test_c_dead_zero_adds_count_and_diversity():
    scores, branch = choose_simple_think_aux_scores(
        _candidates([0, 0, 0, 0], [4, 1, 3, 2], [1, 8, 4, 2])
    )
    assert branch == "dead_zero_count_diversity_rescue"
    assert scores == [1.125, 1.0, 1.5, 1.25]


def test_d_tied_dead_zero_scores_have_exact_zero_advantage():
    scores, _ = choose_simple_think_aux_scores(
        _candidates([0] * 4, [3] * 4, [4] * 4)
    )
    assert scores == [1.5] * 4
    assert torch.equal(simple_group_advantages(scores), torch.zeros(4))


def test_e_wrong_domain_a_values_do_not_count():
    result = beam_a_diversity(
        [("video", value, 1, 1) for value in range(32)], "prod"
    )
    assert result == {"unique_valid_target_a": 0, "d_a": 0.0}


def test_f_invalid_beams_do_not_count():
    result = beam_a_diversity([None] * 31 + [("prod", 3, 1, 1)], "prod")
    assert result == {"unique_valid_target_a": 1, "d_a": 0.125}


def test_g_diversity_is_capped_at_one():
    result = beam_a_diversity(
        [("prod", value, 1, 1) for value in range(12)], "prod"
    )
    assert result == {"unique_valid_target_a": 12, "d_a": 1.0}


def test_h_five_or_more_interests_score_zero():
    assert [interest_count_score(value) for value in (0, 1, 2, 3, 4, 5, 6, 10)] == [
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ]


def test_simple_is_grounding_invariant_and_differs_from_old_always_on_aux():
    simple_a = _candidates([2] * 4, [4, 1, 3, 2])
    simple_b = [dict(item, grounded_n=0, coverage=0.0, d_cot=0.0) for item in simple_a]
    assert choose_simple_think_aux_scores(simple_a) == choose_simple_think_aux_scores(simple_b)

    old = [{
        "primary_reward": value,
        "s_cot": 1.0,
        "s_prefix": 0.0,
        "s_explore": 0.0,
        "has_exact": False,
    } for value in (8, 2, 0.5, 0)]
    old_scores, _ = choose_think_aux_scores(old)
    simple_scores, _ = choose_simple_think_aux_scores(
        _candidates([8, 2, 0.5, 0], [4, 4, 4, 4])
    )
    assert old_scores == [1.0] * 4
    assert simple_scores == [0.0] * 4


def test_population_normalization_matches_requested_formula():
    scores = [1.0, 0.0, 1.0, 1.0]
    observed = simple_group_advantages(scores)
    values = torch.tensor(scores)
    expected = (values - values.mean()) / (values.std(correction=0) + 1e-4)
    assert torch.allclose(observed, expected)
    assert math.isclose(float(observed.mean()), 0.0, abs_tol=1e-7)
