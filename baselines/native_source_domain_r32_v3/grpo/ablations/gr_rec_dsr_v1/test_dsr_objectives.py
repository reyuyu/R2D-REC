"""CPU tests for Think DSR branch selection and score math."""
import math

import torch

from gr_rec_dsr_v1.dsr_objectives import (
    choose_think_aux_scores,
    exploration_score,
    group_aux_advantages,
    prefix_support,
)


GOLD1 = ("prod", 1, 10, 100)
GOLD2 = ("prod", 2, 20, 200)


def candidate(reward, cot, prefix=0.0, explore=0.0, exact=False):
    return {
        "primary_reward": reward,
        "s_cot": cot,
        "s_prefix": prefix,
        "s_explore": explore,
        "has_exact": exact,
    }


def test_primary_variance_uses_only_cot():
    group = [candidate(0, 0.1), candidate(0, 0.2), candidate(0.5, 0.3), candidate(2, 0.4)]
    scores, branch = choose_think_aux_scores(group)
    assert branch == "primary_variance" and scores == [0.1, 0.2, 0.3, 0.4]


def test_all_zero_dead_branch():
    group = [candidate(0, 0.5, explore=value) for value in (0.0, 0.25, 0.5, 1.0)]
    scores, branch = choose_think_aux_scores(group)
    assert branch == "dead_zero"
    assert scores[-1] > scores[0]


def test_zero_std_positive_prefix_branch():
    group = [candidate(0.5, 0.5, prefix=value) for value in (0.1, 0.2, 0.3, 0.4)]
    scores, branch = choose_think_aux_scores(group)
    assert branch == "prefix_rescue" and scores[-1] > scores[0]


def test_saturated_high_is_cot_only():
    group = [candidate(8.0, value, prefix=1.0, exact=True) for value in (0.1, 0.2, 0.3, 0.4)]
    scores, branch = choose_think_aux_scores(group)
    assert branch == "cot_only_saturated_or_other"
    assert scores == [0.1, 0.2, 0.3, 0.4]


def test_equal_aux_scores_have_zero_advantage():
    assert torch.equal(group_aux_advantages([0.5] * 4, 4), torch.zeros(4))


def test_concave_prefix_support_rewards_dispersion():
    # Same four total A hits. Splitting 2+2 across both gold prefixes has a
    # larger mean sqrt support than concentrating all 4 on one prefix.
    concentrated = [GOLD1] * 4 + [None] * 28
    dispersed = [GOLD1] * 2 + [GOLD2] * 2 + [None] * 28
    assert prefix_support(dispersed, {GOLD1, GOLD2})["s_a"] > prefix_support(concentrated, {GOLD1, GOLD2})["s_a"]


def test_wrong_a_diversity_is_gated_by_zero_cot():
    beams = [("prod", index, 1, 1) for index in range(1, 33)]
    explore = exploration_score(beams, "prod")["s_explore"]
    group = [candidate(0, 0.0, explore=explore) for _ in range(4)]
    scores, branch = choose_think_aux_scores(group)
    assert branch == "dead_zero" and explore > 0 and scores == [0.0] * 4


if __name__ == "__main__":
    from gr_rec_dsr_v1.run_cpu_tests import main
    raise SystemExit(main([__name__]))
