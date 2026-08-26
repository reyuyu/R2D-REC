import importlib.util
import itertools
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import think_credit_v1 as credit  # noqa: E402


def abc(a, b=1, c=1):
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


GOLD3 = [abc(1, 1, 1), abc(2, 2, 2), abc(3, 3, 3)]


def candidate(value, gold=GOLD3, valid=True):
    parsed = credit.split_abc(value) if value else None
    gold_parts = [credit.split_abc(item) for item in gold]
    return {
        "format_valid": valid,
        "parsed_abc": value,
        "A_hit": bool(valid and parsed and parsed[0] in {item[0] for item in gold_parts}),
        "AB_hit": bool(valid and parsed and parsed[:2] in {item[:2] for item in gold_parts}),
        "exact": bool(valid and parsed and parsed in set(gold_parts)),
    }


def group(values, gold=GOLD3):
    assert len(values) == 8
    return [candidate(value, gold) for value in values]


def test_all_wrong_has_zero_credit_no_gold_rescue_and_no_hpr_a():
    plan = credit.plan_think_credit(group([abc(9)] * 8), GOLD3)
    assert all(value == 0 for row in plan.token_credits for value in row)
    assert all(row == (True, False, False) for row in plan.semantic_active_mask)
    assert plan.a_rescue.reason == "NO_GOLD_A"
    assert plan.bc_hpr.trigger == "NONE"


def test_single_correct_a_has_max_reward_and_wrong_a_has_no_negative():
    plan = credit.plan_think_credit(group([abc(1, 9, 9)] + [abc(9)] * 7), GOLD3)
    assert plan.token_credits[0][0] == pytest.approx(0.5 * (1 - 1 / 8) / 8)
    assert all(value >= 0 for row in plan.token_credits for value in row)


def test_five_same_correct_a_has_smaller_reward_and_collapse_rescue():
    plan = credit.plan_think_credit(group([abc(1, 9, 9)] * 5 + [abc(9)] * 3), GOLD3)
    assert plan.token_credits[0][0] == pytest.approx(0.5 * (1 - 5 / 8) / 8)
    assert plan.a_rescue.reason == "A_MODE_COLLAPSE"


def test_coverage_three_disables_rescue_even_with_mode_count_five():
    values = [abc(1, 9, 9)] * 5 + [abc(2, 9, 9), abc(3, 9, 9), abc(9)]
    plan = credit.plan_think_credit(group(values), GOLD3)
    assert plan.a_statistics["d_A"] == 3
    assert not plan.a_rescue.triggered


def test_k_a_one_never_mode_collapse_but_zero_coverage_rescues():
    gold = [abc(1)]
    hit = credit.plan_think_credit(group([abc(1)] * 8, gold), gold)
    miss = credit.plan_think_credit(group([abc(9)] * 8, gold), gold)
    assert hit.a_statistics["T_A"] == 1 and not hit.a_rescue.triggered
    assert miss.a_rescue.reason == "NO_GOLD_A"


def test_a_hit_without_ab_produces_hpr_b_and_zero_b():
    plan = credit.plan_think_credit(group([abc(1, 9, 9)] + [abc(9)] * 7), GOLD3)
    assert plan.candidates[0].kinds == (credit.POSITIVE, credit.ZERO, credit.GATED)
    assert plan.bc_hpr.trigger == "HPR_B"
    assert plan.bc_hpr.sites[0].target_tokens == ("<s_b_1>",)


def test_ab_without_exact_produces_hpr_c_and_zero_c():
    plan = credit.plan_think_credit(group([abc(1, 1, 9)] + [abc(9)] * 7), GOLD3)
    assert plan.candidates[0].kinds == (credit.POSITIVE, credit.POSITIVE, credit.ZERO)
    assert plan.bc_hpr.trigger == "HPR_C"
    assert plan.bc_hpr.sites[0].target_tokens == ("<s_c_1>",)


def test_exact_produces_all_positive_and_hpr_none():
    plan = credit.plan_think_credit(group([abc(1)] + [abc(9)] * 7), GOLD3)
    assert plan.candidates[0].kinds == (credit.POSITIVE,) * 3
    assert plan.bc_hpr.trigger == "HPR_NONE"


@pytest.mark.parametrize(
    "values,trigger",
    [
        ([abc(1, 9, 9)] * 5 + [abc(9)] * 3, "HPR_B"),
        ([abc(1, 1, 9)] * 5 + [abc(9)] * 3, "HPR_C"),
        ([abc(1)] * 5 + [abc(9)] * 3, "HPR_NONE"),
    ],
)
def test_a_collapse_can_coexist_with_each_bc_plan(values, trigger):
    plan = credit.plan_think_credit(group(values), GOLD3)
    assert plan.a_rescue.reason == "A_MODE_COLLAPSE"
    assert plan.bc_hpr.trigger == trigger


def test_b_and_c_positive_frontier_values():
    plan = credit.plan_think_credit(group([abc(1)] + [abc(9)] * 7), GOLD3)
    assert plan.token_credits[0][1] == pytest.approx(1.5 * (1 - 1 / 8) / 8)
    assert plan.token_credits[0][2] == pytest.approx(6.0 * (1 - 1 / 8) / 8)


@pytest.mark.parametrize(
    "bad",
    [
        {"A_hit": False, "AB_hit": True, "exact": False},
        {"A_hit": True, "AB_hit": False, "exact": True},
    ],
)
def test_hierarchy_invariant_fails_closed(bad):
    candidates = group([abc(9)] * 8)
    candidates[0].update(bad)
    with pytest.raises(ValueError, match="hierarchy invariant"):
        credit.plan_think_credit(candidates, GOLD3)


def test_g_not_eight_fails_closed():
    with pytest.raises(ValueError, match="G8"):
        credit.plan_think_credit(group([abc(9)] * 8)[:7], GOLD3)


def test_duplicate_gold_is_deduplicated_for_a_target():
    gold = [abc(1), abc(1), abc(2, 2, 2)]
    plan = credit.plan_think_credit(group([abc(9)] * 8, gold), gold)
    assert plan.a_statistics["K_A"] == 2
    assert plan.a_rescue.target_a_tokens == ("<s_a_1>", "<s_a_2>")


def test_wrong_a_repeated_eight_does_not_trigger_mode_collapse():
    plan = credit.plan_think_credit(group([abc(9)] * 8), GOLD3)
    assert plan.a_statistics["max_correct_gold_A_count"] == 0
    assert plan.a_rescue.reason == "NO_GOLD_A"
    assert not plan.monitoring["A_mode_collapse"]


def test_format_invalid_has_no_semantic_negative():
    candidates = group([abc(9)] * 8)
    candidates[0] = candidate(None, valid=False)
    plan = credit.plan_think_credit(candidates, GOLD3)
    assert plan.token_credits[0] == (0.0, 0.0, 0.0)
    assert plan.semantic_active_mask[0] == (False, False, False)


def test_frequency_monotonic_and_exhaustive_invariants():
    previous = None
    for count in range(1, 9):
        values = [abc(1, 9, 9)] * count + [abc(9)] * (8 - count)
        plan = credit.plan_think_credit(group(values), GOLD3)
        value = plan.token_credits[0][0]
        assert all(item >= 0 for row in plan.token_credits for item in row)
        if previous is not None:
            assert value <= previous
        previous = value
        assert set(plan.a_rescue.target_a_tokens) == {"<s_a_1>", "<s_a_2>", "<s_a_3>"}
    for pattern in itertools.product(range(4), repeat=4):
        values = [abc(value if value else 9, 9, 9) for value in pattern] + [abc(9)] * 4
        plan = credit.plan_think_credit(group(values), GOLD3)
        if plan.a_statistics["d_A"] >= plan.a_statistics["T_A"]:
            assert not plan.a_rescue.triggered


def test_bc_targets_are_directly_reused_from_frozen_hpr(monkeypatch):
    called = {"value": False}
    original = credit.plan_hpr

    def wrapped(candidates, gold):
        called["value"] = True
        return original(candidates, gold)

    monkeypatch.setattr(credit, "plan_hpr", wrapped)
    plan = credit.plan_think_credit(group([abc(1, 9, 9)] + [abc(9)] * 7), GOLD3)
    assert called["value"] and plan.bc_hpr.trigger == "HPR_B"
