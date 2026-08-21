"""Exact CPU regressions for NoThink Frontier credit."""

import unittest

from frontier_credit import FORMAT_ADV_TOTAL, distribute_format_penalty, plan_frontier_credits
from format_validator import effective_scalar_reward, validate_nothink_completion
from nothink_bridge import BRIDGE_DEAD_A, plan_dead_zero_bridge
from nothink_hierarchical_credit import HierarchyState

SID = "<|prod_begin|><s_a_1><s_b_2><s_c_3>"
DECLARATION = "该用户最近点击了商品:"


def state_for_reward(reward):
    values = {
        -0.25: (True, False, False, False, False),
        0.0: (True, True, False, False, False),
        0.5: (True, True, True, False, False),
        2.0: (True, True, True, True, False),
        8.0: (True, True, True, True, True),
    }
    return HierarchyState(*values[float(reward)])


def plan(rewards):
    return plan_frontier_credits([state_for_reward(value) for value in rewards], [True] * 8)


class FrontierCreditTests(unittest.TestCase):
    def test_1_to_4_uniform_frontiers(self):
        cases = (
            ([-0.25] * 8, 0, -0.03125, "ALL_WRONG_DOMAIN_FRONTIER"),
            ([0.0] * 8, 1, -0.0625, "UNIFORM_A_FAILURE_FRONTIER"),
            ([0.5] * 8, 2, -0.1875, "UNIFORM_B_FAILURE_FRONTIER"),
            ([2.0] * 8, 3, -0.75, "UNIFORM_C_FAILURE_FRONTIER"),
        )
        for rewards, column, expected, taxonomy in cases:
            with self.subTest(taxonomy=taxonomy):
                result = plan(rewards)
                self.assertEqual(result.taxonomy, taxonomy)
                self.assertTrue(all(item.credits[column] == expected for item in result.candidates))
                self.assertTrue(all(sum(value != 0 for value in item.credits) == 1 for item in result.candidates))

    def test_5_uniform_exact_is_zero_signal(self):
        result = plan([8.0] * 8)
        self.assertEqual(result.taxonomy, "UNIFORM_EXACT_ZERO_SIGNAL")
        self.assertTrue(all(item.credits == (0.0, 0.0, 0.0, 0.0) for item in result.candidates))

    def test_6_mixed_archetype_exact_values(self):
        result = plan([0.0, 0.5, 2.0, 8.0] * 2)
        expected = (
            (0.0, -0.0625, 0.0, 0.0),
            (0.0, 0.015625, -0.1875, 0.0),
            (0.0, 0.015625, 0.09375, -0.75),
            (0.0, 0.015625, 0.09375, 0.5625),
        ) * 2
        self.assertEqual(tuple(item.credits for item in result.candidates), expected)

    def test_7_singleton_b_success_survives(self):
        result = plan([0.0] * 7 + [2.0])
        self.assertTrue(all(item.credits[1] == -0.0625 for item in result.candidates[:7]))
        self.assertEqual(result.candidates[-1].credits[1:3], (0.0546875, 0.1640625))

    def test_8_singleton_exact_success_survives(self):
        result = plan([0.0] * 7 + [8.0])
        self.assertTrue(all(item.credits[1] == -0.0625 for item in result.candidates[:7]))
        self.assertEqual(result.candidates[-1].credits[1:], (0.0546875, 0.1640625, 0.65625))

    def test_9_nonempty_cot_forces_reward_and_gates_hierarchy(self):
        validation = validate_nothink_completion(f"<think>reasoning</think>{SID}")
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "nonempty_think")
        self.assertEqual(effective_scalar_reward(8.0, validation), -1.0)
        result = plan_frontier_credits([state_for_reward(8.0)] * 8, [False] + [True] * 7)
        self.assertEqual(result.candidates[0].credits, (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(sum(distribute_format_penalty(19)), FORMAT_ADV_TOTAL)

    def test_10_format_penalty_has_no_length_amplification(self):
        for length in (10, 100, 500):
            self.assertAlmostEqual(sum(distribute_format_penalty(length)), -0.09375)

    def test_11_standard_empty_think_is_valid(self):
        result = validate_nothink_completion(f"<think>\n </think>\n{DECLARATION} {SID}")
        self.assertTrue(result.valid)
        self.assertEqual(result.mode, "branch")
        without_empty_think = validate_nothink_completion(f"{DECLARATION} {SID}")
        self.assertFalse(without_empty_think.valid)
        self.assertEqual(without_empty_think.reason, "unexpected_prose_before_sid")

    def test_12_direct_sid_fallback_is_valid(self):
        for prefix in ("", "<think>\n</think>\n"):
            result = validate_nothink_completion(prefix + SID)
            self.assertTrue(result.valid)
            self.assertEqual(result.mode, "direct_sid_fallback")

    def test_13_prose_plus_sid_is_invalid(self):
        result = validate_nothink_completion("综合用户兴趣来看……" + SID)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "unexpected_prose_before_sid")

    def test_invalid_sid_is_format_violation(self):
        result = validate_nothink_completion("<think></think><|prod_begin|><s_a_1>")
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "invalid_sid")

    def test_zero_reward_bridge_regression_stays_active(self):
        bridge = plan_dead_zero_bridge([0.0] * 8, [("prod", 1, 2, 3)], "prod")
        self.assertTrue(bridge.active)
        self.assertEqual(bridge.branch, BRIDGE_DEAD_A)

    def test_prefix_gate_only_first_failure_is_negative(self):
        for rewards, column in (([-0.25] * 8, 0), ([0.0] * 8, 1), ([0.5] * 8, 2)):
            for candidate in plan(rewards).candidates:
                self.assertLess(candidate.credits[column], 0)
                self.assertEqual(candidate.credits[column + 1:], (0.0,) * (3 - column))


if __name__ == "__main__":
    unittest.main()
