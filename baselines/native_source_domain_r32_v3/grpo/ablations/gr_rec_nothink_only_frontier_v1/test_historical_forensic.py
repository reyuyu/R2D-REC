"""CPU-only checks for historical Frontier replay and aggregation."""

import unittest

from historical_forensic import aggregate_groups, analyze_trace


def sid(a, b, c):
    return f"<|prod_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def standard_completion(value):
    return f"<think>\n</think>\n该用户最近点击了商品: {value}<|im_end|>"


class HistoricalForensicTests(unittest.TestCase):
    def trace(self, rewards, completions=None):
        mapping = {
            0.0: sid(9, 9, 9),
            0.5: sid(1, 9, 9),
            2.0: sid(1, 2, 9),
            8.0: sid(1, 2, 3),
            -0.25: "<|video_begin|><s_a_1><s_b_2><s_c_3>",
        }
        if completions is None:
            completions = [standard_completion(mapping[value]) for value in rewards]
        return {
            "route": "no_think",
            "scope": "global_group",
            "rollout_id": 1,
            "step": 2,
            "group_id": "g0",
            "gold_sids": [sid(1, 2, 3)],
            "candidates": [
                {"candidate_id": index, "completion": completion, "reward": reward}
                for index, (completion, reward) in enumerate(zip(completions, rewards))
            ],
        }

    def test_mixed_group_exact_frontier_exposure_and_mass(self):
        group = analyze_trace(self.trace([0.0, 0.5, 2.0, 8.0] * 2), "synthetic")
        self.assertEqual(group["frontier_counts"], {"domain": 0, "a": 2, "b": 2, "c": 2})
        self.assertEqual(group["stage_success_counts"], {"domain": 8, "a": 6, "b": 4, "c": 2})
        self.assertAlmostEqual(group["credit_mass"]["sum_signed_token_credit"], -0.40625)
        self.assertAlmostEqual(group["credit_mass"]["sum_abs_token_credit"], 3.59375)

    def test_nonempty_think_is_exclusive_format_penalty(self):
        rewards = [8.0] * 8
        completions = [standard_completion(sid(1, 2, 3)) for _ in rewards]
        completions[0] = f"<think>reasoning</think>{sid(1, 2, 3)}"
        group = analyze_trace(self.trace(rewards, completions), "synthetic")
        first = group["candidates"][0]
        self.assertFalse(first["format_valid"])
        self.assertEqual(first["effective_reward"], -1.0)
        self.assertEqual(first["format_credit_total"], -0.09375)
        self.assertEqual(first["stage_credits"], [0.0, 0.0, 0.0, 0.0])

    def test_aggregate_rates_and_c_histogram(self):
        mixed = analyze_trace(self.trace([0.0, 0.5, 2.0, 8.0] * 2), "synthetic")
        dead = analyze_trace(self.trace([0.0] * 8), "synthetic")
        summary = aggregate_groups([mixed, dead])
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["candidate_count"], 16)
        self.assertEqual(summary["frontier_candidate_exposure"]["a"]["count"], 10)
        self.assertEqual(summary["frontier_group_exposure"]["c"]["count"], 1)
        self.assertEqual(summary["c_frontier_count_per_g8_histogram"]["2"], 1)
        self.assertEqual(summary["group_patterns"]["dead_zero"], 1)


if __name__ == "__main__":
    unittest.main()
