from __future__ import annotations

import math
import unittest

from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.data_adapter import DataProvenanceError, build_think_composite_dataset, planned_topology
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.interest_metric import beam_utility, composite_reward, coverage_tier, evidence_similarity, maximum_weight_matching, pair_similarity, population_advantages, score_interest_cot
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_exact_clamp_v1.think_diagnostics import InterestUnit, extract_interest_units

SID1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
SID2 = "<|video_begin|><s_a_4><s_b_5><s_c_6>"
SID3 = "<|prod_begin|><s_a_7><s_b_8><s_c_9>"
SID4 = "<|ad_begin|><s_a_10><s_b_11><s_c_12>"
WRONG = "<|living_begin|><s_a_99><s_b_98><s_c_97>"
PROMPT = "history " + " ".join((SID1, SID2, SID3, SID4))


def cot(items):
    heading = "<think>\n#### 【兴趣归纳】\n"
    return heading + "\n".join(str(i) + ". " + text for i, text in enumerate(items, 1)) + "\n</think>\n[]"


GOLD_ITEMS = [
    "tactical game interest " + SID1,
    "beauty care interest " + SID2,
    "home storage interest " + SID3,
    "car review interest " + SID4,
]


class ParserTests(unittest.TestCase):
    def test_parser_extracts_units_and_grounding(self):
        parsed = extract_interest_units(cot(GOLD_ITEMS), PROMPT)
        self.assertTrue(parsed.parser_success)
        self.assertEqual(len(parsed.units), 4)
        self.assertEqual(parsed.units[0].grounded_evidence_sids, (SID1,))
        self.assertNotIn(SID1, parsed.units[0].normalized_text)

    def test_parser_failure(self):
        parsed = extract_interest_units("<think>missing heading</think>", PROMPT)
        self.assertFalse(parsed.parser_success)
        self.assertEqual(parsed.failure_reason, "missing_interest_heading")

    def test_empty_interest_section(self):
        parsed = extract_interest_units("<think>\n【兴趣归纳】\n</think>", PROMPT)
        self.assertFalse(parsed.parser_success)
        self.assertEqual(parsed.failure_reason, "empty_interest_section")


class MetricTests(unittest.TestCase):
    def test_identity_is_perfect(self):
        score = score_interest_cot(cot(GOLD_ITEMS), cot(GOLD_ITEMS), PROMPT)
        self.assertEqual(score.matched_interest_count, 4)
        self.assertEqual(score.interest_coverage, 1.0)
        self.assertAlmostEqual(score.match_quality, 1.0)
        self.assertAlmostEqual(score.cot_utility, 1.0)

    def test_reorder_is_invariant(self):
        identity = score_interest_cot(cot(GOLD_ITEMS), cot(GOLD_ITEMS), PROMPT)
        reordered = score_interest_cot(cot(list(reversed(GOLD_ITEMS))), cot(GOLD_ITEMS), PROMPT)
        self.assertEqual(reordered.matched_interest_count, identity.matched_interest_count)
        self.assertAlmostEqual(reordered.cot_utility, identity.cot_utility)

    def test_coverage_tiers_for_four(self):
        self.assertEqual([coverage_tier(k, 4) for k in range(5)], [0.0, 0.20, 0.45, 0.70, 1.0])

    def test_coverage_tiers_non_four_gold_counts(self):
        self.assertEqual(coverage_tier(1, 1), 1.0)
        self.assertEqual(coverage_tier(1, 2), 0.45)
        self.assertEqual(coverage_tier(1, 3), 0.45)
        self.assertEqual(coverage_tier(1, 5), 0.20)
        self.assertEqual(coverage_tier(4, 6), 0.70)

    def test_duplicate_cannot_consume_gold_twice(self):
        score = score_interest_cot(cot([GOLD_ITEMS[0]] * 4), cot(GOLD_ITEMS), PROMPT)
        self.assertEqual(score.matched_interest_count, 1)

    def test_extra_interest_does_not_increase_matches(self):
        score = score_interest_cot(cot(GOLD_ITEMS + ["unrelated astronomy"]), cot(GOLD_ITEMS), PROMPT)
        self.assertEqual(score.matched_interest_count, 4)
        self.assertEqual(score.interest_precision, 0.8)

    def test_evidence_rules(self):
        self.assertEqual(evidence_similarity([], []), 0.0)
        self.assertEqual(evidence_similarity([SID1], []), 0.0)
        self.assertEqual(evidence_similarity([SID1], [SID1]), 1.0)
        self.assertEqual(evidence_similarity([SID1], [SID2]), 0.0)

    def test_wrong_sid_lowers_pair_score(self):
        good = extract_interest_units(cot([GOLD_ITEMS[0]]), PROMPT).units[0]
        wrong = extract_interest_units(cot(["tactical game interest " + WRONG]), PROMPT).units[0]
        self.assertLess(pair_similarity(wrong, good), pair_similarity(good, good))

    def test_text_only_similarity_has_no_grounding_credit(self):
        gold = extract_interest_units(cot([GOLD_ITEMS[0]]), PROMPT).units[0]
        text_only = extract_interest_units(cot(["tactical game interest"]), PROMPT).units[0]
        self.assertAlmostEqual(pair_similarity(text_only, gold), 0.7)

    def test_match_threshold_boundary(self):
        unit = InterestUnit(1, "x", "x", (), ())
        self.assertEqual(len(maximum_weight_matching((unit,), (unit,), similarity_fn=lambda _a, _b: 0.59)), 0)
        self.assertEqual(len(maximum_weight_matching((unit,), (unit,), similarity_fn=lambda _a, _b: 0.60)), 1)

    def test_beam_mapping(self):
        expected = {0: 0.0, 0.5: math.log1p(0.5) / math.log(17), 2: math.log(3) / math.log(17), 8: math.log(9) / math.log(17), 16: 1.0, 32: 1.0}
        for raw, value in expected.items():
            self.assertAlmostEqual(beam_utility(raw), value)
        for invalid in (-1, None, float("nan"), float("inf"), "bad"):
            self.assertEqual(beam_utility(invalid), 0.0)

    def test_composite_is_bounded(self):
        for beam in (-1, 0, 0.5, 2, 8, 16, 100):
            for cot_value in (-2, 0, 0.5, 1, 2):
                self.assertGreaterEqual(composite_reward(beam, cot_value), 0.0)
                self.assertLessEqual(composite_reward(beam, cot_value), 1.0)

    def test_population_advantage_exact(self):
        rewards = [0.0, 1.0, 2.0, 3.0]
        mean = 1.5
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / 4)
        expected = [(x - mean) / (std + 1e-4) for x in rewards]
        self.assertEqual(population_advantages(rewards), expected)

    def test_equal_reward_advantages_are_zero(self):
        self.assertEqual(population_advantages([0.25] * 4), [0.0] * 4)


class DataGuardTests(unittest.TestCase):
    def test_incomplete_provenance_fails_closed(self):
        rows = [{"route": "think", "recommendation_group_id": "g", "prompt": "original"}]
        with self.assertRaises(DataProvenanceError):
            build_think_composite_dataset(rows, {"g": "gold"}, provenance_ready=False)

    def test_prompt_parity_and_no_gold_leakage(self):
        rows = [{"route": "think", "recommendation_group_id": "g", "prompt": "original", "all_gold_sids": []}]
        joined = build_think_composite_dataset(rows, {"g": "SECRET GOLD"}, provenance_ready=True)
        self.assertEqual(joined[0]["prompt"], rows[0]["prompt"])
        self.assertNotIn("SECRET GOLD", joined[0]["prompt"])
        self.assertEqual(joined[0]["gold_cot"], "SECRET GOLD")

    def test_static_topology(self):
        topology = planned_topology()
        self.assertEqual(topology.training_groups, 1544)
        self.assertEqual(topology.fresh_rollouts, 386)
        self.assertEqual(topology.optimizer_steps, 772)


if __name__ == "__main__":
    unittest.main()
