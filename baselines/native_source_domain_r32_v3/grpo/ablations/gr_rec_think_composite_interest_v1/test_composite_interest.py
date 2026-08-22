from __future__ import annotations

import math
import unittest

from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.calibrate_similarity import auc_rank, counter_f1, lcs_f1, matching_count
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.data_adapter import DataProvenanceError, build_think_composite_dataset, planned_topology
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.g4_activation_audit import audit_reward_vectors
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_composite_interest_v1.interest_metric import MATCH_QUALITY_FLOOR, MATCH_THRESHOLD, beam_utility, composite_reward, coverage_tier, evidence_similarity, maximum_weight_matching, pair_similarity, population_advantages, score_interest_cot
from baselines.native_source_domain_r32_v3.grpo.ablations.gr_rec_think_exact_clamp_v1.think_diagnostics import InterestUnit, extract_interest_units

SID1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
SID2 = "<|video_begin|><s_a_4><s_b_5><s_c_6>"
SID3 = "<|prod_begin|><s_a_7><s_b_8><s_c_9>"
SID4 = "<|ad_begin|><s_a_10><s_b_11><s_c_12>"
WRONG = "<|living_begin|><s_a_99><s_b_98><s_c_97>"
PROMPT = "history " + " ".join((SID1, SID2, SID3, SID4))
HEADING = "\u3010\u5174\u8da3\u5f52\u7eb3\u3011"


def cot(items, heading=HEADING):
    return "<think>\n" + heading + "\n" + "\n".join(str(i) + ". " + text for i, text in enumerate(items, 1)) + "\n</think>\n[]"


GOLD_ITEMS = [
    "tactical game interest " + SID1,
    "beauty care interest " + SID2,
    "home storage interest " + SID3,
    "car review interest " + SID4,
]


class ParserTests(unittest.TestCase):
    def test_canonical_parser_regression(self):
        parsed = extract_interest_units(cot(GOLD_ITEMS), PROMPT)
        self.assertTrue(parsed.parser_success)
        self.assertEqual(len(parsed.units), 4)
        self.assertEqual(parsed.units[0].index, 1)
        self.assertEqual(parsed.units[0].grounded_evidence_sids, (SID1,))
        self.assertEqual(parsed.units[0].normalized_text, "tactical game interest")

    def test_safe_heading_colon_variant(self):
        parsed = extract_interest_units(cot(GOLD_ITEMS, HEADING + "\uff1a"), PROMPT)
        self.assertTrue(parsed.parser_success)
        self.assertEqual(len(parsed.units), 4)

    def test_safe_heading_internal_bold_variant(self):
        heading = "\u3010**\u5174\u8da3\u5f52\u7eb3**\u3011"
        self.assertTrue(extract_interest_units(cot(GOLD_ITEMS, heading), PROMPT).parser_success)

    def test_safe_plain_markdown_heading_variant(self):
        heading = "### \u5174\u8da3\u5f52\u7eb3"
        self.assertTrue(extract_interest_units(cot(GOLD_ITEMS, heading), PROMPT).parser_success)

    def test_inline_prose_is_not_silently_parsed(self):
        text = "<think>\n" + HEADING + "\nfirst interest, second interest, third interest\n</think>"
        parsed = extract_interest_units(text, PROMPT)
        self.assertFalse(parsed.parser_success)
        self.assertEqual(parsed.failure_reason, "empty_interest_section")

    def test_missing_heading_failure(self):
        parsed = extract_interest_units("<think>missing heading</think>", PROMPT)
        self.assertFalse(parsed.parser_success)
        self.assertEqual(parsed.failure_reason, "missing_interest_heading")


class MetricTests(unittest.TestCase):
    def test_identity_with_evidence_is_perfect(self):
        score = score_interest_cot(cot(GOLD_ITEMS), cot(GOLD_ITEMS), PROMPT)
        self.assertEqual(score.matched_interest_count, 4)
        self.assertEqual(score.interest_coverage, 1.0)
        self.assertAlmostEqual(score.match_quality, 1.0)
        self.assertAlmostEqual(score.cot_utility, 1.0)

    def test_identity_without_evidence_is_perfect(self):
        items = ["tactical game", "beauty care"]
        score = score_interest_cot(cot(items), cot(items), PROMPT)
        self.assertEqual(score.matched_interest_count, 2)
        self.assertAlmostEqual(score.mean_match_similarity, 1.0)
        self.assertAlmostEqual(score.cot_utility, 1.0)

    def test_gold_evidence_candidate_missing_is_penalized(self):
        gold = extract_interest_units(cot([GOLD_ITEMS[0]]), PROMPT).units[0]
        candidate = extract_interest_units(cot(["tactical game interest"]), PROMPT).units[0]
        self.assertAlmostEqual(pair_similarity(candidate, gold), 0.7)
        self.assertLess(pair_similarity(candidate, gold), pair_similarity(gold, gold))

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
        score = score_interest_cot(cot(["unique alpha"] * 4), cot(["unique alpha"]), PROMPT)
        self.assertEqual(score.matched_interest_count, 1)

    def test_extra_interest_does_not_increase_matches(self):
        score = score_interest_cot(cot(GOLD_ITEMS + ["unrelated interest"]), cot(GOLD_ITEMS), PROMPT)
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

    def test_match_threshold_boundary(self):
        unit = InterestUnit(1, "x", "x", (), ())
        self.assertEqual(MATCH_THRESHOLD, 0.30)
        self.assertEqual(len(maximum_weight_matching((unit,), (unit,), similarity_fn=lambda _a, _b: 0.2999)), 0)
        self.assertEqual(len(maximum_weight_matching((unit,), (unit,), similarity_fn=lambda _a, _b: 0.30)), 1)
        self.assertEqual(MATCH_QUALITY_FLOOR, 0.60)

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

    def test_population_advantage_exact_and_equal_zero(self):
        rewards = [0.0, 1.0, 2.0, 3.0]
        mean = 1.5
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / 4)
        self.assertEqual(population_advantages(rewards), [(x - mean) / (std + 1e-4) for x in rewards])
        self.assertEqual(population_advantages([0.25] * 4), [0.0] * 4)


class GroupActivationTests(unittest.TestCase):
    def test_equal_beam_different_cot_rescues_advantage(self):
        result = audit_reward_vectors([2.0] * 4, [0.0, 0.16, 0.36, 0.56])
        self.assertEqual(result["A_beam_raw"], [0.0] * 4)
        self.assertGreater(result["composite_population_std"], 0.0)
        self.assertNotEqual(result["A_composite"], [0.0] * 4)

    def test_equal_beam_equal_cot_stays_zero(self):
        result = audit_reward_vectors([2.0] * 4, [0.16] * 4)
        self.assertEqual(result["A_beam_raw"], [0.0] * 4)
        self.assertEqual(result["A_composite"], [0.0] * 4)
        self.assertEqual(result["composite_population_std"], 0.0)

    def test_composite_order_and_population_correction_zero(self):
        cot_values = [0.56, 0.0, 0.36, 0.16]
        result = audit_reward_vectors([2.0] * 4, cot_values)
        expected = [composite_reward(2.0, value) for value in cot_values]
        self.assertEqual(result["composite_reward_vector"], expected)
        self.assertEqual(
            sorted(range(4), key=lambda index: result["composite_reward_vector"][index]),
            [1, 3, 2, 0],
        )
        mean = sum(expected) / 4
        expected_std = math.sqrt(sum((value - mean) ** 2 for value in expected) / 4)
        self.assertAlmostEqual(result["composite_population_std"], expected_std)



class CalibrationHelperTests(unittest.TestCase):
    def test_lcs_and_rank_helpers(self):
        self.assertEqual(lcs_f1("abc", "abc"), 1.0)
        self.assertLess(lcs_f1("abc", "xyz"), 1.0)
        self.assertEqual(auc_rank([1.0], [0.0]), 1.0)

    def test_thresholded_matching_helper(self):
        matrix = [[0.30, 0.0], [0.31, 0.0]]
        self.assertEqual(matching_count(matrix, 0.30), 1)


class DataGuardTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"route": "think", "recommendation_group_id": "g1", "prompt": PROMPT, "all_gold_sids": []},
            {"route": "think", "recommendation_group_id": "g2", "prompt": PROMPT, "all_gold_sids": []},
            {"route": "no_think", "recommendation_group_id": "g1", "prompt": PROMPT, "all_gold_sids": []},
        ]
        self.valid_gold = cot([GOLD_ITEMS[0]])

    def test_missing_gold_is_explicitly_excluded(self):
        joined = build_think_composite_dataset(self.rows, {"g1": self.valid_gold}, eligible_group_ids={"g1"})
        self.assertEqual([row["recommendation_group_id"] for row in joined], ["g1"])

    def test_parser_invalid_gold_is_explicitly_excluded(self):
        joined = build_think_composite_dataset(self.rows, {"g1": self.valid_gold, "g2": "invalid"}, eligible_group_ids={"g1"})
        self.assertEqual(len(joined), 1)

    def test_eligible_missing_gold_fails_closed(self):
        with self.assertRaises(DataProvenanceError):
            build_think_composite_dataset(self.rows, {}, eligible_group_ids={"g1"})

    def test_eligible_parser_invalid_gold_fails_closed(self):
        with self.assertRaises(DataProvenanceError):
            build_think_composite_dataset(self.rows, {"g1": "invalid"}, eligible_group_ids={"g1"})

    def test_prompt_parity_and_no_gold_leakage(self):
        joined = build_think_composite_dataset(self.rows, {"g1": self.valid_gold}, eligible_group_ids={"g1"})
        self.assertEqual(joined[0]["prompt"], self.rows[0]["prompt"])
        self.assertNotIn(self.valid_gold, joined[0]["prompt"])
        self.assertEqual(joined[0]["gold_cot"], self.valid_gold)

    def test_eligible_topology(self):
        topology = planned_topology(group_count=1446)
        self.assertEqual(topology.training_groups, 1440)
        self.assertEqual(topology.fresh_rollouts, 360)
        self.assertEqual(topology.optimizer_steps, 720)


if __name__ == "__main__":
    unittest.main()
