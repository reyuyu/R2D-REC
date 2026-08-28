import unittest

from .suffix_objective import (
    MAX_GENERATED_CANDIDATES,
    parse_suffix_completion,
    resample_decision,
    suffix_mask_for_ids,
    suffix_reward,
)


GOLD = ["<|video_begin|><s_a_1><s_b_2><s_c_3>"]


class SuffixObjectiveTests(unittest.TestCase):
    def test_full_nothink_v1_reward_hierarchy(self):
        cases = (
            ("<|video_begin|><s_a_1><s_b_2><s_c_3>", 8.0),
            ("<|video_begin|><s_a_1><s_b_2><s_c_9>", 2.0),
            ("<|video_begin|><s_a_1><s_b_9><s_c_9>", 0.5),
            ("<|video_begin|><s_a_9><s_b_9><s_c_9>", 0.0),
            ("<|prod_begin|><s_a_1><s_b_2><s_c_3>", -0.25),
            ("not a SID", -1.0),
        )
        for suffix, expected in cases:
            with self.subTest(suffix=suffix):
                reward, _ = suffix_reward(f"reason</think>{suffix}", GOLD)
                self.assertEqual(reward, expected)

    def test_single_sid_after_think(self):
        text = "reason</think>next: <|video_begin|><s_a_1><s_b_2><s_c_3>"
        reward, parsed = suffix_reward(text, GOLD)
        self.assertEqual(reward, 8.0)
        self.assertEqual(parsed.sid_count, 1)
        self.assertFalse(parsed.multi_sid_output)
        self.assertEqual(parsed.parser_status, "ok")

    def test_multiple_sids_are_flagged_and_last_sid_is_rewarded(self):
        text = (
            "reason</think>candidate <|video_begin|><s_a_9><s_b_9><s_c_9> "
            "final <|video_begin|><s_a_1><s_b_2><s_c_3>"
        )
        reward, parsed = suffix_reward(text, GOLD)
        self.assertEqual(reward, 8.0)
        self.assertEqual(parsed.sid_count, 2)
        self.assertTrue(parsed.multi_sid_output)
        self.assertEqual(parsed.parser_status, "multiple_suffix_sids")
        self.assertEqual(parsed.parsed_sid, ("video", 1, 2, 3))

    def test_sid_before_think_does_not_count(self):
        text = "<|video_begin|><s_a_1><s_b_2><s_c_3></think>no answer"
        reward, parsed = suffix_reward(text, GOLD)
        self.assertEqual(reward, -1.0)
        self.assertEqual(parsed.sid_count, 0)

    def test_missing_close_has_no_trainable_suffix(self):
        text = "reason <|video_begin|><s_a_1><s_b_2><s_c_3>"
        reward, parsed = suffix_reward(text, GOLD)
        self.assertEqual(reward, -1.0)
        self.assertFalse(parsed.closed)
        self.assertEqual(parsed.parser_status, "missing_think_close")

    def test_suffix_mask_starts_after_full_close_sequence(self):
        self.assertEqual(
            suffix_mask_for_ids([7, 8, 9, 10, 11, 0], [9, 10], valid_length=5),
            [0, 0, 0, 0, 1, 0],
        )
        self.assertEqual(suffix_mask_for_ids([7, 8, 9], [4]), [0, 0, 0])

    def test_zero_std_retry_budget_is_exactly_32(self):
        rewards = [0.0] * 8
        for attempt in range(3):
            decision = resample_decision(rewards, attempt)
            self.assertTrue(decision["retry"])
            self.assertFalse(decision["exhausted"])
        final = resample_decision(rewards, 3)
        self.assertFalse(final["retry"])
        self.assertTrue(final["exhausted"])
        self.assertEqual(final["generated_candidate_total"], 32)
        self.assertEqual(MAX_GENERATED_CANDIDATES, 32)

    def test_first_nonzero_round_is_accepted(self):
        decision = resample_decision([0, 0, 0, 0, 0, 0, 0, 8], 2)
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["retry"])
        self.assertFalse(decision["exhausted"])
        self.assertEqual(decision["generated_candidate_total"], 24)


if __name__ == "__main__":
    unittest.main()
