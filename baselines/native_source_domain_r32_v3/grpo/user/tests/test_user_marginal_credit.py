import json
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_chain_reward import score_chain  # noqa: E402
from user_marginal_credit import (  # noqa: E402
    action_marginal_credit,
    chain_marginal_credit,
)


A = "<|video_begin|><s_a_101><s_b_201><s_c_301>"
B = "<|prod_begin|><s_a_102><s_b_202><s_c_302>"
X = "<|ad_begin|><s_a_103><s_b_203><s_c_303>"


def action_sample(history=None):
    return {
        "gold_sids": [A, B],
        "history_sids": list(history or []),
    }


def chain_event(date, action, logic, sid):
    return {"date": date, "action": action, "logic": logic, "sid": sid}


GOLD_EVENTS = [
    chain_event("2026-01-01", "watch alpha", "likes alpha", A),
    chain_event("2026-01-02", "buy beta", "needs beta", B),
]


def chain_sample():
    return {
        "gold_events": GOLD_EVENTS,
        "history_sids": [A, B, X],
    }


def chain_completion(events):
    return json.dumps(
        {"logic_chain": {"name": "test", "events": events}},
        ensure_ascii=False,
    )


class ActionMarginalCreditTest(unittest.TestCase):
    def test_all_gold_sids_are_positive(self):
        result = action_marginal_credit(json.dumps([A, B]), action_sample())
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["credits"]), 2)
        for credit in result["credits"]:
            self.assertTrue(credit["is_gold"])
            self.assertGreater(credit["delta"], 0.0)
            self.assertEqual(credit["credit_type"], "positive")
            start, end = credit["char_span"]
            self.assertEqual(json.dumps([A, B])[start:end], credit["sid"])

    def test_gold_positive_and_false_positive_negative(self):
        result = action_marginal_credit(json.dumps([A, X]), action_sample())
        by_sid = {credit["sid"]: credit for credit in result["credits"]}
        self.assertGreater(by_sid[A]["delta"], 0.0)
        self.assertLess(by_sid[X]["delta"], 0.0)
        self.assertEqual(by_sid[X]["credit_type"], "negative")

    def test_false_positive_credit_is_history_independent(self):
        in_history = action_marginal_credit(
            json.dumps([A, X]), action_sample(history=[X])
        )
        out_of_history = action_marginal_credit(
            json.dumps([A, X]), action_sample(history=[])
        )
        in_credit = next(c for c in in_history["credits"] if c["sid"] == X)
        out_credit = next(c for c in out_of_history["credits"] if c["sid"] == X)
        self.assertLess(in_credit["delta"], 0.0)
        self.assertLess(out_credit["delta"], 0.0)
        self.assertEqual(in_credit["delta"], out_credit["delta"])
        self.assertEqual(in_credit["credit_type"], out_credit["credit_type"])

    def test_duplicate_occurrences_after_first_are_zero(self):
        result = action_marginal_credit(json.dumps([A, A, B]), action_sample())
        self.assertEqual(len(result["credits"]), 3)
        first_a, second_a, b_credit = result["credits"]
        self.assertGreater(first_a["delta"], 0.0)
        self.assertEqual(first_a["occurrence"], 1)
        self.assertEqual(second_a["sid"], A)
        self.assertEqual(second_a["occurrence"], 2)
        self.assertEqual(second_a["delta"], 0.0)
        self.assertEqual(second_a["credit_type"], "zero")
        self.assertGreater(b_credit["delta"], 0.0)

    def test_malformed_action_returns_no_credits(self):
        result = action_marginal_credit("not-json", action_sample())
        self.assertFalse(result["valid"])
        self.assertEqual(result["credits"], [])


class ChainMarginalCreditTest(unittest.TestCase):
    def test_exact_two_event_chain_has_positive_events(self):
        completion = chain_completion(GOLD_EVENTS)
        result = chain_marginal_credit(completion, chain_sample())
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["credits"]), 2)
        for event, credit in zip(GOLD_EVENTS, result["credits"]):
            self.assertGreater(credit["delta"], 0.0)
            self.assertEqual(credit["credit_type"], "positive")
            start, end = credit["char_span"]
            self.assertEqual(json.loads(completion[start:end]), event)

    def test_irrelevant_event_has_negative_credit(self):
        irrelevant = chain_event(
            "2026-01-03", "unrelated action", "unrelated logic", X
        )
        result = chain_marginal_credit(
            chain_completion(GOLD_EVENTS + [irrelevant]), chain_sample()
        )
        self.assertLess(result["credits"][2]["delta"], 0.0)
        self.assertEqual(result["credits"][2]["credit_type"], "negative")

    def test_removing_high_quality_match_lowers_reward(self):
        result = chain_marginal_credit(
            chain_completion(GOLD_EVENTS), chain_sample()
        )
        first = result["credits"][0]
        self.assertGreater(first["full_reward"], first["reward_without_event"])
        self.assertGreater(first["delta"], 0.0)

    def test_action_and_logic_deltas_are_separate_and_total_is_exact(self):
        partial_bad_logic = chain_event(
            "2026-01-01",
            "watch alpha; unrelated action",
            "completely unrelated reasoning",
            A,
        )
        events = [partial_bad_logic, GOLD_EVENTS[1]]
        completion = chain_completion(events)
        result = chain_marginal_credit(completion, chain_sample())
        credit = result["credits"][0]

        without = score_chain(chain_completion([events[1]]), chain_sample())
        expected_reward_delta = result["full_reward"] - without.total_reward
        expected_component_delta = (
            0.5 * credit["delta_action_alignment"]
            + 0.5 * credit["delta_logic_alignment"]
        )
        self.assertGreater(result["full_action_alignment"], 0.0)
        self.assertLess(result["full_action_alignment"], 1.0)
        self.assertLess(
            result["full_logic_alignment"], result["full_action_alignment"]
        )
        self.assertNotEqual(
            credit["delta_action_alignment"], credit["delta_logic_alignment"]
        )
        self.assertAlmostEqual(credit["delta_total"], expected_component_delta, 15)
        self.assertAlmostEqual(credit["delta"], expected_reward_delta, 15)
        self.assertAlmostEqual(credit["reward_delta"], expected_reward_delta, 15)

    def test_malformed_chain_returns_no_credits(self):
        result = chain_marginal_credit("[]", chain_sample())
        self.assertFalse(result["valid"])
        self.assertEqual(result["credits"], [])


if __name__ == "__main__":
    unittest.main()
