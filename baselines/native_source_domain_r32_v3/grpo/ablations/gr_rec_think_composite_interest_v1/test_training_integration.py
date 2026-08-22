from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

from .composite_trainer import (
    ThinkCompositeInterestRecGRPOTrainer, assert_gold_isolation,
    build_global_runtime, classify_winner, score_candidate, slice_global,
)
from .interest_metric import composite_reward, population_advantages
from .run_gr_rec_think_composite_interest_v1 import CHECKPOINT_STEPS, PROBE_IDS, SEED
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units

SID = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
PROMPT = "history " + SID
HEADING = "\u3010\u5174\u8da3\u5f52\u7eb3\u3011"
GOLD = "<think>\n" + HEADING + "\n1. tactical game " + SID + "\n</think>"
GOOD = GOLD
OTHER = "<think>\n" + HEADING + "\n1. beauty care\n</think>"


def records(completions, group_id="g1"):
    return [{"group_id": group_id, "rank": 0, "local_index": index,
             "prompt": PROMPT, "gold_cot": GOLD, "completion": completion}
            for index, completion in enumerate(completions)]


class TrainingChainTests(unittest.TestCase):
    def test_candidate_parser_failure_is_zero_not_exception(self):
        row = score_candidate("invalid", GOLD, PROMPT, 2.0)
        self.assertFalse(row["parser_success"])
        self.assertEqual(row["cot_utility"], 0.0)

    def test_gold_parser_failure_is_hard(self):
        with self.assertRaises(RuntimeError):
            score_candidate(GOOD, "invalid", PROMPT, 2.0)

    def test_gold_leakage_guard(self):
        assert_gold_isolation({"prompt": PROMPT, "gold_cot": GOLD}, PROMPT, PROMPT)
        with self.assertRaisesRegex(RuntimeError, "GOLD_COT_LEAKAGE"):
            assert_gold_isolation({"prompt": PROMPT + GOLD, "gold_cot": GOLD})

    def test_runtime_uses_composite_not_raw_reward_advantage(self):
        runtime = build_global_runtime(records([GOOD, OTHER, GOOD, OTHER]), [2.0] * 4)
        expected = population_advantages([
            composite_reward(2.0, row["cot_utility"]) for row in runtime["candidates"]
        ])
        self.assertEqual(runtime["advantages"], expected)
        self.assertNotEqual(runtime["advantages"], [0.0] * 4)

    def test_zero_std_stays_exact_zero(self):
        runtime = build_global_runtime(records([GOOD] * 4), [2.0] * 4)
        self.assertEqual(runtime["advantages"], [0.0] * 4)

    def test_ddp_global_group_alignment_and_slice(self):
        rows = records([GOOD] * 4, "a") + records([OTHER] * 4, "b")
        runtime = build_global_runtime(rows, [2.0] * 8)
        self.assertEqual(len(runtime["groups"]), 2)
        self.assertEqual(slice_global(list(range(16)), 2, 4, 4), [8, 9, 10, 11])
        with self.assertRaises(RuntimeError):
            build_global_runtime(rows[:3] + rows[4:8], [2.0] * 7)

    def test_winner_categories_are_disjoint(self):
        tie = classify_winner([2.0, 2.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0])
        strict = classify_winner([2.0, 0.0, 0.0, 0.0], [0.4, 0.8, 0.0, 0.0])
        self.assertTrue(tie["top_set_tie_break"])
        self.assertFalse(tie["strict_beam_reversal"])
        self.assertFalse(strict["top_set_tie_break"])
        self.assertTrue(strict["strict_beam_reversal"])

    def test_candidate_monitor_schema_and_bounds(self):
        row = score_candidate(GOOD, GOLD, PROMPT, 16.0)
        required = {"beam_raw", "beam_utility", "matched_interest_count",
                    "interest_coverage", "interest_precision", "mean_match_similarity",
                    "coverage_tier", "match_quality", "cot_utility",
                    "composite_reward", "raw_n", "grounded_n", "grounding_coverage"}
        self.assertTrue(required.issubset(row))
        self.assertTrue(all(math.isfinite(row[key]) for key in
                            ("beam_utility", "cot_utility", "composite_reward")))

    def test_trainer_is_thin_direct_subclass(self):
        self.assertEqual(ThinkCompositeInterestRecGRPOTrainer.__bases__[0].__name__, "RecGRPOTrainer")
        source = inspect.getsource(ThinkCompositeInterestRecGRPOTrainer)
        self.assertNotIn("think_exact_clamp_advantages", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("_compute_loss", source)

    def test_probe_and_checkpoint_contract(self):
        self.assertEqual(SEED, 20260818)
        self.assertEqual(len(PROBE_IDS), 4)
        self.assertEqual(CHECKPOINT_STEPS, (100, 200, 250, 300, 350, 400, 450, 500, 600, 720))

    def test_parser_parity_uses_shared_parser(self):
        self.assertTrue(extract_interest_units(GOOD, PROMPT).parser_success)

    def test_monitor_writer_has_dedicated_schema(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "scripts/monitor/writer.py").read_text(encoding="utf-8")
        self.assertIn("def write_composite", source)
        self.assertIn("composite_interest.jsonl", source)

    def test_gold_cot_is_not_forwarded_to_beam(self):
        from grpo_trl_trainer import make_think_reward_func
        seen = {}

        def beam_spy(prompts, completions, completion_ids, gold_sets):
            seen["arguments"] = (prompts, completions, completion_ids, gold_sets)
            return [2.0]

        reward = make_think_reward_func(beam_spy)
        result = reward(
            [PROMPT], [GOOD], [[1, 2]], route=["think"],
            all_gold_sids=[[SID]], gold_cot=[GOLD],
        )
        self.assertEqual(result, [2.0])
        self.assertEqual(len(seen["arguments"]), 4)
        self.assertNotIn(GOLD, seen["arguments"][0])
        self.assertNotIn(GOLD, seen["arguments"][3])


if __name__ == "__main__":
    unittest.main()
