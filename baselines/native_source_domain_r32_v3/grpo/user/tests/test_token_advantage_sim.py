import math
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate_user_token_advantage import (
    compile_effective_penalties,
    population_advantages,
    simulate_token_advantages,
)


def candidate(records, token_count=6):
    return {
        "route": "action",
        "completion_token_count": token_count,
        "penalty": {"records": records},
    }


def record(kind, indices, included=True):
    return {
        "kind": kind,
        "included": included,
        "masked_token_indices": indices,
    }


class TokenAdvantageSimulationTests(unittest.TestCase):
    def test_population_std_matches_grpo_definition(self):
        rewards = [0.0, 1.0, 2.0, 3.0]
        advantages, mean, std = population_advantages(rewards)
        self.assertEqual(mean, statistics.fmean(rewards))
        self.assertEqual(std, statistics.pstdev(rewards))
        self.assertAlmostEqual(advantages[0], (0.0 - mean) / (std + 1e-4))

    def test_zero_std_gives_zero_task_advantage(self):
        advantages, _, std = population_advantages([0.5, 0.5, 0.5, 0.5])
        self.assertEqual(std, 0.0)
        self.assertEqual(advantages, [0.0, 0.0, 0.0, 0.0])

    def test_masked_and_unmasked_tokens(self):
        item = candidate([record("hallucinated_sid", [1, 2])])
        penalties, _ = compile_effective_penalties(item, "fixed", {"hallucinated_sid": 0.5})
        output = simulate_token_advantages(0.8, penalties)
        self.assertEqual(output, [0.8, -0.5, -0.5, 0.8, 0.8, 0.8])

    def test_overlap_takes_strongest_lambda_without_sum(self):
        item = candidate(
            [record("hallucinated_sid", [1, 2]), record("duplicate_sid", [2, 3])]
        )
        penalties, winners = compile_effective_penalties(
            item, "fixed", {"hallucinated_sid": 0.5, "duplicate_sid": 0.25}
        )
        self.assertEqual(penalties, [0.0, 0.5, 0.5, 0.25, 0.0, 0.0])
        self.assertEqual(winners[2], {"hallucinated_sid"})
        self.assertNotEqual(penalties[2], 0.75)

    def test_sqrt_normalization(self):
        item = candidate([record("duplicate_sid", [1, 2, 3, 4])])
        penalties, _ = compile_effective_penalties(item, "sqrt", {"duplicate_sid": 0.5})
        for index in (1, 2, 3, 4):
            self.assertAlmostEqual(penalties[index], 0.5 / math.sqrt(4))

    def test_non_whitelisted_record_never_masks(self):
        item = candidate([record("wrong_selection_sid", [1, 2])])
        penalties, _ = compile_effective_penalties(item, "fixed", {})
        self.assertEqual(penalties, [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
