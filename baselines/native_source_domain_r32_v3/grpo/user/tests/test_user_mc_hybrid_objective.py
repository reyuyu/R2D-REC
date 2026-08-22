import sys
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_hybrid_objective import mc_hybrid_objective  # noqa: E402
from user_mc_objective import mc_unit_credit_loss  # noqa: E402


def unit(delta, indices):
    return {"delta": delta, "generated_token_indices": list(indices)}


class MCHybridObjectiveTests(unittest.TestCase):
    def test_sequence_advantage_uses_population_std(self):
        rewards = torch.tensor([0.4, 0.5, 0.9, 0.2], dtype=torch.float64)
        logps = torch.full((4, 3), -1.0, dtype=torch.float64)
        _, metadata = mc_hybrid_objective(
            logps, rewards, [[], [], [], []], torch.ones_like(logps)
        )
        expected_mean = rewards.mean()
        expected_std = rewards.std(unbiased=False)
        expected_advantages = (rewards - expected_mean) / (expected_std + 1e-6)
        self.assertAlmostEqual(metadata["group_reward_mean"], float(expected_mean), 15)
        self.assertAlmostEqual(metadata["group_reward_std"], float(expected_std), 15)
        self.assertAlmostEqual(metadata["group_reward_min"], 0.2, 15)
        self.assertAlmostEqual(metadata["group_reward_max"], 0.9, 15)
        self.assertAlmostEqual(metadata["group_reward_spread"], 0.7, 15)
        self.assertTrue(torch.allclose(metadata["sequence_advantages"], expected_advantages))

    def test_equal_rewards_zero_sequence_advantages_and_loss(self):
        logps = torch.randn((4, 5), dtype=torch.float64)
        _, metadata = mc_hybrid_objective(
            logps, [0.5] * 4, [[], [], [], []], torch.ones_like(logps)
        )
        self.assertTrue(torch.equal(metadata["sequence_advantages"], torch.zeros(4, dtype=logps.dtype)))
        self.assertEqual(float(metadata["sequence_loss"]), 0.0)

    def test_hybrid_weighting_formula(self):
        logps = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=torch.float64)
        _, metadata = mc_hybrid_objective(
            logps,
            [0.8, 0.2],
            [[unit(0.4, [0])], [unit(-0.2, [1])]],
            torch.ones_like(logps),
        )
        expected = metadata["sequence_loss"] + 0.3 * metadata["local_loss"]
        self.assertTrue(torch.equal(metadata["total_loss"], expected))
        self.assertEqual(metadata["sequence_weight"], 1.0)
        self.assertEqual(metadata["local_weight"], 0.3)

    def test_sequence_gradient_covers_nonlocal_real_tokens_but_not_padding(self):
        logps = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float64)
        loss, _ = mc_hybrid_objective(
            logps,
            [1.0, 0.0],
            [[unit(0.2, [0])], []],
            mask,
        )
        loss.backward()
        self.assertNotEqual(float(logps.grad[0, 1]), 0.0)
        self.assertNotEqual(float(logps.grad[0, 2]), 0.0)
        self.assertNotEqual(float(logps.grad[1, 0]), 0.0)
        self.assertNotEqual(float(logps.grad[1, 1]), 0.0)
        self.assertTrue(torch.equal(logps.grad[:, 3], torch.zeros(2, dtype=logps.dtype)))
        self.assertEqual(float(logps.grad[1, 2]), 0.0)

    def test_local_loss_matches_existing_objective_exactly(self):
        values = torch.tensor(
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=torch.float64
        )
        units = [[unit(0.4, [0, 1])], [unit(-0.3, [1, 2])]]
        direct_logps = values.clone().requires_grad_(True)
        hybrid_logps = values.clone().requires_grad_(True)
        mask = torch.ones_like(values)
        direct_loss, direct_metadata = mc_unit_credit_loss(direct_logps, units, mask)
        hybrid_loss, hybrid_metadata = mc_hybrid_objective(
            hybrid_logps, [0.5, 0.5], units, mask
        )
        self.assertTrue(torch.equal(hybrid_metadata["local_loss"], direct_loss))
        nested = hybrid_metadata["local_objective_metadata"]
        self.assertEqual(nested.keys(), direct_metadata.keys())
        self.assertTrue(torch.equal(nested["candidate_losses"], direct_metadata["candidate_losses"]))
        self.assertEqual(nested["active_unit_count"], direct_metadata["active_unit_count"])
        direct_loss.backward()
        hybrid_loss.backward()
        self.assertLess(float(direct_logps.grad[0, 0]), 0.0)
        self.assertLess(float(hybrid_logps.grad[0, 0]), 0.0)
        self.assertTrue(
            torch.allclose(hybrid_logps.grad, 0.3 * direct_logps.grad, rtol=0.0, atol=1e-15)
        )


if __name__ == "__main__":
    unittest.main()
