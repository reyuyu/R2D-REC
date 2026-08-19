import math
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_training_objective import (
    build_token_advantages,
    clipped_grpo_loss_from_logps,
    group_population_advantages,
    standard_sequence_grpo_loss_from_logps,
)


def compiled(records, token_count=6):
    return {"records": records, "token_count": token_count}


def record(kind, indices, included=True):
    return {"kind": kind, "included": included, "masked_token_indices": indices}


class UserGRPOObjectiveTests(unittest.TestCase):
    def test_no_mask_loss_per_token_and_gradient_parity(self):
        torch.manual_seed(20260819)
        current_a = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)
        current_b = current_a.detach().clone().requires_grad_(True)
        old = torch.randn(8, 7, dtype=torch.float64)
        advantages = torch.randn(8, dtype=torch.float64)
        mask = (torch.rand(8, 7) > 0.2).to(torch.float64)
        standard, standard_tokens, _ = standard_sequence_grpo_loss_from_logps(
            current_a, old, advantages, mask
        )
        token_advantages = advantages[:, None].expand_as(current_b)
        local, local_tokens, _ = clipped_grpo_loss_from_logps(
            current_b, old, token_advantages, mask
        )
        standard.backward()
        local.backward()
        self.assertLessEqual(abs(float(standard - local)), 1e-15)
        self.assertLessEqual(float((standard_tokens - local_tokens).abs().max()), 1e-15)
        self.assertLessEqual(float((current_a.grad - current_b.grad).abs().max()), 1e-15)

    def test_positive_sequence_local_negative_gradient_direction(self):
        current = torch.zeros(1, 4, requires_grad=True)
        old = torch.zeros_like(current)
        token_advantages = torch.tensor([[0.8, -0.5, 0.8, -0.25]])
        loss, _, _ = clipped_grpo_loss_from_logps(
            current, old, token_advantages, torch.ones_like(current)
        )
        loss.backward()
        self.assertLess(float(current.grad[0, 0]), 0.0)
        self.assertGreater(float(current.grad[0, 1]), 0.0)
        self.assertLess(float(current.grad[0, 2]), 0.0)
        self.assertGreater(float(current.grad[0, 3]), 0.0)

    def test_local_penalty_changes_only_masked_token_gradients(self):
        old = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float64)
        sequence_advantage = torch.tensor([0.8], dtype=torch.float64)
        token_advantage = torch.tensor([[0.8, -0.5, 0.8, -0.25]], dtype=torch.float64)
        mask = torch.ones_like(old)
        off_current = old.clone().requires_grad_(True)
        on_current = old.clone().requires_grad_(True)
        off_loss, _, _ = standard_sequence_grpo_loss_from_logps(
            off_current, old, sequence_advantage, mask
        )
        on_loss, _, _ = clipped_grpo_loss_from_logps(
            on_current, old, token_advantage, mask
        )
        off_loss.backward()
        on_loss.backward()
        local_mask = torch.tensor([[False, True, False, True]])
        self.assertTrue(torch.equal(off_current.grad[~local_mask], on_current.grad[~local_mask]))
        self.assertTrue(torch.all(off_current.grad[local_mask] != on_current.grad[local_mask]))

    def test_negative_advantage_is_never_weakened(self):
        output, _ = build_token_advantages(
            torch.tensor([-1.2]),
            [compiled([record("duplicate_sid", [1, 2, 3, 4])])],
            ["action"],
            [6],
        )
        self.assertTrue(torch.equal(output, torch.full((1, 6), -1.2)))

    def test_zero_std_local_signal_and_no_violation_control(self):
        advantages, _, stds = group_population_advantages(
            [0.4] * 8, ["a"] * 4 + ["b"] * 4
        )
        self.assertTrue(torch.equal(stds, torch.zeros(2)))
        penalties = [compiled([]) for _ in range(8)]
        penalties[0] = compiled([record("hallucinated_sid", [2])])
        output, metadata = build_token_advantages(
            advantages, penalties, ["action"] * 8, [6] * 8
        )
        self.assertEqual(float(output[0, 2]), -0.5)
        self.assertEqual(float(output[0, 1]), 0.0)
        self.assertTrue(torch.equal(output[1:], torch.zeros(7, 6)))
        self.assertTrue(metadata["local_mask"][0, 2])

    def test_sqrt_normalization_exact_values(self):
        for length, expected in ((1, 0.5), (4, 0.25), (10, 0.5 / math.sqrt(10)), (100, 0.05)):
            output, metadata = build_token_advantages(
                torch.tensor([1.0]),
                [compiled([record("duplicate_event", list(range(length)))], token_count=length)],
                ["chain"],
                [length],
            )
            self.assertTrue(torch.allclose(output, torch.full((1, length), -expected)))
            self.assertTrue(
                torch.allclose(
                    metadata["effective_penalties"], torch.full((1, length), expected)
                )
            )

    def test_overlap_uses_strongest_only(self):
        output, metadata = build_token_advantages(
            torch.tensor([1.0]),
            [
                compiled(
                    [
                        record("hallucinated_sid", [1]),
                        record("duplicate_sid", [1, 2, 3, 4]),
                    ]
                )
            ],
            ["action"],
            [6],
        )
        self.assertAlmostEqual(float(metadata["effective_penalties"][0, 1]), 0.5)
        self.assertAlmostEqual(float(output[0, 1]), -0.5)
        self.assertAlmostEqual(float(output[0, 2]), -0.25)

    def test_wrong_selection_does_not_mask(self):
        output, metadata = build_token_advantages(
            torch.tensor([0.7]),
            [compiled([record("wrong_selection_sid", [1, 2], included=False)])],
            ["action"],
            [6],
        )
        self.assertTrue(torch.equal(output, torch.full((1, 6), 0.7)))
        self.assertFalse(bool(metadata["local_mask"].any()))

    def test_group_normalization_never_crosses_prompt(self):
        advantages, means, stds = group_population_advantages(
            [0, 1, 2, 3, 10, 11, 12, 13], ["a"] * 4 + ["b"] * 4
        )
        self.assertTrue(torch.allclose(means, torch.tensor([1.5, 11.5])))
        self.assertTrue(torch.allclose(stds, torch.tensor([math.sqrt(1.25)] * 2)))
        self.assertTrue(torch.allclose(advantages[:4], advantages[4:]))
        with self.assertRaises(ValueError):
            group_population_advantages([0, 1, 2, 3], ["a", "a", "b", "b"])


if __name__ == "__main__":
    unittest.main()
