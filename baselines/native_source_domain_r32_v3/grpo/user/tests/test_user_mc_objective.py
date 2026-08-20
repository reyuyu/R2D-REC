import sys
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_objective import MCObjectiveError, mc_unit_credit_loss  # noqa: E402


def unit(delta, indices):
    return {"delta": delta, "generated_token_indices": list(indices)}


class MCUnitCreditObjectiveTests(unittest.TestCase):
    def test_positive_unit_gradient_increases_log_probability(self):
        logps = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps, [[unit(0.5, [1, 2])]], torch.ones_like(logps)
        )
        loss.backward()
        self.assertLess(float(logps.grad[0, 1]), 0.0)
        self.assertLess(float(logps.grad[0, 2]), 0.0)
        self.assertEqual(metadata["positive_unit_count"], 1)

    def test_negative_unit_gradient_decreases_log_probability(self):
        logps = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps, [[unit(-0.5, [1, 2])]], torch.ones_like(logps)
        )
        loss.backward()
        self.assertGreater(float(logps.grad[0, 1]), 0.0)
        self.assertGreater(float(logps.grad[0, 2]), 0.0)
        self.assertEqual(metadata["negative_unit_count"], 1)

    def test_zero_credit_has_strictly_zero_direct_gradient(self):
        logps = torch.randn((1, 4), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps, [[unit(0.0, [1, 2])]], torch.ones_like(logps)
        )
        loss.backward()
        self.assertTrue(torch.equal(logps.grad, torch.zeros_like(logps)))
        self.assertEqual(metadata["zero_unit_count"], 1)
        self.assertEqual(metadata["active_unit_count"], 0)

    def test_uncredited_tokens_have_strictly_zero_gradient(self):
        logps = torch.randn((1, 5), dtype=torch.float64, requires_grad=True)
        loss, _ = mc_unit_credit_loss(
            logps, [[unit(0.5, [1, 3])]], torch.ones_like(logps)
        )
        loss.backward()
        self.assertTrue(torch.equal(logps.grad[0, [0, 2, 4]], torch.zeros(3, dtype=logps.dtype)))

    def test_span_mean_normalizes_four_and_forty_token_units(self):
        logps = torch.full((1, 44), -2.0, dtype=torch.float64, requires_grad=True)
        units = [[unit(0.5, range(4)), unit(0.5, range(4, 44))]]
        loss, metadata = mc_unit_credit_loss(logps, units, torch.ones_like(logps))
        records = metadata["unit_records"]
        self.assertEqual(float(records[0]["unit_loss"]), 1.0)
        self.assertEqual(float(records[1]["unit_loss"]), 1.0)
        loss.backward()
        short_gradient = abs(float(logps.grad[0, 0]))
        long_gradient = abs(float(logps.grad[0, 4]))
        self.assertAlmostEqual(short_gradient / long_gradient, 10.0, 14)
        self.assertAlmostEqual(float(logps.grad[0, :4].abs().sum()), 0.5, 14)
        self.assertAlmostEqual(float(logps.grad[0, 4:].abs().sum()), 0.5, 14)

    def test_candidate_reduction_is_sum_and_does_not_rescale_existing_units(self):
        first = torch.tensor([[-2.0, -3.0, -4.0]], dtype=torch.float64, requires_grad=True)
        first_loss, first_metadata = mc_unit_credit_loss(
            first,
            [[unit(0.3, [0]), unit(0.2, [1])]],
            torch.ones_like(first),
        )
        first_loss.backward()
        first_gradient = first.grad.detach().clone()
        self.assertAlmostEqual(float(first_metadata["candidate_losses"][0]), 1.2, 14)

        second = first.detach().clone().requires_grad_(True)
        second_loss, second_metadata = mc_unit_credit_loss(
            second,
            [[unit(0.3, [0]), unit(0.2, [1]), unit(0.1, [2])]],
            torch.ones_like(second),
        )
        second_loss.backward()
        self.assertAlmostEqual(float(second_metadata["candidate_losses"][0]), 1.6, 14)
        self.assertTrue(torch.equal(first_gradient[0, :2], second.grad[0, :2]))

    def test_batch_reduction_is_mean_and_empty_candidate_stays_in_denominator(self):
        logps = torch.tensor([[-2.0, 0.0], [7.0, 8.0]], dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps,
            [[unit(0.5, [0])], []],
            torch.ones_like(logps),
        )
        self.assertEqual(float(metadata["candidate_losses"][0]), 1.0)
        self.assertEqual(float(metadata["candidate_losses"][1]), 0.0)
        self.assertEqual(float(loss), 0.5)
        loss.backward()
        self.assertEqual(float(logps.grad[0, 0]), -0.25)
        self.assertTrue(torch.equal(logps.grad[1], torch.zeros_like(logps.grad[1])))

    def test_all_empty_candidates_keep_zero_loss_connected_for_backward(self):
        logps = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(logps, [[], []], torch.ones_like(logps))
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(metadata["candidate_losses"], torch.zeros(2, dtype=logps.dtype)))
        loss.backward()
        self.assertTrue(torch.equal(logps.grad, torch.zeros_like(logps)))

    def test_same_sign_overlap_gradients_add_algebraically(self):
        logps = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps,
            [[unit(0.4, [0, 1]), unit(0.2, [1, 2])]],
            torch.ones_like(logps),
        )
        loss.backward()
        expected = torch.tensor([[-0.2, -0.3, -0.1]], dtype=torch.float64)
        self.assertTrue(torch.allclose(logps.grad, expected, rtol=0.0, atol=1e-15))
        self.assertEqual(metadata["active_token_count"], 3)
        self.assertEqual(metadata["active_token_assignment_count"], 4)
        self.assertEqual(metadata["overlap_token_count"], 1)
        self.assertEqual(metadata["same_sign_overlap_token_count"], 1)
        self.assertEqual(metadata["mixed_sign_overlap_token_count"], 0)
        self.assertEqual(metadata["max_active_units_per_token"], 2)
        record = metadata["overlap_records"][0]
        self.assertEqual(record["candidate_index"], 0)
        self.assertEqual(record["token_index"], 1)
        self.assertEqual(record["unit_indices"], [0, 1])
        self.assertEqual(record["deltas"], [0.4, 0.2])
        self.assertEqual(record["per_unit_logp_coefficients"], [-0.2, -0.1])
        self.assertAlmostEqual(record["net_logp_coefficient"], -0.3, 15)

    def test_mixed_sign_overlap_gradients_partially_cancel(self):
        logps = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps,
            [[unit(0.4, [0, 1]), unit(-0.2, [1, 2])]],
            torch.ones_like(logps),
        )
        loss.backward()
        expected = torch.tensor([[-0.2, -0.1, 0.1]], dtype=torch.float64)
        self.assertTrue(torch.allclose(logps.grad, expected, rtol=0.0, atol=1e-15))
        self.assertEqual(metadata["same_sign_overlap_token_count"], 0)
        self.assertEqual(metadata["mixed_sign_overlap_token_count"], 1)
        self.assertAlmostEqual(
            metadata["overlap_records"][0]["net_logp_coefficient"], -0.1, 15
        )

    def test_mixed_sign_overlap_exact_cancellation_is_natural(self):
        logps = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        loss, metadata = mc_unit_credit_loss(
            logps,
            [[unit(0.4, [0, 1]), unit(-0.4, [1, 2])]],
            torch.ones_like(logps),
        )
        loss.backward()
        self.assertEqual(float(logps.grad[0, 1]), 0.0)
        self.assertEqual(
            metadata["overlap_records"][0]["net_logp_coefficient"], 0.0
        )

    def test_overlap_loss_equals_independent_unit_sum(self):
        logps = torch.tensor([[-2.0, -3.0, -5.0]], dtype=torch.float64)
        units = [unit(0.4, [0, 1]), unit(-0.2, [1, 2])]
        loss, metadata = mc_unit_credit_loss(
            logps, [units], torch.ones_like(logps)
        )
        expected = -0.4 * logps[0, [0, 1]].mean() + 0.2 * logps[0, [1, 2]].mean()
        self.assertEqual(float(loss), float(expected))
        self.assertEqual(float(metadata["candidate_losses"][0]), float(expected))

    def test_index_and_completion_mask_safety_checks(self):
        logps = torch.zeros((1, 4), dtype=torch.float64)
        cases = [
            (unit(0.5, []), torch.ones_like(logps), "non-empty"),
            (unit(0.5, [1, 1]), torch.ones_like(logps), "duplicates"),
            (unit(0.5, [-1]), torch.ones_like(logps), "outside"),
            (unit(0.5, [4]), torch.ones_like(logps), "outside"),
            (unit(0.5, [2]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), "masked or padded"),
        ]
        for bad_unit, mask, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(MCObjectiveError, message):
                mc_unit_credit_loss(logps, [[bad_unit]], mask)

    def test_metadata_counts_credit_mass_and_active_tokens(self):
        logps = torch.full((2, 5), -2.0, dtype=torch.float64)
        loss, metadata = mc_unit_credit_loss(
            logps,
            [
                [unit(0.3, [0, 1]), unit(-0.2, [2])],
                [unit(0.0, [3, 4])],
            ],
            torch.ones_like(logps),
        )
        self.assertAlmostEqual(float(loss), 0.1, 15)
        self.assertEqual(metadata["positive_unit_count"], 1)
        self.assertEqual(metadata["negative_unit_count"], 1)
        self.assertEqual(metadata["zero_unit_count"], 1)
        self.assertAlmostEqual(metadata["positive_credit_mass"], 0.3)
        self.assertAlmostEqual(metadata["negative_credit_mass"], 0.2)
        self.assertEqual(metadata["active_unit_count"], 2)
        self.assertEqual(metadata["active_token_count"], 3)
        self.assertEqual(metadata["active_token_assignment_count"], 3)
        self.assertEqual(metadata["overlap_token_count"], 0)
        self.assertEqual(metadata["same_sign_overlap_token_count"], 0)
        self.assertEqual(metadata["mixed_sign_overlap_token_count"], 0)
        self.assertEqual(metadata["max_active_units_per_token"], 1)


if __name__ == "__main__":
    unittest.main()
