from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from trainer_g8_microbatch_equivalence import run_audit


class TrainerG8MicrobatchEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = run_audit()

    def test_all_required_cases(self):
        self.assertEqual(set(self.audit["cases"]), {"HPR_A", "HPR_B", "HPR_C", "HPR_NONE", "FORMAT_INVALID"})
        self.assertTrue(all(case["status"] == "PASS" for case in self.audit["cases"].values()))

    def test_loss_equivalence(self):
        self.assertTrue(self.audit["frontier_loss_equivalent"])
        self.assertTrue(self.audit["hpr_loss_equivalent"])
        self.assertTrue(self.audit["total_loss_equivalent"])

    def test_gradient_equivalence(self):
        self.assertTrue(self.audit["gradient_equivalence"])

    def test_forward_counts(self):
        self.assertEqual(self.audit["logical_policy_scoring_passes_per_group"], 1)
        self.assertEqual(self.audit["physical_policy_forward_calls_per_group"], 4)
        self.assertEqual(self.audit["hpr_extra_forward_calls"], 0)

    def test_frozen_contracts(self):
        self.assertEqual(self.audit["ppo_old_logp_source"], "FULL_FORWARD_RESCORE")
        self.assertEqual(self.audit["hpr_lambda"], 0.02)
        self.assertTrue(self.audit["business_group_unit_preserved"])


if __name__ == "__main__":
    unittest.main()
