from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from trainer_streaming_backward_equivalence import run_audit


class TrainerStreamingBackwardEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = run_audit()

    def test_required_case_equivalence(self):
        self.assertEqual(set(self.audit["cases"]), {"HPR_A", "HPR_B", "HPR_C", "HPR_NONE", "FORMAT_INVALID"})
        self.assertTrue(all(case["status"] == "PASS" for case in self.audit["cases"].values()))

    def test_values_and_gradients(self):
        self.assertTrue(self.audit["frontier_value_equivalent"])
        self.assertTrue(self.audit["hpr_value_equivalent"])
        self.assertTrue(self.audit["total_value_equivalent"])
        self.assertTrue(self.audit["gradient_equivalent"])
        self.assertTrue(all(case["reported_loss_composition_closed"] for case in self.audit["cases"].values()))

    def test_physical_execution_counts(self):
        self.assertEqual(self.audit["physical_forward_calls_per_group"], 4)
        self.assertEqual(self.audit["physical_backward_calls_per_group"], 4)
        self.assertTrue(self.audit["full_g8_plan_built_once"])

    def test_graph_lifetime(self):
        self.assertFalse(self.audit["graph_bearing_state_retained_across_chunks"])

    def test_frozen_contracts(self):
        self.assertEqual(self.audit["G"], 8)
        self.assertEqual(self.audit["trainer_microbatch_size"], 2)
        self.assertEqual(self.audit["ppo_old_logp_source"], "FULL_FORWARD_RESCORE")
        self.assertEqual(self.audit["hpr_lambda"], 0.02)
        self.assertEqual(self.audit["execution"]["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
