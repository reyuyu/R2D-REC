import inspect
import unittest

from . import gpu_preflight
from .preflight_contract import HARD_CONDITIONS, evaluate_preflight_conditions


class PreflightContractTests(unittest.TestCase):
    def test_all_hard_conditions_pass(self):
        result = evaluate_preflight_conditions({name: True for name in HARD_CONDITIONS})
        self.assertTrue(result["preflight_pass"])
        self.assertEqual(result["failure_reasons"], [])

    def test_one_false_condition_fails_closed(self):
        observations = {name: True for name in HARD_CONDITIONS}
        observations["suffix_action_gradient_nonzero"] = False
        result = evaluate_preflight_conditions(observations)
        self.assertFalse(result["preflight_pass"])
        self.assertEqual(result["failure_reasons"], ["suffix_action_gradient_nonzero"])

    def test_harness_has_no_training_or_update_entry(self):
        source = inspect.getsource(gpu_preflight)
        self.assertNotIn("trainer.train(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("scheduler.step(", source)
        self.assertNotIn("save_model(", source)


if __name__ == "__main__":
    unittest.main()
