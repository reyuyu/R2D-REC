import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from diagnose_user_chain import (  # noqa: E402
    _allocate_probe_targets,
    _classification,
    _grounding_label,
    bootstrap_mean_ci,
    largest_remainder,
    paired_deltas,
    quartile_for,
)


class UserChainDiagnosisTest(unittest.TestCase):
    def test_largest_remainder_matches_train_chain_distribution(self):
        self.assertEqual(
            largest_remainder({"2": 225, "3": 825, "4": 375, "5": 75}, 40),
            {"2": 6, "3": 22, "4": 10, "5": 2},
        )
        self.assertEqual(sum(largest_remainder({"native": 1200, "converted": 300}, 40).values()), 40)

    def test_joint_targets_preserve_event_and_source_margins(self):
        rows = []
        for event, count in ((2, 225), (3, 825), (4, 375), (5, 75)):
            for index in range(count):
                rows.append({
                    "gold_event_count": event,
                    "converted_from_cot": index < count // 5,
                    "prompt_token_count": index + event * 1000,
                })
        targets = _allocate_probe_targets(rows, 40)
        event_margin = collections.Counter()
        source_margin = collections.Counter()
        for (event, source, _quartile), count in targets.items():
            event_margin[event] += count
            source_margin[source] += count
        self.assertEqual(dict(event_margin), {"2": 6, "3": 22, "4": 10, "5": 2})
        self.assertEqual(dict(source_margin), {"converted_cot": 8, "native_nocot": 32})

    def test_quartile_boundaries_are_deterministic(self):
        boundaries = [10.0, 20.0, 30.0]
        self.assertEqual(quartile_for(10, boundaries), "Q1")
        self.assertEqual(quartile_for(11, boundaries), "Q2")
        self.assertEqual(quartile_for(30, boundaries), "Q3")
        self.assertEqual(quartile_for(31, boundaries), "Q4")

    def test_paired_delta_counts_and_quantiles(self):
        def row(value):
            return {
                "total_reward": value,
                "action_alignment": value * 2,
                "logic_alignment": 0.0,
            }
        groups = {
            "C0": {"a": row(0.5), "b": row(0.5), "c": row(0.5)},
            "C40": {"a": row(0.7), "b": row(0.505), "c": row(0.2)},
        }
        result = paired_deltas(groups, "C0", "C40")
        self.assertEqual(result["total_reward"]["improved"], 1)
        self.assertEqual(result["total_reward"]["unchanged"], 1)
        self.assertEqual(result["total_reward"]["degraded"], 1)
        self.assertAlmostEqual(result["total_reward"]["mean"], (-0.095) / 3)

    def test_bootstrap_is_repeatable(self):
        left = bootstrap_mean_ci([-0.2, -0.1, 0.0, 0.1], samples=500)
        right = bootstrap_mean_ci([-0.2, -0.1, 0.0, 0.1], samples=500)
        self.assertEqual(left, right)
        self.assertLessEqual(left[0], left[1])

    def test_grounding_label(self):
        self.assertEqual(_grounding_label([{"status": "grounded"}]), "grounded")
        self.assertEqual(_grounding_label([{"status": "grounded"}, {"status": "ungrounded"}]), "partially_grounded")
        self.assertEqual(_grounding_label([]), "ungrounded")

    def test_classification_rules(self):
        confirmed = {"total_reward": {"bootstrap_95pct_ci": [-0.05, -0.01], "degraded": 25, "count": 40, "mean": -0.03}}
        noise = {"total_reward": {"bootstrap_95pct_ci": [-0.02, 0.02], "degraded": 20, "count": 40, "mean": -0.002}}
        unclear = {"total_reward": {"bootstrap_95pct_ci": [-0.05, 0.01], "degraded": 30, "count": 40, "mean": -0.02}}
        self.assertEqual(_classification(confirmed), "CONFIRMED_CHAIN_DEGRADATION")
        self.assertEqual(_classification(noise), "SMALL_PROBE_NOISE")
        self.assertEqual(_classification(unclear), "INCONCLUSIVE")

    def test_script_contains_no_training_primitive(self):
        source = (SCRIPTS / "diagnose_user_chain.py").read_text(encoding="utf-8")
        for forbidden in ("optimizer.step(", ".backward(", "trainer.train("):
            self.assertNotIn(forbidden, source)
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("model.requires_grad_(False)", source)


if __name__ == "__main__":
    unittest.main()
