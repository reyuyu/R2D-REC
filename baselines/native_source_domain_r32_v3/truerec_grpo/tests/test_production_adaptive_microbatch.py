"""CPU contract tests for deterministic production microbatch selection."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "trainer", ROOT / "diagnostics", ROOT / "data", ROOT / "initialization"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from run_truerec_pilot_v1 import (  # noqa: E402
    ProductionTrainingError, apply_group_scoring_microbatch,
    require_matched_scoring_microbatch, select_streaming_microbatch_size,
)
from training_driver_v1 import frozen_contract  # noqa: E402
from truerec_grpo_trainer_v1 import TrueRecGRPOTrainerV1  # noqa: E402


class AdaptiveMicrobatchContractTests(unittest.TestCase):
    def test_threshold_selection_is_deterministic(self):
        self.assertEqual(select_streaming_microbatch_size(2200), 2)
        self.assertEqual(select_streaming_microbatch_size(2201), 1)
        self.assertEqual(select_streaming_microbatch_size(3489), 1)
        with self.assertRaises(ProductionTrainingError):
            select_streaming_microbatch_size(0)

    def test_frozen_contract_contains_policy(self):
        contract = frozen_contract()
        self.assertEqual(contract["trainer_microbatch_size"], 2)
        self.assertIs(contract["adaptive_streaming_microbatch"], True)
        self.assertEqual(contract["long_context_threshold_tokens"], 2200)
        self.assertEqual(contract["long_context_microbatch_size"], 1)
        self.assertIs(contract["old_rescore_matches_streaming_microbatch"], True)

    def test_trainer_instance_accepts_only_one_or_two(self):
        trainer = TrueRecGRPOTrainerV1(object(), lambda _: 0, 0, streaming_microbatch_size=1)
        self.assertEqual(trainer.streaming_microbatch_size, 1)
        trainer.set_streaming_microbatch_size(2)
        self.assertEqual(trainer.streaming_microbatch_size, 2)
        with self.assertRaises(ValueError):
            trainer.set_streaming_microbatch_size(4)

    def test_one_selection_configures_matching_normal_and_long_paths(self):
        trainer = TrueRecGRPOTrainerV1(object(), lambda _: 0, 0)
        normal = apply_group_scoring_microbatch(trainer, 2200)
        self.assertEqual((normal, trainer.streaming_microbatch_size), (2, 2))
        require_matched_scoring_microbatch(normal, trainer)
        long = apply_group_scoring_microbatch(trainer, 2201)
        self.assertEqual((long, trainer.streaming_microbatch_size), (1, 1))
        require_matched_scoring_microbatch(long, trainer)
        with self.assertRaises(ProductionTrainingError):
            require_matched_scoring_microbatch(2, trainer)


if __name__ == "__main__":
    unittest.main()
