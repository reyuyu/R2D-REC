"""CPU-only runner, parent, and frozen-contract checks."""

import unittest
from unittest.mock import patch

import torch

from frontier_trainer import NoThinkOnlyFrontierTrainer
from grpo_trl_trainer import ROUTE_ID, ROUTE_LOSS_W
from run_nothink_only_frontier_train import (
    EXPECTED_ADAPTER, EXPECTED_BASE, FORMAL_RUN_ID, RUN_ID_PREFIX,
    _ManifestWriter, formal_checkpoint_steps, validate_experiment_args,
)
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


class Sink:
    def write_manifest(self, payload):
        return payload


class FrontierRunnerTests(unittest.TestCase):
    def test_parent_and_frozen_contract(self):
        self.assertTrue(issubclass(NoThinkOnlyFrontierTrainer, ThinkExactClampRecGRPOTrainer))
        self.assertEqual(ROUTE_LOSS_W["no_think"], 0.5)
        self.assertEqual(formal_checkpoint_steps(1544), (250, 500, 666, 750, 1000, 1250, 1544))
        args = validate_experiment_args(["--run-id", FORMAL_RUN_ID])
        self.assertEqual((args.lr, args.seed), (1e-6, 20260816))
        self.assertTrue(EXPECTED_BASE.endswith("onereason-8b-pretrain-competition"))
        self.assertIn("BATA-BASELINE-R32-2E-GC04-4GPU", EXPECTED_ADAPTER)

    def test_manifest_records_frozen_contract(self):
        manifest = _ManifestWriter(Sink()).write_manifest({})
        self.assertEqual(manifest["parent_experiment"], "GR_REC_NoThinkOnly_Hier_v1")
        self.assertEqual(manifest["training_routes"], ["no_think"])
        self.assertFalse(manifest["training_beam32"])
        self.assertEqual(manifest["nothink_route_multiplier"], 0.5)
        self.assertEqual(manifest["dead_zero_bridge"], {"branch": "gold_A_only", "lambda": 0.02})
        self.assertEqual(manifest["format_violation"]["sequence_advantage_total"], -0.09375)

    def test_bad_args_are_rejected(self):
        for argv in (["--run-id", "wrong"], ["--run-id", FORMAL_RUN_ID, "--resume-from-checkpoint", "/tmp/other/checkpoint-2"]):
            with self.assertRaises(ValueError):
                validate_experiment_args(argv)

    def test_route_guards_reject_think_without_model_work(self):
        trainer = object.__new__(NoThinkOnlyFrontierTrainer)
        with self.assertRaises(RuntimeError):
            trainer._calculate_rewards([{"route": "think"}])
        with self.assertRaises(RuntimeError):
            trainer._compute_loss(None, {"route_id": torch.tensor([ROUTE_ID["think"]])})

    def test_token_representation_is_exclusive_and_length_normalized(self):
        trainer = object.__new__(NoThinkOnlyFrontierTrainer)
        trainer._nothink_bridge_runtime = {
            "format_valid": (False, True),
            "token_credits": ((0.0, 0.0, 0.0, 0.0), (0.0, 0.125, -0.1875, 0.0)),
            "token_positions": (None, (0, 1, 2, 3)),
        }
        output = {
            "completion_ids": torch.zeros((2, 5), dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
            "advantages": torch.zeros(2),
        }
        trainer._attach_nothink_token_advantages(output)
        self.assertAlmostEqual(float(output["token_advantages"][0].sum()), -0.09375)
        self.assertEqual(int(output["sequence_penalty_mask"][0].sum()), 3)
        self.assertEqual(int(output["sequence_penalty_mask"][1].sum()), 0)
        self.assertEqual(
            output["token_advantages"][1].tolist(), [0.0, 0.125, -0.1875, 0.0, 0.0]
        )

    def test_runtime_forces_nonempty_cot_rewards_to_minus_one(self):
        trainer = object.__new__(NoThinkOnlyFrontierTrainer)
        trainer.accelerator = type(
            "Accelerator", (), {"process_index": 0, "num_processes": 1, "device": "cpu"}
        )()
        trainer.reward_weights = torch.tensor([1.0])
        trainer.processing_class = object()
        trainer._build_bridge_runtime = lambda *args: {
            "token_credits": tuple(args[-2]),
            "token_positions": tuple(args[-1]),
        }
        inputs = [{
            "route": "no_think",
            "recommendation_group_id": "g0",
            "all_gold_sids": ["<|prod_begin|><s_a_1><s_b_2><s_c_3>"],
            "target_domain": "prod",
            "prompt": "prompt",
        }] * 8
        completions = [f"<think>reasoning</think><|prod_begin|><s_a_1><s_b_2><s_c_3>"] * 8
        with patch("trl.trainer.grpo_trainer.gather_object", side_effect=lambda value: value):
            rewards = trainer._prepare_frontier_runtime(
                inputs, completions, [[1, 2, 3]] * 8, torch.full((8, 1), 8.0)
            )
        self.assertEqual(rewards[:, 0].tolist(), [-1.0] * 8)
        self.assertEqual(trainer._nothink_bridge_runtime["format_violation_candidate_count"], 8)
        self.assertTrue(all(
            credits == (0.0, 0.0, 0.0, 0.0)
            for credits in trainer._nothink_bridge_runtime["token_credits"]
        ))

    def test_prefix_is_experiment_specific(self):
        self.assertEqual(RUN_ID_PREFIX, "GR-REC-NOTHINK-ONLY-FRONTIER-G8BASE-E1-")


if __name__ == "__main__":
    unittest.main()
