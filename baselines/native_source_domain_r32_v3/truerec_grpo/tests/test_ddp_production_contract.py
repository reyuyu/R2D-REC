"""CPU/static gates for the four-rank global-G8 production path."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "diagnostics", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from checkpoint_ddp_v1 import DistributedCheckpointError, validate_world_size  # noqa: E402
from distributed_trainer_v1 import DDP_WORLD_SIZE, LOCAL_G, DistributedTrueRecGRPOTrainerV1  # noqa: E402
from policy_scoring_v1 import score_full_sequences  # noqa: E402
from run_truerec_pilot_ddp_v1 import distributed_restore_gate, run_one_group  # noqa: E402
from training_driver_v1 import frozen_contract  # noqa: E402


class BatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.randn(8, 16)); self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(int(input_ids.shape[0]))
        return SimpleNamespace(logits=self.weight[input_ids])


class DDPProductionContractTests(unittest.TestCase):
    def test_world_and_business_semantics_are_frozen(self):
        self.assertEqual((DDP_WORLD_SIZE, LOCAL_G), (4, 2))
        contract = frozen_contract()
        self.assertEqual(contract["production_world_size"], 4)
        self.assertEqual(contract["local_g_per_rank"], 2)
        self.assertIs(contract["ddp_world_size_loss_scaling"], True)

    def test_local_g2_formal_scorer_honors_matched_microbatch(self):
        completions = [[3, 4, 5], [3, 4, 5]]
        mb2 = BatchPolicy().eval()
        score_full_sequences(mb2, [1, 2], completions, 0, "cpu", grad_enabled=False, scoring_microbatch_size=2)
        mb1 = BatchPolicy().eval()
        score_full_sequences(mb1, [1, 2], completions, 0, "cpu", grad_enabled=False, scoring_microbatch_size=1)
        self.assertEqual(mb2.batch_sizes, [2])
        self.assertEqual(mb1.batch_sizes, [1, 1])

    def test_single_gpu_checkpoint_world_size_is_rejected(self):
        validate_world_size(4)
        with self.assertRaises(DistributedCheckpointError):
            validate_world_size(1)

    def test_distributed_restore_gate_reads_nested_driver_state(self):
        restored = {"driver_state": {"global_step": 1}, "next_group_index": 1}
        ranks = [
            {
                "model_restore_exact": True,
                "rng_restored": {"python": True, "numpy": True, "torch_cpu": True, "torch_cuda": True},
                "optimizer_step": 1,
            }
            for _ in range(4)
        ]
        self.assertTrue(distributed_restore_gate(restored, ranks))
        ranks[2]["optimizer_step"] = 0
        self.assertFalse(distributed_restore_gate(restored, ranks))

    def test_world_size_scaling_and_one_selection_are_mechanical(self):
        trainer_source = inspect.getsource(DistributedTrueRecGRPOTrainerV1.backward_global_group)
        runner_source = inspect.getsource(run_one_group)
        self.assertIn("(world_size * local_total).backward()", trainer_source)
        self.assertIn("((stop - start) / G)", trainer_source)
        self.assertIn("scoring_microbatch_size=selected_mb", runner_source)
        self.assertIn("streaming_microbatch_size=selected_mb", runner_source)
        self.assertIn('local_generation_kwargs["num_return_sequences"] = LOCAL_G', runner_source)


if __name__ == "__main__":
    unittest.main()
