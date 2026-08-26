"""CPU contracts for D1 persistence and diagnostic-only scoring."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from policy_scoring_v1 import SCORING_MICROBATCH_SIZE, score_full_sequences  # noqa: E402
from production_memory_hardening_d1 import (  # noqa: E402
    atomic_write_json, build_pre_step_components, classify_root_cause, run, token_parity_rows,
)


class BatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.randn(8, 13)); self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(int(input_ids.shape[0]))
        return SimpleNamespace(logits=self.weight[input_ids])


class D1ContractTests(unittest.TestCase):
    def test_formal_default_mb2_and_explicit_diagnostic_mb1(self):
        completions = [[3, 4, 5] for _ in range(8)]
        default = BatchPolicy().eval()
        score_full_sequences(default, [1, 2], completions, 0, "cpu", grad_enabled=False)
        diagnostic = BatchPolicy().eval()
        score_full_sequences(diagnostic, [1, 2], completions, 0, "cpu", grad_enabled=False, scoring_microbatch_size=1)
        self.assertEqual(SCORING_MICROBATCH_SIZE, 2)
        self.assertEqual(default.batch_sizes, [2, 2, 2, 2])
        self.assertEqual(diagnostic.batch_sizes, [1] * 8)

    def test_components_written_before_failure_and_failed_subgates_exact(self):
        trainer = SimpleNamespace(
            physical_policy_forward_calls=8, streaming_backward_calls=8,
            streaming_full_g8_plan_builds=1, hpr_extra_forward_calls=0,
        )
        streamed = SimpleNamespace(frontier_value=1.0, hpr_value_raw=2.0, hpr_value_weighted=0.04, total_value=1.04)
        gradients = {
            "lora_params_with_grad": 1, "lora_params_with_nonzero_grad": 1, "lora_grad_norm": 1.0,
            "base_params_with_grad": 0, "nan_grad_count": 0, "inf_grad_count": 0,
        }
        components = build_pre_step_components(
            {"abs_mean": 0.01, "abs_max": 0.02, "ratio_mean": 1.0, "ratio_min": 0.9, "ratio_max": 1.1},
            trainer, streamed, gradients,
            {"free_memory_at_start_gb": 79.0, "peak_allocated_gb": 1.0, "peak_reserved_gb": 2.0,
             "allocated_after_backward_gb": 1.0, "reserved_after_backward_gb": 2.0},
        )
        self.assertEqual(components["FAILED_SUBGATES"], ["GATE_RATIO_PASS"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre_step_components.json"
            atomic_write_json(path, components)
            self.assertEqual(json.loads(path.read_text())["FAILED_SUBGATES"], ["GATE_RATIO_PASS"])

    def test_token_rows_are_exactly_24(self):
        candidates = tuple(
            SimpleNamespace(completion_ids=(1, 2, 3), old_logps=(-1.0, -2.0, -3.0)) for _ in range(8)
        )
        rows = token_parity_rows(SimpleNamespace(candidates=candidates), torch.tensor([[-1.0, -2.0, -3.0]] * 8))
        self.assertEqual(len(rows), 24)
        self.assertEqual((rows[23]["candidate_index"], rows[23]["action_position"]), (7, 2))

    def test_batch_shape_classification_and_no_optimizer_step(self):
        components = {
            "GATE_RATIO_PASS": False, "GATE_CALL_COUNTS_PASS": True,
            "GATE_LOSS_PASS": True, "GATE_GRADIENT_PASS": True,
            "FAILED_SUBGATES": ["GATE_RATIO_PASS"],
        }
        diagnostic = {
            "current_mb1_vs_diagnostic_old_mb1": {"abs_max": 0.0},
            "formal_old_mb2_vs_diagnostic_old_mb1": {"abs_max": 0.01},
        }
        self.assertEqual(classify_root_cause(components, diagnostic), "CONFIRMED_BF16_BATCH_SHAPE_SCORING_PARITY")
        self.assertNotIn(".step(", inspect.getsource(run))

    def test_diagnostic_scorer_cannot_replace_formal_old_logps(self):
        old = ((-1.0, -2.0, -3.0),) * 8
        candidates = tuple(SimpleNamespace(old_logps=row) for row in old)
        diagnostic = BatchPolicy().eval()
        score_full_sequences(diagnostic, [1, 2], [[3, 4, 5]] * 8, 0, "cpu", grad_enabled=False, scoring_microbatch_size=1)
        self.assertEqual(tuple(candidate.old_logps for candidate in candidates), old)


if __name__ == "__main__":
    unittest.main()
