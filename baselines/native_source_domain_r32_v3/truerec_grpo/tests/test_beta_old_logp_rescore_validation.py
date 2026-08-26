from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from beta_old_logp_rescore_validation import TRAINER_SHA256, causal_alignment_pass, comparison_stats, independent_direct_gather, position_discrepancy
from batch_collator_v1 import PaddedBusinessGroup
from truerec_grpo_trainer_v1 import gather_padded_action_logps


def batch():
    input_ids = torch.tensor([[10, 11, 1, 2, 3]] * 8)
    return PaddedBusinessGroup(input_ids, torch.ones_like(input_ids, dtype=torch.bool), torch.tensor([[1, 2, 3]] * 8), torch.ones(8, 3, dtype=torch.bool), torch.zeros(8, 3), torch.tensor([[2, 3, 4]] * 8), torch.tensor([[1, 2, 3]] * 8), torch.tensor([2] * 8), "right", 0)


class BetaOldLogpRescoreValidationTest(unittest.TestCase):
    def test_01_repeat_comparison_identity(self):
        values = torch.randn(8, 3); result = comparison_stats(values, values.clone(), torch.ones(8, 3, dtype=torch.bool)); self.assertEqual((result["abs_max"], result["ratio_mean"]), (0.0, 1.0))
    def test_02_position_discrepancy(self):
        first = torch.zeros(8, 3); second = torch.tensor([[1.0, 2.0, 3.0]] * 8); result = position_discrepancy(first, second, torch.ones(8, 3, dtype=torch.bool)); self.assertEqual((result["A"]["abs_max"], result["B"]["abs_max"], result["C"]["abs_max"]), (1.0, 2.0, 3.0))
    def test_03_causal_alignment(self): self.assertTrue(causal_alignment_pass(batch()))
    def test_04_direct_gather_matches_helper(self):
        torch.manual_seed(14); logits = torch.randn(8, 5, 20); value = batch(); self.assertTrue(torch.equal(independent_direct_gather(logits, value), gather_padded_action_logps(logits, value)))
    def test_05_trainer_sha_unchanged(self):
        blob = subprocess.check_output(["git", "-C", str(ROOT.parents[2]), "show", "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)
    def test_06_repeat_threshold_fixed(self):
        result = comparison_stats(torch.zeros(8, 3), torch.full((8, 3), 2e-6), torch.ones(8, 3, dtype=torch.bool)); self.assertGreater(result["abs_max"], 1e-6)


if __name__ == "__main__": unittest.main()
