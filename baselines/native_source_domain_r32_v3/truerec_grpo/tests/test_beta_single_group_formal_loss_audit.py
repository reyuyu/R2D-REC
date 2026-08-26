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

from batch_collator_v1 import PaddedBusinessGroup
from beta_old_logp_rescore_validation import TRAINER_SHA256
from beta_single_group_formal_loss_audit import CurrentLogpCapturePolicy, MIN_FREE_GIB


class Output:
    def __init__(self, logits): self.logits = logits


class Policy(torch.nn.Module):
    def __init__(self, logits): super().__init__(); self.anchor = torch.nn.Parameter(torch.tensor(0.0)); self.logits = logits
    def forward(self, input_ids, attention_mask): return Output(self.logits + self.anchor * 0.0)


def batch():
    input_ids = torch.tensor([[10, 11, 1, 2, 3]] * 8)
    return PaddedBusinessGroup(input_ids, torch.ones_like(input_ids, dtype=torch.bool), torch.tensor([[1, 2, 3]] * 8), torch.ones(8, 3, dtype=torch.bool), torch.zeros(8, 3), torch.tensor([[2, 3, 4]] * 8), torch.tensor([[1, 2, 3]] * 8), torch.tensor([2] * 8), "right", 0)


class SingleGroupFormalLossAuditTest(unittest.TestCase):
    def test_01_capture_wrapper_keeps_only_action_logps(self):
        torch.manual_seed(1201)
        value = batch(); logits = torch.randn(8, 5, 20, requires_grad=True)
        wrapper = CurrentLogpCapturePolicy(Policy(logits), value)
        output = wrapper(value.input_ids, value.attention_mask)
        self.assertEqual(output.logits.shape, logits.shape)
        self.assertEqual(wrapper.current_logps.shape, (8, 3))
        self.assertFalse(wrapper.current_logps.requires_grad)
        self.assertEqual(wrapper.forward_calls, 1)
        self.assertFalse(hasattr(wrapper, "logits"))

    def test_02_free_memory_gate_is_fixed(self):
        self.assertEqual(MIN_FREE_GIB, 70.0)

    def test_02b_capture_wrapper_supports_trainer_microbatches(self):
        torch.manual_seed(1202)
        value = batch()
        wrapper = CurrentLogpCapturePolicy(Policy(torch.randn(2, 5, 20, requires_grad=True)), value)
        for start in range(0, 8, 2):
            wrapper(value.input_ids[start:start + 2], value.attention_mask[start:start + 2])
        self.assertEqual(wrapper.current_logps.shape, (8, 3))
        self.assertEqual(wrapper.forward_calls, 4)

    def test_03_trainer_core_sha_unchanged(self):
        blob = subprocess.check_output(["git", "-C", str(ROOT.parents[2]), "show", "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)

    def test_04_no_forbidden_calls(self):
        source = (ROOT / "diagnostics" / "beta_single_group_formal_loss_audit.py").read_text(encoding="utf-8")
        self.assertNotIn(".generate(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)


if __name__ == "__main__":
    unittest.main()
