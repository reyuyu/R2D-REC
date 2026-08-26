from pathlib import Path
import hashlib
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from batch_collator_v1 import PaddedBusinessGroup
from beta_old_logp_rescore_validation import TRAINER_SHA256
from beta_single_group_loss_audit import lora_parameter_sha
from beta_streaming_backward_audit import DetachedCurrentLogpCapture, audit_backward_returns, gradient_audit


class Output:
    def __init__(self, logits):
        self.logits = logits


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.base = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        vocab = torch.arange(20, dtype=torch.float32).view(1, 1, 20)
        return Output(vocab.expand(input_ids.shape[0], input_ids.shape[1], 20) + self.lora_a.sum() * 0.01)


def padded_batch():
    input_ids = torch.tensor([[10, 11, 1, 2, 3]] * 8)
    return PaddedBusinessGroup(
        input_ids, torch.ones_like(input_ids, dtype=torch.bool), torch.tensor([[1, 2, 3]] * 8),
        torch.ones(8, 3, dtype=torch.bool), torch.zeros(8, 3), torch.tensor([[2, 3, 4]] * 8),
        torch.tensor([[1, 2, 3]] * 8), torch.tensor([2] * 8), "right", 0,
    )


class BetaStreamingBackwardAuditTest(unittest.TestCase):
    def test_detached_logp_capture_preserves_four_chunks(self):
        value = padded_batch()
        wrapper = DetachedCurrentLogpCapture(Policy(), value)
        for start in range(0, 8, 2):
            output = wrapper(value.input_ids[start:start + 2], value.attention_mask[start:start + 2])
            self.assertEqual(output.logits.shape[:2], (2, 5))
        self.assertEqual(wrapper.current_logps.shape, (8, 3))
        self.assertFalse(wrapper.current_logps.requires_grad)
        self.assertEqual(wrapper.forward_calls, 4)

    def test_backward_return_audit_does_not_change_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        calls = []
        with audit_backward_returns(lambda: calls.append(1)):
            for multiplier in (1.0, 2.0, 3.0, 4.0):
                (parameter * multiplier).backward()
        self.assertEqual((len(calls), float(parameter.grad)), (4, 10.0))

    def test_gradient_and_lora_sha_audits(self):
        model = Policy()
        before, count = lora_parameter_sha(model)
        (model.lora_a.square().sum()).backward()
        audit = gradient_audit(model)
        after, after_count = lora_parameter_sha(model)
        self.assertEqual((before, count), (after, after_count))
        self.assertGreater(audit["lora_params_with_nonzero_grad"], 0)
        self.assertGreater(audit["lora_grad_norm"], 0)
        self.assertEqual(audit["base_params_with_grad"], 0)
        self.assertEqual((audit["nan_grad_count"], audit["inf_grad_count"]), (0, 0))

    def test_runner_has_no_forbidden_actions(self):
        source = (ROOT / "diagnostics" / "beta_streaming_backward_audit.py").read_text(encoding="utf-8")
        self.assertNotIn(".generate(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("compute_group(", source)
        self.assertNotIn("save_pretrained(", source)

    def test_trainer_core_sha_unchanged(self):
        blob = subprocess.check_output([
            "git", "-C", str(ROOT.parents[2]), "show",
            "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py",
        ])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)


if __name__ == "__main__":
    unittest.main()
