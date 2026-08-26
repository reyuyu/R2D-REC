from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from beta_deterministic_policy_mode_validation import (
    EXPECTED_LORA_DROPOUT,
    POLICY_SCORING_MODE,
    SCORING_MICROBATCH_SIZE,
    TRAINER_SHA256,
    audit_dropout_modules,
    dropout_config_values,
    parameter_trainability,
)
from policy_scoring_v1 import graph_connected_to_parameters, score_full_sequences


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 3, bias=False)
        self.lora_A = torch.nn.Linear(3, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 3, bias=False)
        self.dropout = torch.nn.Dropout(0.05)
        self.base.weight.requires_grad_(False)

    def forward(self, value):
        return self.base(value) + self.lora_B(self.dropout(self.lora_A(value)))


class DeterministicPolicyModeValidationTest(unittest.TestCase):
    def test_01_frozen_contract(self):
        self.assertEqual(POLICY_SCORING_MODE, "eval")
        self.assertEqual(EXPECTED_LORA_DROPOUT, 0.05)
        self.assertEqual(SCORING_MICROBATCH_SIZE, 2)

    def test_02_nested_dropout_config(self):
        values = dropout_config_values({"attention_dropout": 0.0, "adapter": {"lora_dropout": 0.05}, "unrelated": 4})
        self.assertEqual(values, {"attention_dropout": 0.0, "adapter.lora_dropout": 0.05})

    def test_03_eval_disables_nonzero_dropout(self):
        model = TinyPolicy()
        model.eval()
        audit = audit_dropout_modules(model)
        self.assertFalse(audit["rl_dropout_active"])
        self.assertEqual(audit["module_count"], 1)
        self.assertFalse(audit["modules"][0]["training"])

    def test_04_train_mode_detects_active_dropout(self):
        audit = audit_dropout_modules(TinyPolicy().train())
        self.assertTrue(audit["rl_dropout_active"])
        self.assertEqual(audit["active_modules"][0]["p"], 0.05)

    def test_05_only_lora_trainable(self):
        counts = parameter_trainability(TinyPolicy())
        self.assertGreater(counts["trainable_lora_param_count"], 0)
        self.assertEqual(counts["base_trainable_param_count"], 0)

    def test_06_graph_reaches_lora_without_backward(self):
        model = TinyPolicy().eval()
        output = model(torch.ones(2, 3))
        lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
        self.assertTrue(output.requires_grad)
        self.assertTrue(graph_connected_to_parameters(output, lora))
        self.assertTrue(all(parameter.grad is None for parameter in lora))

    def test_07_disconnected_graph_rejected(self):
        model = TinyPolicy().eval()
        unrelated = torch.ones(2, 3, requires_grad=True) * 2
        lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
        self.assertFalse(graph_connected_to_parameters(unrelated, lora))

    def test_08_trainer_sha_unchanged(self):
        blob = subprocess.check_output(["git", "-C", str(ROOT.parents[2]), "show", "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)

    def test_09_diagnostic_has_no_forbidden_calls(self):
        source = (ROOT / "diagnostics" / "beta_deterministic_policy_mode_validation.py").read_text(encoding="utf-8")
        self.assertNotIn(".generate(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)

    def test_10_old_current_chunking_matches_and_preserves_graph_contract(self):
        from types import SimpleNamespace

        input_ids = torch.arange(40).reshape(8, 5) % 3
        batch = SimpleNamespace(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            causal_logit_indices=torch.tensor([[1, 2, 3]] * 8),
            completion_ids=torch.tensor([[0, 1, 2]] * 8),
        )
        class TokenPolicy(TinyPolicy):
            def forward(self, input_ids, attention_mask):
                values = torch.nn.functional.one_hot(input_ids, 3).float()
                return SimpleNamespace(logits=super().forward(values))
        policy = TokenPolicy().eval()
        lora = [parameter for name, parameter in policy.named_parameters() if "lora_" in name]
        completions = tuple(tuple(row) for row in batch.completion_ids.tolist())
        old = score_full_sequences(policy, [0, 1], completions, 0, "cpu", grad_enabled=False, trainable_parameters=lora)
        current = score_full_sequences(policy, [0, 1], completions, 0, "cpu", grad_enabled=True, trainable_parameters=lora)
        self.assertEqual(old.logps, current.logps)
        self.assertFalse(any(old.requires_grad_by_microbatch) or any(old.graph_connected_by_microbatch))
        self.assertTrue(all(current.requires_grad_by_microbatch) and all(current.graph_connected_by_microbatch))

    def test_11_microbatch_preserves_g8_order(self):
        from types import SimpleNamespace

        class FixedPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            def forward(self, input_ids, attention_mask):
                base = torch.tensor([0.0, 1.0, 2.0]) + self.anchor * 0.0
                return SimpleNamespace(logits=base.expand(input_ids.shape[0], input_ids.shape[1], 3))

        completions = ((0,), (1,), (2,), (2, 1), (1, 0), (0, 2), (2, 0, 1), (1, 2, 0))
        result = score_full_sequences(FixedPolicy().eval(), [2, 1], completions, 0, "cpu", grad_enabled=False)
        token_logps = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), -1)
        expected = tuple(tuple(float(token_logps[token]) for token in row) for row in completions)
        self.assertEqual(result.logps, expected)


if __name__ == "__main__":
    unittest.main()
