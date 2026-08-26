from pathlib import Path
import hashlib
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from beta_old_logp_rescore_validation import TRAINER_SHA256
from beta_one_optimizer_step_audit import base_parameter_versions, build_lora_optimizer, lora_tensor_shas, ratio_stats


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.lora_b = torch.nn.Parameter(torch.tensor([3.0]))
        self.base = torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False)


class BetaOneOptimizerStepAuditTest(unittest.TestCase):
    def test_lora_only_adamw_inventory(self):
        model = TinyModel()
        optimizer, audit = build_lora_optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertTrue(audit["optimizer_lora_only"])
        self.assertEqual(audit["optimizer_param_count"], 3)
        self.assertEqual(audit["duplicate_optimizer_tensor_count"], 0)

    def test_one_step_changes_lora_not_base(self):
        model = TinyModel()
        optimizer, _ = build_lora_optimizer(model)
        lora_before = lora_tensor_shas(model)
        base_before = base_parameter_versions(model)
        optimizer.zero_grad(set_to_none=True)
        (model.lora_a.square().sum() + model.lora_b.square().sum()).backward()
        optimizer.step()
        lora_after = lora_tensor_shas(model)
        self.assertTrue(any(lora_before[name] != lora_after[name] for name in lora_before))
        self.assertEqual(base_before, base_parameter_versions(model))

    def test_post_step_ratio_statistics(self):
        old = torch.zeros((2, 3))
        current = torch.tensor([[0.0, 0.1, -0.1], [0.0, 0.0, 0.2]])
        mask = torch.ones((2, 3), dtype=torch.bool)
        stats = ratio_stats(current, old, mask)
        self.assertEqual(stats["token_count"], 6)
        self.assertEqual(stats["changed_token_count"], 3)
        self.assertTrue(all(math_value > 0 for math_value in (stats["mean"], stats["min"], stats["max"])))

    def test_runner_has_exactly_one_step_and_no_persistence(self):
        source = (ROOT / "diagnostics" / "beta_one_optimizer_step_audit.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("optimizer.step()"), 1)
        self.assertNotIn("scheduler.step(", source)
        self.assertNotIn(".generate(", source)
        self.assertNotIn("save_pretrained(", source)
        self.assertNotIn("torch.save(", source)

    def test_trainer_core_sha_unchanged(self):
        blob = subprocess.check_output([
            "git", "-C", str(ROOT.parents[2]), "show",
            "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py",
        ])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)


if __name__ == "__main__":
    unittest.main()
