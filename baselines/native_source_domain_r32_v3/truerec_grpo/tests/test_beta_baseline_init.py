from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "initialization"))
sys.path.insert(0, str(ROOT / "eval"))

from beta_baseline_init import BETA_CHECKPOINT, FROZEN_CORE_SHA, INIT_FAMILY, PRETRAINED_BASE, ContractError, load_contract
from official_aligned_renderer import render_official_aligned_prefix


class MockRenderer:
    def rl_context_ids(self, system, user, domain): return [10, 11, 176245]
    def encode(self, text): return [176245]


class BetaBaselineInitTest(unittest.TestCase):
    def test_01_beta_checkpoint_path(self): self.assertEqual(Path(load_contract().checkpoint_path), BETA_CHECKPOINT)
    def test_02_beta_gamma_checkpoint_rejected(self):
        with self.assertRaises(ContractError): replace(load_contract(), checkpoint_path="/data/BETA-GAMMA/checkpoint").validate_static()
    def test_03_pretrained_base_path(self): self.assertEqual(Path(load_contract().pretrained_base_path), PRETRAINED_BASE)
    def test_04_no_bridge_contract(self): self.assertFalse(load_contract().bridge)
    def test_05_fixed_domain_terminal_context_contract(self): self.assertTrue(load_contract().fixed_domain_in_context)
    def test_06_domain_not_action(self): self.assertFalse(load_contract().domain_generated_by_model)
    def test_07_action_exactly_abc3(self): self.assertEqual((load_contract().action_levels, load_contract().action_tokens), (("A", "B", "C"), 3))
    def test_08_rollout_g8(self): self.assertEqual(load_contract().rollout_g, 8)
    def test_09_beta_gamma_not_init(self): self.assertFalse(load_contract().beta_gamma_used_as_init)
    def test_10_phase07_raw_not_reused(self): self.assertFalse(load_contract().reuse_phase07_beta_gamma_raw)
    def test_11_phase08_plan_not_reused(self): self.assertFalse(load_contract().reuse_phase08_beta_gamma_plan)
    def test_12_live_rollout_required(self): self.assertEqual(load_contract().policy_rollout_source, "live_beta_policy")
    def test_13_online_hpr_required(self): self.assertEqual(load_contract().current_hpr_plan_source, "online_current_group")
    def test_14_init_family(self): self.assertEqual(load_contract().init_family, INIT_FAMILY)
    def test_15_active_bridge_rejected(self):
        with self.assertRaises(ContractError): replace(load_contract(), bridge=True).validate_static()
    def test_16_domain_generation_rejected(self):
        with self.assertRaises(ContractError): replace(load_contract(), domain_generated_by_model=True).validate_static()
    def test_17_wrong_action_length_rejected(self):
        with self.assertRaises(ContractError): replace(load_contract(), action_tokens=4).validate_static()
    def test_18_frozen_core_shas(self):
        repo = ROOT.parents[2]
        for relative, digest in FROZEN_CORE_SHA.items():
            if relative == "trainer/rollout_runtime_v1.py":
                # Phase 1.2B explicitly extends this adapter to accept EOS ID lists.
                continue
            blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/{relative}"])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)
    def test_19_fixed_domain_is_terminal_before_action(self):
        record = {"recommendation_group_id": "g", "target_domain": "video", "fixed_domain_token": "<|video_begin|>", "system": "s", "user_content_nothink": "u"}
        rendered = render_official_aligned_prefix(record, MockRenderer())
        self.assertEqual((rendered.context_ids[-1], rendered.action_start_position), (176245, len(rendered.context_ids)))


if __name__ == "__main__": unittest.main()
