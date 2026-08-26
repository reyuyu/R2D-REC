from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from beta_single_group_loss_audit import GROUP_ID, compare_current_old, reconstruct_business_group, validate_phase12b_artifact, verify_total_composition
from batch_collator_v1 import collate_business_group
from truerec_runtime_v1 import build_group_runtime_plan


TOKENS = {4: "<s_a_4>", 5: "<s_b_5>", 6: "<s_c_6>", 1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>"}


class Tokenizer:
    def convert_ids_to_tokens(self, value): return TOKENS[int(value)]
    def convert_tokens_to_ids(self, value): return {token: token_id for token_id, token in TOKENS.items()}[value]


class Renderer:
    tokenizer = Tokenizer()
    def rl_context_ids(self, system, user, domain): return [10, 11, 176247]
    def encode(self, text): return [176247]


def artifact():
    candidates = []
    for index in range(8):
        candidates.append({"sample_index": index, "raw_token_ids": [4, 5, 6], "actual_completion_length": 3, "old_logps": [-1.0, -2.0, -3.0], "format_valid": True, "parsed_abc": "<s_a_4><s_b_5><s_c_6>", "A_hit": False, "AB_hit": False, "exact": False, "frontier": "A_FAIL"})
    return {"model": {"family": "BETA_BASELINE"}, "group": {"recommendation_group_id": GROUP_ID}, "candidates": candidates}


def record():
    return {"recommendation_group_id": GROUP_ID, "system": "s", "user_content_nothink": "u", "fixed_domain_token": "<|prod_begin|>", "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"], "history_sids": []}


class BetaSingleGroupLossAuditTest(unittest.TestCase):
    def test_01_phase12b_artifact_group_count_gate(self): validate_phase12b_artifact(artifact())
    def test_02_saved_old_logp_shape_gate(self):
        value = artifact(); value["candidates"][0]["old_logps"] = [-1.0, -2.0]
        with self.assertRaises(Exception): validate_phase12b_artifact(value)
    def test_03_exact_sampled_sequence_reconstruction(self):
        group = reconstruct_business_group(record(), artifact(), Renderer()); batch = collate_business_group(group, 0)
        self.assertTrue(torch.equal(batch.input_ids[:, -3:], torch.tensor([[4, 5, 6]] * 8)))
    def test_04_current_old_comparison(self):
        old = torch.tensor([[-1.0, -2.0, -3.0]] * 8); current = old + 0.01; mask = torch.ones_like(old, dtype=torch.bool)
        result = compare_current_old(current, old, mask); self.assertAlmostEqual(result["abs_max"], 0.01, places=5); self.assertTrue(math.isfinite(result["ratio_mean"]))
    def test_05_hpr_a_live_plan(self):
        group = reconstruct_business_group(record(), artifact(), Renderer()); plan = build_group_runtime_plan([candidate.metrics for candidate in group.candidates], group.all_gold_abc, Renderer.tokenizer.convert_tokens_to_ids)
        self.assertEqual((plan.hpr.trigger, plan.monitoring["frontier_A_negative"], plan.monitoring["frontier_A_positive"]), ("HPR_A", 8, 0))
    def test_06_total_loss_composition(self): self.assertTrue(verify_total_composition(1.0, 2.0, 0.04, 1.04))
    def test_07_wrong_total_rejected(self): self.assertFalse(verify_total_composition(1.0, 2.0, 0.04, 1.05))


if __name__ == "__main__": unittest.main()
