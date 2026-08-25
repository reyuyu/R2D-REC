from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "credit"))
sys.path.insert(0, str(ROOT / "trainer"))

from action_alignment import CONTEXT_NONACTION_DIRECT_LOSS_TERMS, DOMAIN_DIRECT_LOSS_TERMS, align_action
from truerec_loss_v1 import HPR_LAMBDA, compose_total_loss, frontier_ppo_loss, multi_positive_log_mass_loss
from truerec_runtime_v1 import RuntimeHPRPlan, RuntimeHPRSite, build_group_runtime_plan, hpr_loss_from_shared_logits, sampled_action_logps
from truerec_trainer_v1 import TrueRecGroupBatch, TrueRecTrainerCore


GOLD = ["<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_2><s_c_7>", "<s_a_1><s_b_8><s_c_9>"]
TOKEN_IDS = {"<s_a_1>": 1, "<s_b_2>": 2, "<s_c_3>": 3, "<s_c_7>": 7, "<s_b_8>": 8, "<s_c_9>": 9}


def candidate(frontier: str, sample: int = 0):
    values = {
        "A_FAIL": (False, False, False, "<s_a_4><s_b_5><s_c_6>"),
        "B_FAIL": (True, False, False, "<s_a_1><s_b_5><s_c_6>"),
        "C_FAIL": (True, True, False, "<s_a_1><s_b_2><s_c_6>"),
        "EXACT": (True, True, True, "<s_a_1><s_b_2><s_c_3>"),
    }[frontier]
    return {"format_valid": True, "A_hit": values[0], "AB_hit": values[1], "exact": values[2], "parsed_abc": values[3], "sample_index": sample, "wrong_history_copy": False}


def candidates(frontier: str): return [{**candidate(frontier, index), "sample_index": index} for index in range(8)]


class MockOutput:
    def __init__(self, logits): self.logits = logits


class MockPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0)); self.logits = logits; self.calls = 0
    def forward(self, input_ids):
        self.calls += 1
        return MockOutput(self.logits + self.weight * 0.0)


def policy_fixture(frontier="A_FAIL"):
    torch.manual_seed(10)
    logits = torch.randn(8, 5, 12)
    completion = {"A_FAIL": [4, 5, 6], "B_FAIL": [1, 5, 6], "C_FAIL": [1, 2, 6], "EXACT": [1, 2, 3]}[frontier]
    completion_ids = torch.tensor([completion] * 8)
    input_ids = torch.cat([torch.full((8, 2), 10), completion_ids], dim=1)
    old = sampled_action_logps(logits, 2, completion_ids).detach()
    return logits, TrueRecGroupBatch(input_ids, 2, completion_ids, old, candidates(frontier), GOLD, TOKEN_IDS.__getitem__)


class TrueRecTrainerV1Test(unittest.TestCase):
    def test_01_action_alignment(self): self.assertEqual(align_action(10, [1, 2, 3]).action_indices, (10, 11, 12))
    def test_02_causal_logit_alignment(self): self.assertEqual(align_action(10, [1, 2, 3]).logit_indices, (9, 10, 11))
    def test_03_domain_not_loss_target(self): self.assertEqual(DOMAIN_DIRECT_LOSS_TERMS, 0)
    def test_04_context_not_loss_target(self): self.assertEqual(CONTEXT_NONACTION_DIRECT_LOSS_TERMS, 0)
    def test_05_ratio_one(self):
        current = torch.zeros(1, 8, 3); out = frontier_ppo_loss(current, current.clone(), torch.ones_like(current), torch.ones_like(current, dtype=torch.bool), .2)
        self.assertTrue(torch.equal(out.ratios, torch.ones_like(current)))
    def test_06_ppo_clip_positive(self):
        cur = torch.tensor([[[torch.log(torch.tensor(2.0)), 0., 0.]]]); old = torch.zeros_like(cur); credit = torch.tensor([[[1., 0., 0.]]]); mask = credit.bool()
        self.assertAlmostEqual(float(frontier_ppo_loss(cur, old, credit, mask, .2).loss), -1.2, places=6)
    def test_07_ppo_clip_negative(self):
        cur = torch.tensor([[[torch.log(torch.tensor(2.0)), 0., 0.]]]); old = torch.zeros_like(cur); credit = torch.tensor([[[-1., 0., 0.]]]); mask = credit.bool()
        self.assertAlmostEqual(float(frontier_ppo_loss(cur, old, credit, mask, .2).loss), 2.0, places=6)
    def test_08_gated_zero(self):
        x = torch.zeros(1, 8, 3); out = frontier_ppo_loss(x, x, torch.ones_like(x), torch.zeros_like(x, dtype=torch.bool), .2)
        self.assertEqual(float(out.loss), 0.0)
    def test_09_candidate_credited_token_sum(self):
        x = torch.zeros(1, 1, 3); credit = torch.tensor([[[1., 2., 3.]]]); out = frontier_ppo_loss(x, x, credit, torch.ones_like(x, dtype=torch.bool), .2)
        self.assertEqual(float(out.candidate_losses[0, 0]), -6.0)
    def test_10_group_g8_mean(self):
        x = torch.zeros(1, 8, 3); credit = torch.zeros_like(x); credit[0, :, 0] = torch.arange(1, 9); out = frontier_ppo_loss(x, x, credit, credit.bool(), .2)
        self.assertEqual(float(out.group_losses[0]), -4.5)
    def test_11_batch_group_mean(self):
        x = torch.zeros(2, 8, 3); credit = torch.zeros_like(x); credit[0, :, 0] = 1.; credit[1, :, 0] = 3.; out = frontier_ppo_loss(x, x, credit, credit.bool(), .2)
        self.assertEqual(float(out.loss), -2.0)
    def test_12_multi_positive_log_mass(self):
        logits = torch.tensor([0., 1., 2.]); expected = -torch.logsumexp(torch.log_softmax(logits, -1)[torch.tensor([1, 2])], 0)
        self.assertTrue(torch.allclose(multi_positive_log_mass_loss(logits, [1, 2]), expected))
    def test_13_single_target_ce_equivalence(self):
        logits = torch.tensor([0., 1., 2.]); expected = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2]))
        self.assertTrue(torch.allclose(multi_positive_log_mass_loss(logits, [2]), expected))
    def test_14_duplicate_onpolicy_positions_mean(self):
        logits = torch.zeros(8, 5, 4); logits[0, 1, 1] = 1.; logits[1, 1, 1] = 3.
        plan = RuntimeHPRPlan("HPR_A", (RuntimeHPRSite("A", (1,), ((0, 0), (1, 0))),))
        expected = torch.stack([multi_positive_log_mass_loss(logits[i, 1], [1]) for i in (0, 1)]).mean()
        self.assertTrue(torch.allclose(hpr_loss_from_shared_logits(logits, 2, plan), expected))
    def test_15_multiple_sites_group_mean(self):
        logits = torch.zeros(8, 5, 4); plan = RuntimeHPRPlan("HPR_B", (RuntimeHPRSite("B", (1,), ((0, 1),)), RuntimeHPRSite("B", (2,), ((1, 1),))))
        expected = torch.stack([multi_positive_log_mass_loss(logits[0, 2], [1]), multi_positive_log_mass_loss(logits[1, 2], [2])]).mean()
        self.assertTrue(torch.allclose(hpr_loss_from_shared_logits(logits, 2, plan), expected))
    def test_16_hpr_none_zero(self):
        logits = torch.randn(8, 5, 4); self.assertEqual(float(hpr_loss_from_shared_logits(logits, 2, RuntimeHPRPlan("HPR_NONE", ()))), 0.0)
    def test_17_lambda(self): self.assertEqual(HPR_LAMBDA, 0.02)
    def test_18_total_composition(self):
        out = compose_total_loss(torch.tensor(2.), torch.tensor(3.)); self.assertAlmostEqual(float(out.total_loss), 2.06, places=6)
    def test_19_hpr_a_integration(self): self.assertEqual(build_group_runtime_plan(candidates("A_FAIL"), GOLD, TOKEN_IDS.__getitem__).hpr.trigger, "HPR_A")
    def test_20_hpr_b_integration(self): self.assertEqual(build_group_runtime_plan(candidates("B_FAIL"), GOLD, TOKEN_IDS.__getitem__).hpr.trigger, "HPR_B")
    def test_21_hpr_c_integration(self): self.assertEqual(build_group_runtime_plan(candidates("C_FAIL"), GOLD, TOKEN_IDS.__getitem__).hpr.trigger, "HPR_C")
    def test_22_one_shared_forward(self):
        logits, group = policy_fixture("A_FAIL"); policy = MockPolicy(logits); trainer = TrueRecTrainerCore(policy); trainer.compute_group(group)
        self.assertEqual((policy.calls, trainer.policy_forward_calls), (1, 1))
    def test_23_no_hpr_extra_forward(self):
        logits, group = policy_fixture("B_FAIL"); trainer = TrueRecTrainerCore(MockPolicy(logits)); trainer.compute_group(group); self.assertEqual(trainer.hpr_extra_forward_calls, 0)
    def test_24_no_parameter_mutation(self):
        logits, group = policy_fixture("C_FAIL"); policy = MockPolicy(logits); before = policy.weight.detach().clone(); TrueRecTrainerCore(policy).compute_group(group); self.assertTrue(torch.equal(before, policy.weight.detach()))
    def test_25_phase08_credit_sha_unchanged(self):
        expected = {"frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b", "hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32"}
        repo = ROOT.parents[2]
        for name, digest in expected.items():
            blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/credit/{name}"])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)
    def test_26_old_logps_required_shape(self):
        x = torch.zeros(1, 8, 3)
        with self.assertRaises(ValueError): frontier_ppo_loss(x, torch.zeros(1, 8, 2), x, x.bool(), .2)
    def test_27_exact_action_length_gate(self):
        with self.assertRaises(ValueError): align_action(2, [1, 2])
    def test_28_monitor_fields(self):
        logits, group = policy_fixture("A_FAIL"); monitor = TrueRecTrainerCore(MockPolicy(logits)).compute_group(group).monitoring
        required = {"group_count", "candidate_count", "A_hit_rate", "AB_hit_rate", "exact_rate", "HPR_A_groups", "frontier_A_positive", "frontier_A_negative", "frontier_loss", "hpr_loss_raw", "hpr_loss_weighted", "total_loss", "wrong_history_copy_rate"}
        self.assertTrue(required <= set(monitor))
    def test_29_batch_one_forward_per_group(self):
        logits, group = policy_fixture("A_FAIL"); policy = MockPolicy(logits); trainer = TrueRecTrainerCore(policy); trainer.compute_batch([group, group]); self.assertEqual(policy.calls, 2)


if __name__ == "__main__": unittest.main()
