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

from batch_collator_v1 import collate_business_group
from evaluation_runtime_v1 import evaluate_dev512, evaluate_probe20, final_checkpoint_selection_interface
from frontier_credit_v1 import FORMAT_INVALID_TOTAL
from rollout_runtime_v1 import FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, G, OLD_LOGPS_FROM_FULL_FORWARD_RESCORE, OLD_LOGPS_FROM_ROLLOUT_POLICY, PPO_OLD_LOGP_SOURCE, build_rollout_group, capture_sampled_logps, generation_contract, normalize_eos_token_ids, rollout_business_group, trim_generated_completion
from policy_scoring_v1 import POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE
from truerec_grpo_trainer_v1 import FORMAT_PENALTY_CONTEXT_TERMS, FORMAT_PENALTY_DOMAIN_TERMS, LOGICAL_POLICY_SCORING_PASSES_PER_GROUP, PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP, ROUTE_MULTIPLIER, TRAINER_MICROBATCH_SIZE, TrueRecGRPOTrainerV1, format_credit_tensors, gather_padded_action_logps
from truerec_loss_v1 import HPR_LAMBDA
from truerec_runtime_v1 import build_group_runtime_plan


VOCAB = 32
TOKENS = {
    1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>",
    5: "<s_b_5>", 6: "<s_c_6>", 7: "<s_c_7>", 8: "<s_b_8>",
    9: "<s_c_9>", 20: "bad", 21: "also_bad",
}
TOKEN_IDS = {value: key for key, value in TOKENS.items()}
GOLD = ("<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_2><s_c_7>", "<s_a_1><s_b_8><s_c_9>")


def id_to_token(token_id: int) -> str:
    return TOKENS.get(int(token_id), f"token_{token_id}")


def record(group_id="g0"):
    return {
        "recommendation_group_id": group_id,
        "fixed_domain_token": "<video>",
        "all_gold_abc": GOLD,
        "history_sids": ("<video><s_a_4><s_b_5><s_c_6>",),
    }


def rollout(group_id="g0", context_length=4, completions=None, logits=None):
    completions = completions or [[1, 2, 3], [1, 2, 7], [1, 2, 6], [1, 5, 6], [4, 5, 6], [1, 2, 3], [1, 8, 9], [4, 5, 6]]
    if logits is None:
        torch.manual_seed(11)
        logits = torch.randn(G, 3, VOCAB)
    return build_rollout_group(record(group_id), list(range(10, 10 + context_length)), completions, logits, id_to_token), logits


class MockOutput:
    def __init__(self, logits): self.logits = logits


class MockPolicy:
    def __init__(self, vocab=VOCAB): self.calls = 0; self.vocab = vocab; self.attention_masks = []
    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        self.attention_masks.append(attention_mask.clone())
        vocab_offsets = torch.arange(self.vocab, dtype=torch.float32).view(1, 1, -1)
        return MockOutput(input_ids.float().unsqueeze(-1) / 1000.0 + vocab_offsets / 1000.0)


class MockRenderer:
    def rl_context_ids(self, system, user, domain): return [10, 11, 12, 13]


class MockGenerateModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.kwargs = None; self.scores = None; self.anchor = torch.nn.Parameter(torch.tensor(0.0))
    def generate(self, **kwargs):
        self.kwargs = kwargs
        completions = torch.tensor([[1, 2, 3]] * 8)
        output = type("GenerateOutput", (), {})()
        output.sequences = torch.cat([torch.tensor([[10, 11, 12, 13]] * 8), completions], dim=1)
        torch.manual_seed(99)
        output.scores = tuple(torch.randn(8, VOCAB) for _ in range(3)); self.scores = output.scores
        return output
    def forward(self, input_ids, attention_mask):
        values = torch.arange(input_ids.shape[0] * input_ids.shape[1] * VOCAB, dtype=torch.float32, device=input_ids.device)
        return MockOutput(values.reshape(input_ids.shape[0], input_ids.shape[1], VOCAB) / 1000.0 + self.anchor * 0.0)


class Phase11IntegrationTest(unittest.TestCase):
    def test_01_generation_contract(self):
        self.assertEqual(generation_contract(), {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": 3, "num_return_sequences": 8})

    def test_02_one_group_is_g8(self): self.assertEqual(len(rollout()[0].candidates), 8)

    def test_03_group_boundary_preserved(self): self.assertEqual(rollout("business-17")[0].recommendation_group_id, "business-17")

    def test_04_sample_order_preserved(self): self.assertEqual([item.sample_index for item in rollout()[0].candidates], list(range(8)))

    def test_05_old_logp_capture(self):
        torch.manual_seed(2); logits = torch.randn(8, 3, VOCAB); ids = [[1, 2, 3]] * 8
        captured = capture_sampled_logps(logits, ids)
        expected = torch.log_softmax(logits, -1).gather(-1, torch.tensor(ids).unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.equal(torch.tensor(captured), expected))

    def test_06_old_logps_come_from_rollout_policy(self): self.assertTrue(OLD_LOGPS_FROM_ROLLOUT_POLICY)

    def test_07_old_current_separation(self):
        group, _ = rollout(); batch = collate_business_group(group, 0); current_logits = torch.randn(8, batch.input_ids.shape[1], VOCAB)
        old_before = batch.old_logps.clone(); current = gather_padded_action_logps(current_logits, batch)
        current_logits[:, batch.causal_logit_indices[:, 0], 1] += 1.0
        self.assertTrue(torch.equal(old_before, batch.old_logps)); self.assertFalse(torch.equal(current, gather_padded_action_logps(current_logits, batch)))

    def test_08_ratio_changes_after_perturbation(self):
        group, _ = rollout(); batch = collate_business_group(group, 0); logits = torch.randn(8, batch.input_ids.shape[1], VOCAB)
        current = gather_padded_action_logps(logits, batch); self.assertTrue(bool(torch.any(torch.exp(current - batch.old_logps) != 1.0)))

    def test_09_valid_abc_alignment(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 8); batch = collate_business_group(group, 0)
        self.assertTrue(torch.equal(batch.input_ids[:, -3:], torch.tensor([[1, 2, 3]] * 8)))

    def test_10_variable_context_lengths(self):
        short, _ = rollout("short", 2); long, _ = rollout("long", 9)
        self.assertEqual((collate_business_group(short, 0).context_lengths[0].item(), collate_business_group(long, 0).context_lengths[0].item()), (2, 9))

    def test_11_right_padding_alignment(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 7 + [[20, 21]])
        batch = collate_business_group(group, 0, "right"); self.assertEqual(batch.action_indices[-1].tolist(), [4, 5, 6])

    def test_12_left_padding_alignment(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 7 + [[20, 21]])
        batch = collate_business_group(group, 0, "left"); self.assertEqual(batch.action_indices[-1].tolist(), [5, 6, 7])

    def test_13_attention_mask_right_pad(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 7 + [[20, 21]])
        self.assertEqual(collate_business_group(group, 0, "right").attention_mask[-1].tolist(), [True] * 6 + [False])

    def test_14_attention_mask_left_pad(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 7 + [[20, 21]])
        self.assertEqual(collate_business_group(group, 0, "left").attention_mask[-1].tolist(), [False] + [True] * 6)

    def test_15_phase07_parser_reuse(self):
        group, _ = rollout(); self.assertEqual(group.candidates[0].metrics["frontier"], "EXACT"); self.assertEqual(group.candidates[2].metrics["frontier"], "C_FAIL")

    def test_16_frontier_plan_reuse(self):
        group, _ = rollout(); plan = build_group_runtime_plan([item.metrics for item in group.candidates], GOLD, TOKEN_IDS.__getitem__)
        self.assertEqual(plan.token_credits.shape, (8, 3))

    def test_17_hpr_plan_reuse(self):
        group, _ = rollout(completions=[[4, 5, 6]] * 8); plan = build_group_runtime_plan([item.metrics for item in group.candidates], GOLD, TOKEN_IDS.__getitem__)
        self.assertEqual(plan.hpr.trigger, "HPR_A")

    def test_18_invalid_not_a_fail(self):
        group, _ = rollout(completions=[[20, 21]] * 8); self.assertTrue(all(item.metrics["frontier"] == "INVALID_FORMAT" for item in group.candidates))

    def test_19_invalid_total(self): self.assertEqual(FORMAT_INVALID_TOTAL, -0.09375)

    def test_20_invalid_length_three_normalization(self):
        group, _ = rollout(completions=[[20, 21, 20]] * 8); credit, mask = format_credit_tensors(group)
        self.assertTrue(torch.equal(mask, torch.ones_like(mask))); self.assertAlmostEqual(float(credit[0].sum()), -0.09375)

    def test_21_invalid_length_two_normalization(self):
        group, _ = rollout(completions=[[20, 21]] * 8); credit, mask = format_credit_tensors(group)
        self.assertEqual(mask[0].tolist(), [True, True, False]); self.assertEqual(credit[0].tolist(), [-0.046875, -0.046875, 0.0])

    def test_22_invalid_penalty_not_context_or_domain(self): self.assertEqual((FORMAT_PENALTY_CONTEXT_TERMS, FORMAT_PENALTY_DOMAIN_TERMS), (0, 0))

    def test_23_shared_training_forward(self):
        group, _ = rollout(completions=[[4, 5, 6]] * 8); policy = MockPolicy(); trainer = TrueRecGRPOTrainerV1(policy, TOKEN_IDS.__getitem__, 0); trainer.compute_group(group)
        self.assertEqual((TRAINER_MICROBATCH_SIZE, LOGICAL_POLICY_SCORING_PASSES_PER_GROUP, PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP), (2, 1, 4))
        self.assertEqual((policy.calls, trainer.logical_policy_scoring_passes, trainer.physical_policy_forward_calls, trainer.train_policy_forward_calls), (4, 1, 4, 4))

    def test_24_no_hpr_extra_forward(self):
        group, _ = rollout(completions=[[1, 5, 6]] * 8); trainer = TrueRecGRPOTrainerV1(MockPolicy(), TOKEN_IDS.__getitem__, 0); trainer.compute_group(group)
        self.assertEqual(trainer.hpr_extra_forward_calls, 0)

    def test_25_attention_mask_passed_to_policy(self):
        group, _ = rollout(completions=[[1, 2, 3]] * 7 + [[20, 21]]); policy = MockPolicy(); trainer = TrueRecGRPOTrainerV1(policy, TOKEN_IDS.__getitem__, 0); trainer.compute_group(group)
        self.assertTrue(torch.equal(torch.cat(policy.attention_masks), collate_business_group(group, 0).attention_mask))

    def test_26_batch_group_reduction(self):
        first, _ = rollout("a"); second, _ = rollout("b", completions=[[4, 5, 6]] * 8); policy = MockPolicy(); trainer = TrueRecGRPOTrainerV1(policy, TOKEN_IDS.__getitem__, 0)
        out = trainer.compute_batch([first, second]); self.assertEqual(out.monitoring["business_group_count"], 2); self.assertEqual(out.monitoring["rollout_candidate_count"], 16)

    def test_27_no_route_multiplier(self): self.assertIsNone(ROUTE_MULTIPLIER)

    def test_28_hpr_lambda(self): self.assertEqual(HPR_LAMBDA, 0.02)

    def test_29_probe_subset_dev(self):
        dev = [f"g{i}" for i in range(512)]; self.assertEqual(evaluate_probe20(dev[:20], dev).expected_groups, 20)

    def test_30_probe_outside_dev_rejected(self):
        with self.assertRaises(ValueError): evaluate_probe20([f"p{i}" for i in range(20)], [f"d{i}" for i in range(512)])

    def test_31_dev_interface(self): self.assertFalse(evaluate_dev512([f"d{i}" for i in range(512)]).optimizer_allowed)

    def test_32_final_selection_disabled(self): self.assertFalse(final_checkpoint_selection_interface().checkpoint_selection_allowed)

    def test_33_phase08_sha_unchanged(self):
        expected = {"frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b", "hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32"}
        self._assert_head_shas("credit", expected)

    def test_34_phase10_sha_unchanged(self):
        expected = {"action_alignment.py": "2ca6c2b1ca034e84be55e359100ed86cf01503200fefa789e30abb2661f505e1", "truerec_loss_v1.py": "4a52f40639c332824e5dc45d6e4da1defcdbf4fc9f8e0c0dc380d5c4e5287d57", "truerec_runtime_v1.py": "2f18378f165d523e8eceaf85f8d575c7a237faacb99a072d8f22d326cef254f7", "truerec_trainer_v1.py": "7b5da094f7cb000fe5d66123eb4310a208e6fd963c0443b55efe9fb2cdddbdfb"}
        self._assert_head_shas("trainer", expected)

    def test_35_real_generate_adapter(self):
        model = MockGenerateModel(); value = record(); value.update({"system": "s", "user_content_nothink": "u"})
        group = rollout_business_group(model, value, MockRenderer(), id_to_token, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        self.assertEqual((len(group.candidates), group.context_ids), (8, (10, 11, 12, 13)))
        self.assertEqual({key: model.kwargs[key] for key in generation_contract()}, generation_contract())
        self.assertEqual(model.kwargs["eos_token_id"], [151645, 151643])
        self.assertEqual(model.kwargs["pad_token_id"], 151643)

    def test_36_generate_adapter_captures_score_logps(self):
        model = MockGenerateModel(); value = record(); value.update({"system": "s", "user_content_nothink": "u"})
        group = rollout_business_group(model, value, MockRenderer(), id_to_token, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        generation = torch.log_softmax(torch.stack(model.scores, dim=1), -1)[0, torch.arange(3), torch.tensor([1, 2, 3])]
        sequence = torch.tensor([[10, 11, 12, 13, 1, 2, 3]] * 2)
        logits = model(sequence, torch.ones_like(sequence)).logits
        rescored = torch.log_softmax(logits[0, torch.tensor([3, 4, 5])], -1)[torch.arange(3), torch.tensor([1, 2, 3])]
        self.assertTrue(torch.equal(torch.tensor(group.candidates[0].generation_score_logps), generation))
        self.assertTrue(torch.equal(torch.tensor(group.candidates[0].old_logps), rescored))
        self.assertFalse(torch.equal(torch.tensor(group.candidates[0].old_logps), generation))

    def test_37_formal_old_logp_contract(self):
        self.assertEqual(PPO_OLD_LOGP_SOURCE, "FULL_FORWARD_RESCORE")
        self.assertTrue(OLD_LOGPS_FROM_FULL_FORWARD_RESCORE)
        self.assertFalse(GENERATION_SCORES_USED_FOR_PPO)
        self.assertEqual(GENERATION_SCORE_LOGPS_ROLE, "DIAGNOSTIC_ONLY")
        self.assertEqual((POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE), ("eval", 2))

    def test_38_rescore_preserves_ids_metadata_and_order(self):
        model = MockGenerateModel(); value = record("preserved"); value.update({"system": "s", "user_content_nothink": "u"})
        group = rollout_business_group(model, value, MockRenderer(), id_to_token, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        self.assertEqual(group.recommendation_group_id, "preserved")
        self.assertEqual(group.context_ids, (10, 11, 12, 13))
        self.assertEqual([candidate.sample_index for candidate in group.candidates], list(range(8)))
        self.assertEqual([candidate.completion_ids for candidate in group.candidates], [(1, 2, 3)] * 8)
        self.assertEqual(group.all_gold_abc, GOLD)

    def test_39_eos_normalization(self):
        self.assertEqual(normalize_eos_token_ids((151645, 151643)), [151645, 151643])
        self.assertEqual(normalize_eos_token_ids([151645, 151643]), [151645, 151643])

    def test_40_trim_real_eos_151645(self):
        self.assertEqual(trim_generated_completion([1, 2, 151645, 9], FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID), [1, 2, 151645])

    def test_41_trim_eos_pad_151643_keeps_first_eos(self):
        self.assertEqual(trim_generated_completion([1, 2, 151643, 151643], FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID), [1, 2, 151643])

    def test_42_trim_no_eos_trailing_pad(self):
        self.assertEqual(trim_generated_completion([1, 2, 151643, 151643], [151645], FORMAL_PAD_TOKEN_ID), [1, 2])

    def test_43_formal_eos_pad_rejected_if_changed(self):
        model = MockGenerateModel(); value = record(); value.update({"system": "s", "user_content_nothink": "u"})
        with self.assertRaises(ValueError): rollout_business_group(model, value, MockRenderer(), id_to_token, FORMAL_PAD_TOKEN_ID, [151645])

    def _assert_head_shas(self, folder, expected):
        repo = ROOT.parents[2]
        for name, digest in expected.items():
            blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/{folder}/{name}"])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)


if __name__ == "__main__": unittest.main()
