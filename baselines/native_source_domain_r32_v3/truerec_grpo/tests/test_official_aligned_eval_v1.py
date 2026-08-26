from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from beam32_contract import BeamDecoderContract, make_beam_candidate, synthetic_beam_fixture
from compare_checkpoints import checkpoint_comparison_row
from eval_aggregate import aggregate_results
from eval_request import OfficialAlignedEvalRequest, frozen_split_interface
from eval_result import EvaluationBusinessSample, deduplicate_source_rows
from model_spec import ModelSpec
from official_aligned_contract import ACTION_LEVELS, CONTRACT_ID, OFFICIAL_COMPOSITE_FORMULA_AVAILABLE, OFFICIAL_EVAL_SOURCE_CODE_AVAILABLE, TRAIN_ROLLOUT_G, UNKNOWN_OFFICIAL_DECODING_DETAILS, ContractError, PrefixContract, eval_prefix_contract, shared_prefix_contract_pass, train_prefix_contract, validate_no_bridge_contract
from official_aligned_renderer import RENDERER_CONTRACT_ID, render_official_aligned_prefix
from scoring_adapter import SCORING_LABEL, internal_group_diagnostics


TOKENS = {1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>", 5: "<s_b_5>", 6: "<s_c_6>", 20: "bridge", 21: "<|video_begin|>", 22: "prose"}


def token(token_id): return TOKENS[int(token_id)]


def spec(family="beta"):
    return ModelSpec(f"{family}-checkpoint", family, "/checkpoint", "/base", "/adapter", "/tokenizer")


def request(family="beta", split="dev512"): return OfficialAlignedEvalRequest(spec(family), split)


def beam(group="g0", domain="video", valid=True):
    rows = [([1, 2, 3] if valid else [20, 1, 2], float(32 - index)) for index in range(32)]
    return synthetic_beam_fixture(rows, recommendation_group_id=group, target_domain=domain, fixed_domain_token=f"<|{domain}_begin|>", id_to_token=token)


def sample(group="g0", domain="video"):
    return EvaluationBusinessSample(group, domain, (f"<|{domain}_begin|><s_a_1><s_b_2><s_c_3>",), ("<s_a_1><s_b_2><s_c_3>",), beam(group, domain))


class Renderer:
    def rl_context_ids(self, system, user, domain): return [10, 11, 21]
    def encode(self, text): return [21]


class OfficialAlignedEvalTest(unittest.TestCase):
    def test_01_train_bridge_false(self): self.assertFalse(train_prefix_contract().bridge)
    def test_02_eval_bridge_false(self): self.assertFalse(eval_prefix_contract().bridge)
    def test_03_beta_bridge_rejected(self):
        with self.assertRaises(ContractError): validate_no_bridge_contract({"family": "beta", "beta_bridge": "legacy"})
    def test_04_beta_gamma_bridge_rejected(self):
        with self.assertRaises(ContractError): validate_no_bridge_contract({"family": "beta_gamma", "bridge_prefix": "legacy"})
    def test_05_truerec_bridge_rejected(self):
        with self.assertRaises(ContractError): validate_no_bridge_contract({"family": "truerec", "natural_language_bridge": True})
    def test_06_train_fixed_domain(self): self.assertTrue(train_prefix_contract().fixed_domain_in_context)
    def test_07_eval_fixed_domain(self): self.assertTrue(eval_prefix_contract().fixed_domain_in_context)
    def test_08_train_domain_not_action(self): self.assertEqual(train_prefix_contract().action_levels, ("A", "B", "C"))
    def test_09_eval_domain_not_action(self): self.assertEqual(eval_prefix_contract().action_levels, ("A", "B", "C"))
    def test_10_train_exact_abc3(self): self.assertEqual((ACTION_LEVELS, train_prefix_contract().action_tokens), (("A", "B", "C"), 3))
    def test_11_eval_exact_abc3(self): self.assertEqual(eval_prefix_contract().action_tokens, 3)
    def test_12_train_g8(self): self.assertEqual(TRAIN_ROLLOUT_G, 8)
    def test_13_eval_beam32(self): self.assertEqual(BeamDecoderContract().beam_size, 32)
    def test_14_three_families_share_contract(self): self.assertEqual({request(value).eval_contract_id for value in ("beta", "beta_gamma", "truerec")}, {CONTRACT_ID})
    def test_15_one_group_one_sample(self): self.assertEqual(sample().recommendation_group_id, "g0")
    def test_16_route_rows_deduplicated(self):
        row = {"recommendation_group_id": "g", "target_domain": "video", "all_gold_sids": ["sid"], "all_gold_abc": ["abc"]}
        self.assertEqual(len(deduplicate_source_rows([{**row, "route": "think"}, {**row, "route": "no_think"}])), 1)
    def test_17_multi_positive_preserved(self):
        row = {"recommendation_group_id": "g", "target_domain": "video", "all_gold_sids": ["s1", "s2"], "all_gold_abc": ["a1", "a2"]}
        self.assertEqual(deduplicate_source_rows([row])[0]["all_gold_sids"], ("s1", "s2"))
    def test_18_probe_subset_dev(self):
        dev = [f"g{i}" for i in range(512)]; self.assertFalse(frozen_split_interface("probe20", dev[:20], dev).checkpoint_selection_allowed)
    def test_19_dev_selection_allowed(self): self.assertTrue(frozen_split_interface("dev512", [f"d{i}" for i in range(512)]).checkpoint_selection_allowed)
    def test_20_final_selection_disabled(self): self.assertFalse(frozen_split_interface("final2048", [f"f{i}" for i in range(2048)]).checkpoint_selection_allowed)
    def test_21_raw_ids_retained(self): self.assertEqual(beam()[0].raw_token_ids, (1, 2, 3))
    def test_22_shared_parser_valid(self): self.assertTrue(make_beam_candidate("g", "video", "<|video_begin|>", 0, [1, 2, 3], token).format_valid)
    def test_23_bridge_before_abc_rejected(self): self.assertFalse(make_beam_candidate("g", "video", "<|video_begin|>", 0, [20, 1, 2, 3], token).format_valid)
    def test_24_generated_domain_rejected(self): self.assertFalse(make_beam_candidate("g", "video", "<|video_begin|>", 0, [21, 1, 2, 3], token).format_valid)
    def test_25_extra_token_rejected(self): self.assertFalse(make_beam_candidate("g", "video", "<|video_begin|>", 0, [1, 2, 3, 22], token).format_valid)
    def test_26_deterministic_aggregation(self): self.assertEqual(aggregate_results([sample("a"), sample("b")]), aggregate_results([sample("b"), sample("a")]))
    def test_27_no_official_composite(self): self.assertFalse(OFFICIAL_COMPOSITE_FORMULA_AVAILABLE)
    def test_28_unknown_fields_explicit(self): self.assertGreaterEqual(len(UNKNOWN_OFFICIAL_DECODING_DETAILS), 8)
    def test_29_training_runtime_no_bridge_insertion(self):
        text = (ROOT / "trainer" / "rollout_runtime_v1.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("bridge_prefix", text); self.assertNotIn("legacy_bridge", text)
    def test_30_previous_contract_shas(self):
        expected = {
            "credit/frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b",
            "credit/hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32",
            "trainer/action_alignment.py": "2ca6c2b1ca034e84be55e359100ed86cf01503200fefa789e30abb2661f505e1",
            "trainer/truerec_loss_v1.py": "4a52f40639c332824e5dc45d6e4da1defcdbf4fc9f8e0c0dc380d5c4e5287d57",
            "trainer/truerec_runtime_v1.py": "2f18378f165d523e8eceaf85f8d575c7a237faacb99a072d8f22d326cef254f7",
            "trainer/rollout_runtime_v1.py": "c21f9d79ccc9f3e43b55e01278ea31b07ce92d7230b520302848beed8a896b81",
            "trainer/truerec_grpo_trainer_v1.py": "1bf73e7b9888025cb5cc5ba7fb92788bb7841881fecbf566926b6814eb12f552",
        }
        repo = ROOT.parents[2]
        for relative, digest in expected.items():
            blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/{relative}"])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)
    def test_31_official_source_unavailable(self): self.assertFalse(OFFICIAL_EVAL_SOURCE_CODE_AVAILABLE)
    def test_32_renderer_context_ends_domain(self): self.assertEqual(render_official_aligned_prefix({"recommendation_group_id": "g", "target_domain": "video", "fixed_domain_token": "<|video_begin|>", "system": "s", "user_content_nothink": "u"}, Renderer()).action_start_position, 3)
    def test_33_family_cannot_change_contract(self):
        with self.assertRaises(ContractError): ModelSpec("x", "beta", "/c", "/b", "/a", "/t", "legacy")
    def test_34_internal_score_labeled_nonofficial(self): self.assertEqual(internal_group_diagnostics(beam(), ("<s_a_1><s_b_2><s_c_3>",))["score_label"], SCORING_LABEL)
    def test_35_comparison_schema_uniform(self):
        aggregate = aggregate_results([sample()]); self.assertEqual({checkpoint_comparison_row(spec(f), "dev512", aggregate)["eval_contract_id"] for f in ("beta", "beta_gamma", "truerec")}, {CONTRACT_ID})
    def test_36_shared_train_eval_prefix(self): self.assertTrue(shared_prefix_contract_pass())
    def test_37_decoder_is_frontend_only(self): self.assertFalse(BeamDecoderContract().real_gpu_decoder_implemented)
    def test_38_eval_has_no_optimizer_or_backward(self):
        value = frozen_split_interface("dev512", [f"d{i}" for i in range(512)]); self.assertEqual((value.optimizer_allowed, value.backward_allowed), (False, False))


if __name__ == "__main__": unittest.main()
