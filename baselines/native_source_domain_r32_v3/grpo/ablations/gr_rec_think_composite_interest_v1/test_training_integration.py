from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest

from .composite_trainer import (
    ThinkCompositeInterestRecGRPOTrainer, assert_gold_isolation,
    build_global_runtime, classify_winner, decode_reward_completion,
    score_candidate, slice_global,
)
from .interest_metric import composite_reward, population_advantages
from grpo_beam_domain import (
    DOMAIN_PREFIX,
    build_fixed_domain_beam_input,
    domain_prefix,
    parse_fixed_domain_beam_sid,
    validate_gold_domains,
)
from .run_gr_rec_think_composite_interest_v1 import (
    AUTO_SAVE_STEPS, CHECKPOINT_STEPS, PROBE_DOMAIN_ORDER, PROBE_IDS, PROBE_ROUNDS,
    PROBE_STEPS, SAVE_TOTAL_LIMIT, SEED,
    checkpoint_save_config, frozen_contract, launch_training,
    should_save_checkpoint,
)
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units

SID = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
PROMPT = "history " + SID
HEADING = "\u3010\u5174\u8da3\u5f52\u7eb3\u3011"
GOLD = "<think>\n" + HEADING + "\n1. tactical game " + SID + "\n</think>"
GOOD = GOLD
OTHER = "<think>\n" + HEADING + "\n1. beauty care\n</think>"


def records(completions, group_id="g1"):
    return [{"group_id": group_id, "rank": 0, "local_index": index,
             "prompt": PROMPT, "gold_cot": GOLD, "completion": completion}
            for index, completion in enumerate(completions)]


class SpecialTokenFixture:
    def decode(self, candidate_ids, skip_special_tokens=False):
        self.last_skip_special_tokens = skip_special_tokens
        if skip_special_tokens:
            return "<think>\n" + HEADING + "\n1. tactical game\n"
        return GOOD


class TrainingChainTests(unittest.TestCase):
    class PrefixTokenizer:
        token_ids = {
            "<|video_begin|>": [101],
            "<|prod_begin|>": [102],
            "<|ad_begin|>": [103],
            "<|living_begin|>": [104],
        }

        def encode(self, text, add_special_tokens=False):
            if add_special_tokens:
                raise AssertionError("domain prefix must not add special tokens")
            return list(self.token_ids[text])

    class ABC3Tokenizer(PrefixTokenizer):
        tokens = {
            201: "<s_a_2406>", 202: "<s_b_3727>", 203: "<s_c_5563>",
            204: "<s_c_4>", 205: "<s_b_9>", 206: "<s_a_9>",
            999: "invalid",
        }

        def convert_ids_to_tokens(self, token_ids, skip_special_tokens=False):
            if skip_special_tokens:
                raise AssertionError("strict Beam parsing must preserve special tokens")
            if isinstance(token_ids, int):
                return self.tokens[token_ids]
            return [self.tokens[token_id] for token_id in token_ids]

    def test_fixed_domain_prefix_mapping_and_unknown_fail_closed(self):
        self.assertEqual(DOMAIN_PREFIX, {
            "video": "<|video_begin|>",
            "prod": "<|prod_begin|>",
            "ad": "<|ad_begin|>",
            "living": "<|living_begin|>",
        })
        for domain, prefix in DOMAIN_PREFIX.items():
            self.assertEqual(domain_prefix(domain), prefix)
        with self.assertRaisesRegex(ValueError, "UNKNOWN_TARGET_DOMAIN"):
            domain_prefix("search")

    def test_beam_context_appends_only_fixed_domain_prefix(self):
        prompt_ids = [10, 11]
        cot_ids = [20, 21, 22]
        old_context = prompt_ids + cot_ids
        new_context, prefix, prefix_ids = build_fixed_domain_beam_input(
            self.PrefixTokenizer(), prompt_ids, cot_ids, "prod",
        )
        self.assertEqual(prefix, "<|prod_begin|>")
        self.assertEqual(prefix_ids, [102])
        self.assertEqual(new_context, old_context + [102])
        self.assertEqual(new_context[:-1], old_context)

    def test_production_beam_task_uses_fixed_domain_context(self):
        import run_grpo_trl_smoke as production
        from unittest.mock import patch

        captured = {}

        class Model:
            training = False

            def eval(self):
                return self

        class Tokenizer(self.PrefixTokenizer):
            def decode(self, token_ids, skip_special_tokens=False):
                return "<think>reason</think>ignored"

            def encode(self, text, add_special_tokens=False):
                if text == "<think>reason</think>":
                    return [20, 21]
                return super().encode(text, add_special_tokens=add_special_tokens)

        def fake_run(model, tokenizer, task):
            captured.update(task)
            return {
                "task_id": task["task_id"], "beam_sec": 0.0, "reward": 0.0,
                "exact": 0, "ab": 0, "a": 0, "invalid": 32,
            }

        with (
            patch.object(production, "cached_prompt_ids", return_value=[10, 11]),
            patch.object(production, "run_beam32_task", side_effect=fake_run),
            patch.object(production, "distributed_beam_enabled", return_value=False),
        ):
            rewards = production.make_beam32_fn(Model(), Tokenizer())(
                ["prompt"], ["completion"], [[7, 8]],
                [{("prod", 1, 2, 3)}], ["prod"],
            )
        self.assertEqual(rewards, [0.0])
        self.assertEqual(captured["input_ids"], [10, 11, 20, 21, 102])
        self.assertEqual(captured["domain_prefix"], "<|prod_begin|>")
        self.assertEqual(captured["target_domain"], "prod")

    def test_strict_abc3_raw_id_parser(self):
        tokenizer = self.ABC3Tokenizer()
        self.assertEqual(
            parse_fixed_domain_beam_sid(tokenizer, [201, 202, 203], "prod"),
            ("prod", 2406, 3727, 5563),
        )
        self.assertIsNone(parse_fixed_domain_beam_sid(tokenizer, [201, 202], "prod"))
        self.assertIsNone(parse_fixed_domain_beam_sid(tokenizer, [201, 202, 203, 204], "prod"))
        self.assertIsNone(parse_fixed_domain_beam_sid(tokenizer, [202, 201, 203], "prod"))

    def test_strict_parser_ignores_later_text_that_legacy_final_sid_would_take(self):
        from grpo_sid import final_sid
        long_text = "<|prod_begin|><s_a_2406><s_b_3727><s_c_5563></think><|video_begin|><s_a_1><s_b_2><s_c_3>"
        self.assertEqual(final_sid(long_text), ("video", 1, 2, 3))
        self.assertEqual(parse_fixed_domain_beam_sid(self.ABC3Tokenizer(), [201, 202, 203], "prod"), ("prod", 2406, 3727, 5563))

    def test_beam_detail_rows_preserve_all_outputs_and_relations(self):
        from run_grpo_trl_smoke import build_beam_detail_rows

        texts = ["exact", "ab", "a", "miss", "invalid"] + ["tail"] * 27
        sids = [
            ("prod", 1, 2, 3),
            ("prod", 1, 2, 9),
            ("prod", 1, 9, 9),
            ("prod", 9, 9, 9),
            None,
        ] + [None] * 27
        token_ids = [[201, 202, 203]] * 32
        tokens = [["<s_a_2406>", "<s_b_3727>", "<s_c_5563>"]] * 32
        rows = build_beam_detail_rows(
            texts, token_ids, tokens, sids, {("prod", 1, 2, 3)})
        self.assertEqual(len(rows), 32)
        self.assertEqual([row["beam_index"] for row in rows], list(range(32)))
        self.assertEqual([row["generated_continuation_text"] for row in rows], texts)
        self.assertEqual(rows[0]["parsed_sid"], ["prod", 1, 2, 3])
        self.assertEqual(rows[0]["generated_token_ids"], [201, 202, 203])
        self.assertEqual(rows[0]["generated_tokens"], tokens[0])
        self.assertEqual(rows[0]["generated_token_count"], 3)
        self.assertEqual(
            [row["relation_to_gold"] for row in rows[:5]],
            ["EXACT", "AB", "A", "VALID_NO_HIT", "INVALID"],
        )

    def test_beam_detail_capture_does_not_change_reward(self):
        import run_grpo_trl_smoke as production
        from unittest.mock import patch

        texts = ["<s_a_2406><s_b_3727><s_c_5563>"] + ["invalid"] * 31
        token_ids = [[201, 202, 203]] + [[999, 202, 203]] * 31
        base_task = {
            "task_id": (0, 0),
            "input_ids": [1],
            "target_domain": "prod",
            "domain_prefix": "<|prod_begin|>",
            "gold": [["prod", 2406, 3727, 5563]],
        }
        with (
            patch.object(production, "generate_batch", return_value=(texts, token_ids)) as generate,
            patch.object(production.torch.cuda, "synchronize"),
        ):
            plain = production.run_beam32_task(
                None, self.ABC3Tokenizer(), {**base_task, "capture_monitor": False}
            )
            captured = production.run_beam32_task(
                None, self.ABC3Tokenizer(), {**base_task, "capture_monitor": True}
            )
        self.assertEqual(plain["reward"], captured["reward"])
        self.assertNotIn("beam_details", plain)
        self.assertEqual(len(captured["beam_details"]), 32)

        self.assertEqual(captured["generated_token_count_mismatch"], 0)
        self.assertTrue(all(count == 3 for count in captured["generated_token_counts"]))
        self.assertEqual(generate.call_args.kwargs["min_new_tokens"], 3)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 3)
        self.assertTrue(generate.call_args.kwargs["return_ids"])
        self.assertTrue(all(row["generated_token_count"] == 3 for row in captured["beam_details"]))
    def test_beam_detail_writer_uses_origin_rank_and_context(self):
        from monitor.writer import MonitorWriter

        with tempfile.TemporaryDirectory() as directory:
            writer = MonitorWriter(True, directory, "run", rank=2)
            writer.set_beam_context(step=7, rollout_id=4, source="training")
            beams = [{"beam_index": index} for index in range(32)]
            self.assertTrue(writer.write_beam_detail({
                "recommendation_group_id": "g1",
                "origin_rank": 2,
                "local_index": 3,
                "beam_raw": 8.0,
                "beams": beams,
            }))
            path = Path(directory) / "run/beam_details/rank2.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((row["step"], row["rollout_id"]), (7, 4))
            self.assertEqual(row["recommendation_group_id"], "g1")
            self.assertEqual(len(row["beams"]), 32)

    def test_gold_domain_mismatch_is_hard_but_not_domain_source(self):
        validate_gold_domains({("prod", 1, 2, 3)}, "prod")
        with self.assertRaisesRegex(ValueError, "GOLD_DOMAIN_MISMATCH"):
            validate_gold_domains({("video", 1, 2, 3)}, "prod")

    def test_raw_reward_decode_preserves_sid_and_grounding(self):
        tokenizer = SpecialTokenFixture()
        raw = decode_reward_completion(tokenizer, [1, 2, 3])
        parsed = extract_interest_units(raw, PROMPT)
        self.assertFalse(tokenizer.last_skip_special_tokens)
        self.assertEqual(parsed.units[0].grounded_evidence_sids, (SID,))
        self.assertEqual(score_candidate(raw, GOLD, PROMPT, 2.0)["grounded_n"], 1)

    def test_special_token_stripping_fixture_loses_evidence(self):
        tokenizer = SpecialTokenFixture()
        stripped = tokenizer.decode([1, 2, 3], skip_special_tokens=True)
        parsed = extract_interest_units(stripped, PROMPT)
        self.assertEqual(parsed.units[0].grounded_evidence_sids, ())
        self.assertLess(
            score_candidate(stripped, GOLD, PROMPT, 2.0)["mean_match_similarity"],
            score_candidate(GOOD, GOLD, PROMPT, 2.0)["mean_match_similarity"],
        )

    def test_training_probe_raw_decode_parity(self):
        from grpo_probe import FixedProbeEvaluator
        tokenizer = SpecialTokenFixture()
        training_text = decode_reward_completion(tokenizer, [1, 2, 3])
        probe_text = tokenizer.decode([1, 2, 3], skip_special_tokens=False)
        self.assertEqual(training_text, probe_text)
        self.assertIn("skip_special_tokens=False", inspect.getsource(FixedProbeEvaluator._think))

    def test_online_reward_score_matches_direct_raw_score(self):
        tokenizer = SpecialTokenFixture()
        online_text = decode_reward_completion(tokenizer, [1, 2, 3])
        self.assertEqual(
            score_candidate(online_text, GOLD, PROMPT, 2.0),
            score_candidate(GOOD, GOLD, PROMPT, 2.0),
        )

    def test_trainer_reward_path_uses_raw_ids(self):
        source = inspect.getsource(ThinkCompositeInterestRecGRPOTrainer._calculate_rewards)
        self.assertIn("decode_reward_completion", source)
        self.assertNotIn("completion_text", source)

    def test_checkpoint_schedule_has_no_700(self):
        self.assertEqual([step for step in range(1, 717) if should_save_checkpoint(step)],
                         list(CHECKPOINT_STEPS))
        self.assertFalse(should_save_checkpoint(700))
        self.assertFalse(should_save_checkpoint(720))
        self.assertGreater(AUTO_SAVE_STEPS, 716)
        self.assertGreaterEqual(SAVE_TOTAL_LIMIT, len(CHECKPOINT_STEPS))
        self.assertEqual(checkpoint_save_config(), {
            "save_strategy": "steps", "save_steps": 10000, "save_total_limit": 4,
        })

    def test_formal_path_does_not_call_dry_run_report(self):
        self.assertNotIn("dry_run_report", inspect.getsource(launch_training))
        self.assertEqual(frozen_contract()["reward"], "0.60*U_beam+0.40*U_cot")

    def test_candidate_parser_failure_is_zero_not_exception(self):
        row = score_candidate("invalid", GOLD, PROMPT, 2.0)
        self.assertFalse(row["parser_success"])
        self.assertEqual(row["cot_utility"], 0.0)

    def test_gold_parser_failure_is_hard(self):
        with self.assertRaises(RuntimeError):
            score_candidate(GOOD, "invalid", PROMPT, 2.0)

    def test_gold_leakage_guard(self):
        assert_gold_isolation({"prompt": PROMPT, "gold_cot": GOLD}, PROMPT, PROMPT)
        with self.assertRaisesRegex(RuntimeError, "GOLD_COT_LEAKAGE"):
            assert_gold_isolation({"prompt": PROMPT + GOLD, "gold_cot": GOLD})

    def test_runtime_uses_composite_not_raw_reward_advantage(self):
        runtime = build_global_runtime(records([GOOD, OTHER, GOOD, OTHER]), [2.0] * 4)
        expected = population_advantages([
            composite_reward(2.0, row["cot_utility"]) for row in runtime["candidates"]
        ])
        self.assertEqual(runtime["advantages"], expected)
        self.assertNotEqual(runtime["advantages"], [0.0] * 4)

    def test_zero_std_stays_exact_zero(self):
        runtime = build_global_runtime(records([GOOD] * 4), [2.0] * 4)
        self.assertEqual(runtime["advantages"], [0.0] * 4)

    def test_ddp_global_group_alignment_and_slice(self):
        rows = records([GOOD] * 4, "a") + records([OTHER] * 4, "b")
        runtime = build_global_runtime(rows, [2.0] * 8)
        self.assertEqual(len(runtime["groups"]), 2)
        self.assertEqual(slice_global(list(range(16)), 2, 4, 4), [8, 9, 10, 11])
        with self.assertRaises(RuntimeError):
            build_global_runtime(rows[:3] + rows[4:8], [2.0] * 7)

    def test_winner_categories_are_disjoint(self):
        tie = classify_winner([2.0, 2.0, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0])
        strict = classify_winner([2.0, 0.0, 0.0, 0.0], [0.4, 0.8, 0.0, 0.0])
        self.assertTrue(tie["top_set_tie_break"])
        self.assertFalse(tie["strict_beam_reversal"])
        self.assertFalse(strict["top_set_tie_break"])
        self.assertTrue(strict["strict_beam_reversal"])

    def test_candidate_monitor_schema_and_bounds(self):
        row = score_candidate(GOOD, GOLD, PROMPT, 16.0)
        required = {"beam_raw", "beam_utility", "matched_interest_count",
                    "interest_coverage", "interest_precision", "mean_match_similarity",
                    "coverage_tier", "match_quality", "cot_utility",
                    "composite_reward", "raw_n", "grounded_n", "grounding_coverage"}
        self.assertTrue(required.issubset(row))
        self.assertTrue(all(math.isfinite(row[key]) for key in
                            ("beam_utility", "cot_utility", "composite_reward")))

    def test_trainer_is_thin_direct_subclass(self):
        self.assertEqual(ThinkCompositeInterestRecGRPOTrainer.__bases__[0].__name__, "RecGRPOTrainer")
        source = inspect.getsource(ThinkCompositeInterestRecGRPOTrainer)
        self.assertNotIn("think_exact_clamp_advantages", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("_compute_loss", source)

    def test_probe_and_checkpoint_contract(self):
        self.assertEqual(SEED, 20260818)
        self.assertEqual(len(PROBE_IDS), 12)
        self.assertEqual(len(set(PROBE_IDS)), 12)
        self.assertEqual(CHECKPOINT_STEPS, (200, 400, 600, 716))
        self.assertEqual(PROBE_STEPS, (0, 200, 400, 600, 716))
        self.assertEqual(len(PROBE_ROUNDS), 3)
        self.assertTrue(all(len(round_ids) == 4 for round_ids in PROBE_ROUNDS))
        self.assertEqual(PROBE_DOMAIN_ORDER, ("video", "prod", "ad", "living"))

    def test_parser_parity_uses_shared_parser(self):
        self.assertTrue(extract_interest_units(GOOD, PROMPT).parser_success)

    def test_monitor_writer_has_dedicated_schema(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "scripts/monitor/writer.py").read_text(encoding="utf-8")
        self.assertIn("def write_composite", source)
        self.assertIn("composite_interest.jsonl", source)

    def test_gold_cot_is_not_forwarded_to_beam(self):
        from grpo_trl_trainer import make_think_reward_func
        seen = {}

        def beam_spy(
            prompts, completions, completion_ids, gold_sets, target_domains,
            recommendation_group_ids=None,
        ):
            seen["arguments"] = (
                prompts, completions, completion_ids, gold_sets, target_domains,
            )
            seen["group_ids"] = recommendation_group_ids
            return [2.0]

        reward = make_think_reward_func(beam_spy)
        result = reward(
            [PROMPT], [GOOD], [[1, 2]], route=["think"],
            all_gold_sids=[[SID]], gold_cot=[GOLD], target_domain=["video"],
            recommendation_group_id=["group-1"],
        )
        self.assertEqual(result, [2.0])
        self.assertEqual(len(seen["arguments"]), 5)
        self.assertNotIn(GOLD, seen["arguments"][0])
        self.assertNotIn(GOLD, seen["arguments"][3])
        self.assertEqual(seen["arguments"][4], ["video"])
        self.assertEqual(seen["group_ids"], ["group-1"])


if __name__ == "__main__":
    unittest.main()
