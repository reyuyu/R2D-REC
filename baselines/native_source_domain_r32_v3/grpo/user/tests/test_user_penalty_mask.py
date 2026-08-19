import argparse
import json
import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_action_reward import score_action
from user_chain_reward import score_chain
from user_common import SID_RE
from user_penalty_mask import ACTION_MASK_KINDS, CHAIN_MASK_KINDS, compile_penalty_mask
from user_span_attribution import TokenSpanMapper


S1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
S2 = "<|prod_begin|><s_a_4><s_b_5><s_c_6>"
S3 = "<|ad_begin|><s_a_7><s_b_8><s_c_9>"


def event(date, action, logic="逻辑说明"):
    return {"date": date, "action": action, "logic": logic}


E1 = event("2026-01-01", f"[视频-长播] {S1}")
E2 = event("2026-01-02", f"[商品-点击] {S2}")


def action_sample(gold=(S1,), history=(S1,)):
    return {"gold_sids": list(gold), "history_sids": list(history)}


def chain_sample(gold=(E1, E2), history=None):
    if history is None:
        history = [{"date": item["date"], "raw": item["action"]} for item in gold]
    return {
        "gold_events": [dict(item) for item in gold],
        "history_events": [dict(item) for item in history],
        "history_sids": SID_RE.findall(" ".join(item["raw"] for item in history)),
    }


def chain_json(events):
    return json.dumps({"logic_chain": {"name": "test", "events": events}}, ensure_ascii=False)


def expected_mask(tokenizer, completion, char_spans):
    mapper = TokenSpanMapper(tokenizer, completion)
    expected = [False] * len(mapper.input_ids)
    for start, end in char_spans:
        span = mapper.map(start, end)
        for index in range(span.start, span.end):
            expected[index] = True
    return expected


def sid_component_span(completion, sid, component, occurrence=0):
    matches = [match for match in SID_RE.finditer(completion) if match.group() == sid]
    match = matches[occurrence]
    raw = match.group()
    boundaries = {
        "domain": (0, raw.index("><s_a_") + 1),
        "a": (raw.index("<s_a_"), raw.index("><s_b_") + 1),
        "b": (raw.index("<s_b_"), raw.index("><s_c_") + 1),
        "c": (raw.index("<s_c_"), len(raw)),
    }
    start, end = boundaries[component]
    return match.start() + start, match.start() + end


class MaskTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(PARENT_CHECKPOINT, local_files_only=True)
        if not cls.tokenizer.is_fast:
            raise RuntimeError("tests require exact tokenizer offsets")

    def compile_action(self, completion, sample):
        result = score_action(completion, sample, self.tokenizer)
        return result, compile_penalty_mask(completion, result.violations, self.tokenizer, "action")

    def compile_chain(self, completion, sample):
        result = score_chain(completion, sample, self.tokenizer)
        return result, compile_penalty_mask(completion, result.violations, self.tokenizer, "chain")

    def assert_mask(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual, expected)


class ActionMaskTests(MaskTestCase):
    def test_01_whitelist_exact(self):
        self.assertEqual(ACTION_MASK_KINDS, {"hallucinated_sid", "duplicate_sid"})

    def test_02_no_violation(self):
        completion = json.dumps([S1])
        _, compiled = self.compile_action(completion, action_sample())
        self.assert_mask(compiled["penalty_mask"], [False] * compiled["token_count"])

    def test_03_domain_invalid_masks_four_components(self):
        sid = "<|ad_begin|><s_a_1><s_b_2><s_c_3>"
        completion = json.dumps([sid])
        result, compiled = self.compile_action(completion, action_sample())
        self.assertEqual(result.violations[0].metadata["first_invalid_component"], "domain")
        spans = [sid_component_span(completion, sid, name) for name in ("domain", "a", "b", "c")]
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, spans))

    def test_04_a_invalid_masks_a_b_c(self):
        sid = "<|video_begin|><s_a_9><s_b_2><s_c_3>"
        completion = json.dumps([sid])
        result, compiled = self.compile_action(completion, action_sample())
        self.assertEqual(result.violations[0].metadata["first_invalid_component"], "a")
        spans = [sid_component_span(completion, sid, name) for name in ("a", "b", "c")]
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, spans))

    def test_05_b_invalid_masks_b_c(self):
        sid = "<|video_begin|><s_a_1><s_b_9><s_c_3>"
        completion = json.dumps([sid])
        result, compiled = self.compile_action(completion, action_sample())
        self.assertEqual(result.violations[0].metadata["first_invalid_component"], "b")
        spans = [sid_component_span(completion, sid, name) for name in ("b", "c")]
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, spans))

    def test_06_c_invalid_masks_only_c(self):
        sid = "<|video_begin|><s_a_1><s_b_2><s_c_9>"
        completion = json.dumps([sid])
        result, compiled = self.compile_action(completion, action_sample())
        self.assertEqual(result.violations[0].metadata["first_invalid_component"], "c")
        expected = expected_mask(self.tokenizer, completion, [sid_component_span(completion, sid, "c")])
        self.assert_mask(compiled["penalty_mask"], expected)
        self.assertEqual(sum(compiled["penalty_mask"]), 1)

    def test_07_second_and_third_duplicate_only(self):
        completion = json.dumps([S1, S1, S1])
        _, compiled = self.compile_action(completion, action_sample())
        matches = [match.span() for match in SID_RE.finditer(completion)]
        expected = expected_mask(self.tokenizer, completion, matches[1:])
        self.assert_mask(compiled["penalty_mask"], expected)
        self.assertEqual(sum(compiled["per_kind_masks"]["duplicate_sid"]), 8)

    def test_08_duplicate_hallucination_overlap_union_deduplicates(self):
        sid = "<|video_begin|><s_a_1><s_b_2><s_c_9>"
        completion = json.dumps([sid, sid])
        _, compiled = self.compile_action(completion, action_sample())
        all_matches = [match.span() for match in SID_RE.finditer(completion)]
        expected_union = expected_mask(
            self.tokenizer,
            completion,
            [sid_component_span(completion, sid, "c", 0), all_matches[1]],
        )
        self.assert_mask(compiled["penalty_mask"], expected_union)
        hall_expected = expected_mask(
            self.tokenizer,
            completion,
            [sid_component_span(completion, sid, "c", 0), sid_component_span(completion, sid, "c", 1)],
        )
        self.assert_mask(compiled["per_kind_masks"]["hallucinated_sid"], hall_expected)
        self.assert_mask(
            compiled["per_kind_masks"]["duplicate_sid"],
            expected_mask(self.tokenizer, completion, [all_matches[1]]),
        )
        self.assertEqual(sum(compiled["penalty_mask"]), 5)

    def test_09_wrong_selection_does_not_enter_mask(self):
        completion = json.dumps([S2])
        result, compiled = self.compile_action(completion, action_sample(gold=(S1,), history=(S1, S2)))
        self.assertIn("wrong_selection_sid", [item.kind for item in result.violations])
        self.assertFalse(any(compiled["penalty_mask"]))

    def test_10_format_violation_does_not_enter_mask(self):
        completion = json.dumps([S1]) + " trailing"
        result, compiled = self.compile_action(completion, action_sample())
        self.assertIn("extra_text_after_json", [item.kind for item in result.violations])
        self.assertFalse(any(compiled["penalty_mask"]))


class ChainMaskTests(MaskTestCase):
    def test_01_whitelist_exact(self):
        self.assertEqual(
            CHAIN_MASK_KINDS,
            {"hallucinated_sid", "date_mismatch", "action_mismatch", "duplicate_event", "chronology_violation", "excess_event"},
        )

    def test_02_hallucinated_sid_only(self):
        changed = dict(E1, action=E1["action"] + f"；[广告-点击] {S3}")
        completion = chain_json([changed, E2])
        _, compiled = self.compile_chain(completion, chain_sample())
        target = next(match.span() for match in SID_RE.finditer(completion) if match.group() == S3)
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, [target]))

    def test_03_date_mismatch_only_date_value(self):
        completion = chain_json([dict(E1, date="1900-01-01"), E2])
        result, compiled = self.compile_chain(completion, chain_sample())
        span = result.event_spans[0]["fields"]["date"]
        expected = expected_mask(self.tokenizer, completion, [(span["char_start"], span["char_end"])])
        self.assert_mask(compiled["penalty_mask"], expected)

    def test_04_action_mismatch_only_wrong_part(self):
        wrong = f"[视频-点赞] {S1}"
        completion = chain_json([dict(E1, action=wrong), E2])
        _, compiled = self.compile_chain(completion, chain_sample())
        start = completion.index(wrong)
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, [(start, start + len(wrong))]))

    def test_05_merged_action_masks_only_wrong_second_part(self):
        part1, part2 = E1["action"], f"[商品-点击] {S2}"
        gold = event("2026-01-01", part1 + "；" + part2)
        history = [{"date": "2026-01-01", "raw": part1}, {"date": "2026-01-01", "raw": part2}]
        wrong = f"[商品-购买] {S2}"
        completion = chain_json([dict(gold, action=part1 + "；" + wrong)])
        _, compiled = self.compile_chain(completion, chain_sample(gold=(gold,), history=history))
        start = completion.index(wrong)
        self.assert_mask(compiled["penalty_mask"], expected_mask(self.tokenizer, completion, [(start, start + len(wrong))]))

    def test_06_duplicate_event_masks_second_event(self):
        completion = chain_json([E1, E1])
        result, compiled = self.compile_chain(completion, chain_sample(gold=(E1,)))
        span = result.event_spans[1]["event"]
        expected = expected_mask(self.tokenizer, completion, [(span["char_start"], span["char_end"])])
        self.assert_mask(compiled["penalty_mask"], expected)

    def test_07_chronology_masks_causing_date(self):
        completion = chain_json([E2, E1])
        result, compiled = self.compile_chain(completion, chain_sample())
        span = result.event_spans[1]["fields"]["date"]
        expected = expected_mask(self.tokenizer, completion, [(span["char_start"], span["char_end"])])
        self.assert_mask(compiled["per_kind_masks"]["chronology_violation"], expected)

    def test_08_sixth_and_seventh_events_only(self):
        events = [event(f"2026-01-{index + 1:02d}", f"[搜索] item-{index}") for index in range(7)]
        sample = chain_sample(gold=events[:5], history=[{"date": item["date"], "raw": item["action"]} for item in events])
        completion = chain_json(events)
        result, compiled = self.compile_chain(completion, sample)
        spans = [result.event_spans[index]["event"] for index in (5, 6)]
        expected = expected_mask(self.tokenizer, completion, [(item["char_start"], item["char_end"]) for item in spans])
        self.assert_mask(compiled["per_kind_masks"]["excess_event"], expected)
        self.assert_mask(compiled["penalty_mask"], expected)

    def test_09_low_logic_reward_has_zero_mask(self):
        changed = [dict(E1, logic="完全错误"), dict(E2, logic="毫无关联")]
        completion = chain_json(changed)
        result, compiled = self.compile_chain(completion, chain_sample())
        self.assertLess(result.logic_f1, 1.0)
        self.assertFalse(any(compiled["penalty_mask"]))

    def test_10_schema_violation_has_zero_mask(self):
        completion = "{}"
        result, compiled = self.compile_chain(completion, chain_sample())
        self.assertTrue(result.parser_errors)
        self.assertFalse(any(compiled["penalty_mask"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    args, remaining = parser.parse_known_args()
    PARENT_CHECKPOINT = args.parent_checkpoint
    unittest.main(argv=[sys.argv[0], *remaining])
