import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_action_reward import score_action
from user_chain_reward import ordered_action_matching, score_chain
from user_common import parse_final_json, rouge_l_f1, set_f1


S1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
S2 = "<|prod_begin|><s_a_4><s_b_5><s_c_6>"
S3 = "<|ad_begin|><s_a_7><s_b_8><s_c_9>"
S4 = "<|living_begin|><s_a_10><s_b_11><s_c_12>"
HALL = "<|video_begin|><s_a_999><s_b_998><s_c_997>"


def action_sample(gold=(S1, S2), history=(S1, S2, S3, S4)):
    return {"gold_sids": list(gold), "history_sids": list(history)}


def event(date, action, logic):
    return {"date": date, "action": action, "logic": logic}


E1 = event("2026-01-01", f"[视频-长播] {S1}", "兴趣触发：观看内容")
E2 = event("2026-01-02", f"[商品-点击] {S2}", "需求深化：查看商品")
E3 = event("2026-01-03", "[搜索] 防晒用品", "决策补全：主动搜索")


def chain_sample(gold=(E1, E2, E3)):
    history = [
        {"date": E1["date"], "raw": E1["action"]},
        {"date": E2["date"], "raw": E2["action"]},
        {"date": E3["date"], "raw": E3["action"]},
        {"date": "2026-01-04", "raw": f"[广告-点击] {S3}"},
    ]
    return {"gold_events": list(gold), "history_events": history, "history_sids": [S1, S2, S3]}


def chain_json(events, name="test"):
    return json.dumps({"logic_chain": {"name": name, "events": events}}, ensure_ascii=False)


class ActionRewardScenarios(unittest.TestCase):
    def test_01_exact(self):
        result = score_action(json.dumps([S1, S2]), action_sample())
        self.assertEqual(result.reward, 1.0)

    def test_02_order_ignored(self):
        self.assertEqual(score_action(json.dumps([S2, S1]), action_sample()).reward, 1.0)

    def test_03_duplicate_does_not_change_f1(self):
        result = score_action(json.dumps([S1, S1, S2]), action_sample())
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.duplicate_sids, [S1])
        self.assertEqual(sum(v.kind == "duplicate_sid" for v in result.violations), 1)

    def test_04_duplicate_third_occurrence_each_penalized(self):
        result = score_action(json.dumps([S1, S1, S1]), action_sample(gold=(S1,)))
        self.assertEqual(sum(v.kind == "duplicate_sid" for v in result.violations), 2)

    def test_05_partial_recall(self):
        result = score_action(json.dumps([S1]), action_sample())
        self.assertAlmostEqual(result.reward, 2 / 3)

    def test_06_wrong_selection(self):
        result = score_action(json.dumps([S1, S3]), action_sample())
        self.assertEqual(result.wrong_selection_sids, [S3])
        self.assertFalse(result.hallucinated_sids)

    def test_07_hallucination(self):
        result = score_action(json.dumps([S1, HALL]), action_sample())
        self.assertEqual(result.hallucinated_sids, [HALL])
        violation = next(v for v in result.violations if v.kind == "hallucinated_sid")
        self.assertEqual(violation.metadata["first_invalid_component"], "a")

    def test_08_wrong_domain_prefix(self):
        alien = "<|ad_begin|><s_a_1><s_b_2><s_c_3>"
        result = score_action(json.dumps([alien]), action_sample(history=(S1,)))
        item = next(v for v in result.violations if v.kind == "hallucinated_sid")
        self.assertEqual(item.metadata["first_invalid_component"], "domain")

    def test_09_malformed_sid_invalidates_format(self):
        result = score_action('["<|video_begin|><s_a_1><s_b_2>"]', action_sample())
        self.assertEqual(result.reward, 0.0)
        self.assertIn("invalid_sid_string", result.parser_errors)

    def test_10_non_string_is_excluded_from_f1(self):
        result = score_action(f'["{S1}", 3]', action_sample())
        self.assertTrue(result.format_valid)
        self.assertAlmostEqual(result.reward, 2 / 3)
        self.assertIn("non_string_array_element", result.parser_errors)

    def test_11_top_level_object_invalid(self):
        result = score_action('{}', action_sample())
        self.assertIn("top_level_not_array", result.parser_errors)

    def test_12_extra_before_invalid_but_recoverable(self):
        result = score_action('answer: ["' + S1 + '"]', action_sample())
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.recoverable_sids, [S1])

    def test_13_extra_after_invalid(self):
        result = score_action(json.dumps([S1]) + " trailing", action_sample())
        self.assertIn("extra_text_after_json", result.parser_errors)

    def test_14_think_prefix_allowed(self):
        result = score_action("<think>x</think>\n" + json.dumps([S1]), action_sample(gold=(S1,)))
        self.assertEqual(result.reward, 1.0)

    def test_15_broken_json(self):
        result = score_action('["' + S1 + '"', action_sample())
        self.assertIn("malformed_json", result.parser_errors)

    def test_16_empty_correct_when_gold_empty(self):
        self.assertEqual(score_action("[]", action_sample(gold=())).reward, 1.0)

    def test_17_violation_char_span_is_local(self):
        text = json.dumps([S1, HALL])
        result = score_action(text, action_sample())
        item = next(v for v in result.violations if v.kind == "hallucinated_sid")
        self.assertEqual(text[item.char_start:item.char_end], HALL)

    def test_18_duplicate_hallucination_has_two_constraints(self):
        result = score_action(json.dumps([HALL, HALL]), action_sample())
        self.assertEqual(sum(v.kind == "hallucinated_sid" for v in result.violations), 2)
        self.assertEqual(sum(v.kind == "duplicate_sid" for v in result.violations), 1)

    def test_19_c_component_attribution(self):
        history = (
            "<|video_begin|><s_a_1><s_b_2><s_c_3>",
            "<|video_begin|><s_a_1><s_b_2><s_c_4>",
        )
        alien = "<|video_begin|><s_a_1><s_b_2><s_c_999>"
        result = score_action(json.dumps([alien]), action_sample(history=history))
        item = next(v for v in result.violations if v.kind == "hallucinated_sid")
        self.assertEqual(item.metadata["first_invalid_component"], "c")
        self.assertEqual(item.metadata["penalized_components"], ["c"])

    def test_20_b_component_attribution(self):
        alien = "<|video_begin|><s_a_1><s_b_999><s_c_3>"
        result = score_action(json.dumps([alien]), action_sample(history=(S1,)))
        item = next(v for v in result.violations if v.kind == "hallucinated_sid")
        self.assertEqual(item.metadata["first_invalid_component"], "b")
        self.assertEqual(item.metadata["penalized_components"], ["b", "c"])

    def test_21_non_sid_string_is_invalid_output_element(self):
        result = score_action(json.dumps([S1, "大静儿在北京"], ensure_ascii=False), action_sample(gold=(S1,)))
        self.assertEqual(result.reward, 1.0)
        self.assertIn("invalid_output_element", result.parser_errors)

    def test_22_unique_set_f1_contract_example(self):
        result = score_action(json.dumps([S1, S1, S2]), action_sample(gold=(S1, S2, S3)))
        self.assertAlmostEqual(result.f1, 0.8)
        self.assertEqual(result.duplicate_value_count, 1)
        self.assertEqual(result.duplicate_occurrence_count, 1)


class ChainRewardScenarios(unittest.TestCase):
    def test_01_exact(self):
        result = score_chain(chain_json([E1, E2, E3]), chain_sample())
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.format_valid)

    def test_02_delete_event_reduces_recall(self):
        result = score_chain(chain_json([E1, E2]), chain_sample())
        self.assertLess(result.reward, 1.0)

    def test_03_extra_event_reduces_precision(self):
        extra = event("2026-01-04", f"[广告-点击] {S3}", "扩展")
        result = score_chain(chain_json([E1, E2, E3, extra]), chain_sample())
        self.assertLess(result.reward, 1.0)

    def test_04_swapped_events_order_sensitive(self):
        result = score_chain(chain_json([E2, E1, E3]), chain_sample())
        self.assertLess(result.action_f1, 1.0)

    def test_05_logic_change_only(self):
        changed = [dict(E1, logic="完全无关"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertEqual(result.action_f1, 1.0)
        self.assertLess(result.logic_f1, 1.0)

    def test_06_action_change(self):
        changed = [dict(E1, action="[搜索] 其他"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertLess(result.action_f1, 1.0)

    def test_07_date_not_in_reward(self):
        changed = [dict(E1, date="2026-01-09"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(any(v.kind == "date_mismatch" for v in result.violations))

    def test_08_hallucinated_sid_hierarchy(self):
        changed = [dict(E1, action=f"[视频-长播] {HALL}"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        kinds = [v.kind for v in result.violations]
        self.assertIn("hallucinated_sid", kinds)
        self.assertNotIn("date_mismatch", kinds)

    def test_09_sid_date_mismatch(self):
        changed = [dict(E1, date="2026-01-09"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertIn("date_mismatch", [v.kind for v in result.violations])

    def test_10_sid_action_mismatch(self):
        changed = [dict(E1, action=f"[视频-点赞] {S1}"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertIn("action_mismatch", [v.kind for v in result.violations])

    def test_11_text_action_date_mismatch(self):
        changed = [E1, E2, dict(E3, date="2026-01-04")]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertIn("date_mismatch", [v.kind for v in result.violations])

    def test_12_text_action_mismatch(self):
        changed = [E1, E2, dict(E3, action="[搜索] 不存在")]
        result = score_chain(chain_json(changed), chain_sample())
        self.assertIn("action_mismatch", [v.kind for v in result.violations])

    def test_13_duplicate_event_second_only(self):
        result = score_chain(chain_json([E1, E1]), chain_sample(gold=(E1,)))
        items = [v for v in result.violations if v.kind == "duplicate_event"]
        self.assertEqual(len(items), 1)

    def test_14_chronology_violation(self):
        result = score_chain(chain_json([E2, E1]), chain_sample(gold=(E1, E2)))
        self.assertEqual(sum(v.kind == "chronology_violation" for v in result.violations), 1)

    def test_15_excess_sixth_event(self):
        events = [dict(E1, date=f"2026-01-0{i}", action=f"[搜索] x{i}") for i in range(1, 7)]
        result = score_chain(chain_json(events), chain_sample(gold=()))
        self.assertEqual(sum(v.kind == "excess_event" for v in result.violations), 1)

    def test_16_top_level_array_invalid(self):
        result = score_chain("[]", chain_sample())
        self.assertFalse(result.format_valid)
        self.assertIn("top_level_not_object", result.parser_errors)

    def test_17_missing_chain_invalid(self):
        result = score_chain("{}", chain_sample())
        self.assertIn("missing_logic_chain", result.parser_errors)

    def test_18_missing_field_invalid(self):
        broken = {"logic_chain": {"name": "x", "events": [{"date": "2026-01-01", "action": "x"}]}}
        result = score_chain(json.dumps(broken), chain_sample())
        self.assertIn("missing_logic", result.parser_errors)

    def test_19_bad_date_format(self):
        result = score_chain(chain_json([dict(E1, date="01/01/2026")]), chain_sample(gold=(E1,)))
        self.assertIn("invalid_date_format", result.parser_errors)

    def test_20_extra_text_invalid(self):
        result = score_chain("prefix " + chain_json([E1]), chain_sample(gold=(E1,)))
        self.assertFalse(result.format_valid)

    def test_21_merged_actions_ground(self):
        merged_action = E1["action"] + "；" + E2["action"]
        merged = event("2026-01-01", merged_action, "合并")
        sample = chain_sample(gold=(merged,))
        sample["history_events"][1]["date"] = "2026-01-01"
        result = score_chain(chain_json([merged]), sample)
        self.assertEqual(result.grounding_status[0]["status"], "grounded")

    def test_22_zero_similarity_unmatched(self):
        self.assertEqual(ordered_action_matching([{"action": "甲"}], [{"action": "乙"}]), [])

    def test_23_monotonic_dp_finds_ordered_pairs(self):
        predicted = [{"action": "x"}, {"action": "a"}, {"action": "b"}]
        gold = [{"action": "a"}, {"action": "b"}]
        matches = ordered_action_matching(predicted, gold)
        self.assertEqual([(i, j) for i, j, _ in matches], [(1, 0), (2, 1)])

    def test_24_duplicate_span_entire_second_event(self):
        text = chain_json([E1, E1])
        result = score_chain(text, chain_sample(gold=(E1,)))
        item = next(v for v in result.violations if v.kind == "duplicate_event")
        self.assertTrue(text[item.char_start:item.char_end].startswith("{"))

    def test_25_similar_logic_has_partial_score(self):
        changed = [dict(E1, logic="兴趣触发观看"), E2, E3]
        result = score_chain(chain_json(changed), chain_sample())
        first = next(item for item in result.matches if item.predicted_index == 0)
        self.assertGreater(first.logic_similarity, 0.0)
        self.assertLess(first.logic_similarity, 1.0)

    def test_26_same_sid_two_dates_is_legitimate(self):
        second = event("2026-01-02", f"[视频-长播] {S1}", "再次观看")
        sample = chain_sample(gold=(E1, second))
        sample["history_events"].append({"date": second["date"], "raw": second["action"]})
        result = score_chain(chain_json([E1, second]), sample)
        self.assertNotIn("duplicate_event", [v.kind for v in result.violations])
        self.assertTrue(all(item["status"] == "grounded" for item in result.grounding_status))

    def test_27_merged_action_partial_grounding(self):
        merged = event("2026-01-01", E1["action"] + "；[搜索] 不存在", "合并")
        result = score_chain(chain_json([merged]), chain_sample(gold=(merged,)))
        self.assertEqual(result.grounding_status[0]["status"], "partially_grounded")

    def test_28_repeated_text_field_spans_are_distinct(self):
        text = chain_json([E1, E1])
        result = score_chain(text, chain_sample(gold=(E1,)))
        first, second = result.event_spans
        self.assertGreater(second["fields"]["action"]["char_start"], first["fields"]["action"]["char_start"])

    def test_29_search_event_without_sid_grounded(self):
        result = score_chain(chain_json([E3]), chain_sample(gold=(E3,)))
        self.assertEqual(result.grounding_status[0]["status"], "grounded")

    def test_30_multiple_sids_in_merged_event(self):
        merged = event("2026-01-01", E1["action"] + "；" + E2["action"], "多 SID")
        sample = chain_sample(gold=(merged,))
        sample["history_events"][1]["date"] = "2026-01-01"
        result = score_chain(chain_json([merged]), sample)
        self.assertEqual(len(result.event_spans[0]["sids"]), 2)
        self.assertEqual(result.grounding_status[0]["status"], "grounded")

    def test_31_escaped_chinese_action_span(self):
        escaped = json.dumps({"logic_chain": {"name": "x", "events": [E3]}}, ensure_ascii=True)
        result = score_chain(escaped, chain_sample(gold=(E3,)))
        span = result.event_spans[0]["fields"]["action"]
        self.assertIn("\\u", escaped[span["char_start"]:span["char_end"]])


class CommonParserScenarios(unittest.TestCase):
    def test_json_string_escapes(self):
        parsed = parse_final_json('{"x":"a\\\"b"}')
        self.assertTrue(parsed.strict)
        self.assertEqual(parsed.value["x"], 'a"b')

    def test_action_similarity_sid_atomic(self):
        self.assertEqual(set_f1(S1, S1), 1.0)

    def test_rouge_order_sensitive(self):
        self.assertLess(rouge_l_f1("甲乙丙", "丙乙甲"), 1.0)


if __name__ == "__main__":
    unittest.main()
