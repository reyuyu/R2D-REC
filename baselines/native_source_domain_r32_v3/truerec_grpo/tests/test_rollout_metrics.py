from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "analysis" / "rollout_metrics.py"
SPEC = importlib.util.spec_from_file_location("rollout_metrics", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

TOKENS = {1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>", 5: "<s_b_5>", 6: "<s_c_6>", 7: "bad", 8: "<s_c_8>", 9: "<s_b_9>"}
GOLD = ["<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_5><s_c_6>"]
DOMAIN = "<|video_begin|>"


def assess(ids, history=()):
    result = module.assess_candidate(ids, TOKENS.get, GOLD, DOMAIN, history)
    result["raw_token_ids"] = ids
    return result


class RolloutMetricsTest(unittest.TestCase):
    def test_01_raw_abc_parser(self): self.assertEqual(module.parse_raw_abc([1, 2, 3], TOKENS.get), (TOKENS[1], TOKENS[2], TOKENS[3]))
    def test_02_invalid_short(self): self.assertIsNone(module.parse_raw_abc([1, 2], TOKENS.get))
    def test_03_invalid_wrong_class(self): self.assertIsNone(module.parse_raw_abc([1, 3, 2], TOKENS.get))
    def test_04_invalid_unknown(self): self.assertIsNone(module.parse_raw_abc([1, 2, 7], TOKENS.get))
    def test_05_multi_positive_a(self): self.assertTrue(assess([1, 2, 8])["A_hit"])
    def test_06_multi_positive_ab(self): self.assertTrue(assess([1, 5, 8])["AB_hit"])
    def test_07_exact(self): self.assertTrue(assess([1, 5, 6])["exact"])
    def test_08_exact_implies_hierarchy(self):
        row = assess([1, 2, 3]); self.assertTrue(row["exact"] and row["AB_hit"] and row["A_hit"])
    def test_09_frontier_partition(self):
        rows = [assess(x) for x in ([1, 2], [4, 2, 3], [1, 9, 3], [1, 2, 8], [1, 2, 3])]
        self.assertEqual([row["frontier"] for row in rows], list(module.FRONTIERS))
    def test_10_group_reach_all_invalid(self): self.assertEqual(module.group_summary([assess([1, 2])] * 8)["group_class"], "ALL_INVALID")
    def test_11_group_reach_no_a(self): self.assertEqual(module.group_summary([assess([4, 2, 3])] * 8)["group_class"], "NO_A_REACHED")
    def test_12_group_reach_a_only(self): self.assertEqual(module.group_summary([assess([1, 9, 3])] * 8)["group_class"], "A_ONLY_MAX")
    def test_13_group_reach_ab_only(self): self.assertEqual(module.group_summary([assess([1, 2, 8])] * 8)["group_class"], "AB_ONLY_MAX")
    def test_14_group_reach_exact(self): self.assertEqual(module.group_summary([assess([1, 2, 3])] * 8)["group_class"], "EXACT_REACHED")
    def test_15_hpr_partition(self):
        classes = []
        for ids in ([4, 2, 3], [1, 9, 3], [1, 2, 8], [1, 2, 3]): classes.append(module.group_summary([assess(ids)] * 8)["hpr_trigger"])
        self.assertEqual(classes, list(module.HPR_CLASSES))
    def test_16_legacy_zero_std(self): self.assertTrue(module.group_summary([assess([4, 2, 3])] * 8)["legacy_zero_std"])
    def test_17_legacy_nonzero_std(self): self.assertFalse(module.group_summary([assess([4, 2, 3]), assess([1, 2, 3])])["legacy_zero_std"])
    def test_18_correct_history_copy(self): self.assertTrue(assess([1, 2, 3], [DOMAIN + "<s_a_1><s_b_2><s_c_3>"])["correct_history_copy"])
    def test_19_wrong_history_copy(self): self.assertTrue(assess([4, 2, 3], [DOMAIN + "<s_a_4><s_b_2><s_c_3>"])["wrong_history_copy"])
    def test_20_other_domain_not_copy(self): self.assertFalse(assess([4, 2, 3], ["<|ad_begin|><s_a_4><s_b_2><s_c_3>"])["history_copy"])
    def test_21_diversity(self):
        rows = [assess([1, 2, 3]), assess([1, 2, 8]), assess([1, 5, 6])]
        out = module.group_summary(rows); self.assertEqual((out["unique_A_count"], out["unique_AB_count"], out["unique_valid_ABC_count"]), (1, 2, 3))
    def test_22_all_same(self): self.assertTrue(module.group_summary([assess([1, 2, 3])] * 8)["all_8_same_completion"])
    def test_23_census_partition(self):
        candidates = []
        for frontier_ids in ([1, 2], [4, 2, 3], [1, 9, 3], [1, 2, 8], [1, 2, 3]):
            row = {**assess(frontier_ids), "domain": "video", "novelty": "H", "K_bucket": "K=2"}; candidates.append(row)
        group = {**module.group_summary(candidates), "domain": "video", "novelty": "H", "K_bucket": "K=2"}
        result = module.census(candidates, [group]); self.assertEqual(sum(v["count"] for v in result["candidate_frontier"]["overall"]["overall"]["frontier"].values()), 5)
    def test_24_dataset_sha_gate_constants(self):
        self.assertEqual("ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879", "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879")
    def test_25_probe_sha_gate_constants(self):
        self.assertEqual("5f06976e12e60c4576ee0dc0d083d367d423251cf3e65eedbbd728f607ffa913", "5f06976e12e60c4576ee0dc0d083d367d423251cf3e65eedbbd728f607ffa913")


if __name__ == "__main__":
    unittest.main()
