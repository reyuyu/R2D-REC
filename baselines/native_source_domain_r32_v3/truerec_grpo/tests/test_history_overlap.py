from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "data" / "audit_history_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_history_overlap", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def parsed(domain: str, a: int, b: int, c: int):
    return module.parse_sid(f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>")


class HistoryOverlapTest(unittest.TestCase):
    def test_exact_abc_overlap(self):
        result = module.classify_overlap((parsed("video", 1, 2, 3),), (parsed("video", 1, 2, 3),))
        self.assertEqual(result["novelty"], "H")
        self.assertTrue(result["GoldABCInHistory"])

    def test_ab_only_overlap(self):
        result = module.classify_overlap((parsed("prod", 1, 2, 3),), (parsed("prod", 1, 2, 9),))
        self.assertEqual(result["novelty"], "N2")
        self.assertTrue(result["GoldABInHistory"])
        self.assertFalse(result["GoldABCInHistory"])

    def test_a_only_overlap(self):
        result = module.classify_overlap((parsed("ad", 1, 2, 3),), (parsed("ad", 1, 9, 9),))
        self.assertEqual(result["novelty"], "N1")
        self.assertTrue(result["GoldAInHistory"])
        self.assertFalse(result["GoldABInHistory"])

    def test_no_overlap(self):
        result = module.classify_overlap((parsed("living", 1, 2, 3),), (parsed("living", 9, 9, 9),))
        self.assertEqual(result["novelty"], "N0")
        self.assertFalse(result["GoldAInHistory"])

    def test_multi_positive_gold_is_existential(self):
        gold = (parsed("video", 1, 2, 3), parsed("video", 7, 8, 9))
        result = module.classify_overlap(gold, (parsed("video", 7, 8, 9),))
        self.assertEqual(result["novelty"], "H")

    def test_classes_are_mutually_exclusive_and_exhaustive(self):
        cases = [
            ((parsed("video", 1, 2, 3),), (parsed("video", 1, 2, 3),)),
            ((parsed("video", 1, 2, 3),), (parsed("video", 1, 2, 4),)),
            ((parsed("video", 1, 2, 3),), (parsed("video", 1, 4, 4),)),
            ((parsed("video", 1, 2, 3),), (parsed("video", 4, 4, 4),)),
        ]
        classes = [module.classify_overlap(gold, history)["novelty"] for gold, history in cases]
        self.assertEqual(classes, ["H", "N2", "N1", "N0"])
        rows = [{"target_domain": "video", "K": 1, "K_bucket": "K=1", **module.classify_overlap(g, h)} for g, h in cases]
        census, _ = module.build_census(rows)
        self.assertEqual(census["class_exclusive"], "PASS")
        self.assertEqual(census["class_exhaustive"], "PASS")
        self.assertEqual(sum(census["overall"]["novelty"][name]["count"] for name in module.NOVELTY_CLASSES), 4)

    def test_think_nothink_group_dedup(self):
        groups = {}
        gold = (parsed("video", 1, 2, 3),)
        history = (parsed("video", 9, 9, 9),)
        module.merge_group(groups, "g", "recommendation_cot", "video", gold, history)
        module.merge_group(groups, "g", "recommendation_nocot", "video", gold, history)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups["g"]["source_rows"], 2)

    def test_history_extraction_ignores_route_marker(self):
        sid = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
        think = module.extract_history_text("任务\n\n历史 " + sid + "\n\n请预测 /think", "")
        no_think = module.extract_history_text("任务\n\n历史 " + sid + "\n\n不同模板 /no_think", "")
        self.assertEqual(think, no_think)


if __name__ == "__main__":
    unittest.main()
