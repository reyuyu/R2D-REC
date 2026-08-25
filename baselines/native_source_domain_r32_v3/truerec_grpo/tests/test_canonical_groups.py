from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "data/build_canonical_groups.py"
SPEC = importlib.util.spec_from_file_location("build_canonical_groups", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def sid(domain: str, a: int = 1, b: int = 2, c: int = 3) -> str:
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def normalized(group: str, route: str, domain: str = "video", golds=None, current=None, history_marker="same"):
    golds = golds or [sid(domain)]
    current = current or golds[0]
    return {
        "group_id": group,
        "route": route,
        "source_segment": "recommendation_cot" if route == "think" else "recommendation_nocot",
        "target_domain": domain,
        "current_gold_sid": module.parse_sid(current),
        "all_gold_sids": tuple(sorted(module.parse_sid(value) for value in golds)),
        "declared_group_size": len(golds),
        "history_text_sha256": history_marker,
        "history_sids": (module.parse_sid(sid("video", 9, 8, 7)),),
        "answer_suffix_class": "DIRECT_DOMAIN_SID",
        "answer_bridge": None,
    }


class CanonicalGroupsTest(unittest.TestCase):
    def test_sid_parser_and_four_domains(self):
        for domain in module.DOMAINS:
            self.assertEqual(module.parse_sid(sid(domain)), (domain, 1, 2, 3))
        with self.assertRaises(ValueError):
            module.parse_sid("not-a-sid")

    def test_same_group_think_nothink_gold_consistency(self):
        groups, conflicts = module.aggregate_normalized([normalized("g", "think"), normalized("g", "nothink")])
        self.assertEqual(len(groups), 1)
        self.assertFalse(conflicts["all_gold"])
        self.assertFalse(conflicts["current_gold"])

    def test_target_domain_conflict_detection(self):
        _, conflicts = module.aggregate_normalized([normalized("g", "think", "video"), normalized("g", "nothink", "ad")])
        self.assertEqual(len(conflicts["target_domain"]), 1)

    def test_gold_conflict_detection(self):
        _, conflicts = module.aggregate_normalized([
            normalized("g", "think", golds=[sid("video", 1, 2, 3)]),
            normalized("g", "nothink", golds=[sid("video", 4, 5, 6)]),
        ])
        self.assertEqual(len(conflicts["all_gold"]), 1)

    def test_history_ignores_route_and_task_template(self):
        history = "用户视频行为: 看过 " + sid("video", 9, 8, 7) + "。"
        think = history + "\n\n请预测目标内容。/think"
        nothink = history + "\n\n请直接回答。/no_think"
        self.assertEqual(module.extract_history_text(think, ""), module.extract_history_text(nothink, ""))

    def test_history_conflict_detected_from_behavior(self):
        rows = [normalized("g", "think", history_marker="one"), normalized("g", "nothink", history_marker="two")]
        _, conflicts = module.aggregate_normalized(rows)
        self.assertEqual(len(conflicts["history"]), 1)

    def test_canonical_group_unique_and_matches_source_unique_ids(self):
        rows = [normalized("a", "think"), normalized("a", "nothink"), normalized("b", "think")]
        groups, _ = module.aggregate_normalized(rows)
        ids = [group["recommendation_group_id"] for group in groups]
        self.assertEqual(len(groups), len(set(row["group_id"] for row in rows)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_answer_suffix_beta_bridge_vs_beta_gamma_direct(self):
        gold = sid("video")
        beta = module.classify_answer_suffix("<think>x</think>该用户最近喜欢的视频有: " + gold)
        gamma = module.classify_answer_suffix("<think>x</think>" + gold)
        self.assertEqual(beta["class"], "NATURAL_LANGUAGE_BRIDGE")
        self.assertEqual(gamma["class"], "DIRECT_DOMAIN_SID")


if __name__ == "__main__":
    unittest.main()
