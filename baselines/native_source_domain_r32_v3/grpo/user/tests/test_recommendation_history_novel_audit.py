import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_recommendation_history_novel import (  # noqa: E402
    classify_task_conflict,
    group_records,
    summarize_domain,
)


def sid(domain, index):
    return f"<|{domain}_begin|><s_a_{index}><s_b_{index + 1}><s_c_{index + 2}>"


def rows(group_id, domain, history, gold):
    return [
        {
            "recommendation_group_id": group_id,
            "route": route,
            "target_domain": domain,
            "prompt": "history: " + ", ".join(history) + (" /think" if route == "think" else " /no_think"),
            "all_gold_sids": gold,
        }
        for route in ("think", "no_think")
    ]


class RecommendationHistoryNovelAuditTest(unittest.TestCase):
    def test_classifies_history_mixed_and_novel_groups(self):
        a, b, c = sid("video", 1), sid("video", 10), sid("video", 20)
        raw = (
            rows("history", "video", [a, b], [a, b])
            + rows("mixed", "video", [a], [a, b])
            + rows("novel", "video", [c], [a, b])
        )
        records, integrity = group_records(raw)
        self.assertTrue(integrity["route_parity"])
        self.assertEqual([record["category"] for record in records], ["history_only", "mixed", "novel_only"])
        summary = summarize_domain(records)
        self.assertAlmostEqual(summary["category"]["history_only_rate"], 1 / 3)
        self.assertAlmostEqual(summary["category"]["mixed_rate"], 1 / 3)
        self.assertAlmostEqual(summary["category"]["novel_only_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mean_history_gold_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_novel_gold_rate"], 0.5)
        self.assertAlmostEqual(summary["history_copy_recall_ceiling"]["median"], 0.5)
        self.assertAlmostEqual(summary["history_copy_recall_ceiling"]["zero_rate"], 1 / 3)
        self.assertAlmostEqual(summary["history_copy_recall_ceiling"]["below_0_25_rate"], 1 / 3)
        self.assertAlmostEqual(summary["history_copy_recall_ceiling"]["below_0_5_rate"], 1 / 3)

    def test_rejects_route_dependent_history(self):
        a, b = sid("prod", 1), sid("prod", 10)
        raw = rows("g", "prod", [a], [a])
        raw[1]["prompt"] = "history: " + b
        with self.assertRaisesRegex(ValueError, "route-dependent History"):
            group_records(raw)

    def test_rejects_cross_domain_gold(self):
        history = sid("prod", 1)
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            group_records(rows("g", "prod", [history], [sid("video", 1)]))

    def test_rejects_missing_route(self):
        a = sid("ad", 1)
        with self.assertRaisesRegex(ValueError, "exactly Think and NoThink"):
            group_records(rows("g", "ad", [a], [a])[:1])

    def test_conflict_requires_extractive_user_and_novel_recommendation(self):
        overall = {
            "mean_novel_gold_rate": 0.8,
            "category": {"novel_only_rate": 0.7},
        }
        user = {"gold_subset_history_rate": 1.0}
        self.assertEqual(classify_task_conflict(overall, user), "TASK_CONFLICT_SUPPORTED")
        user["gold_subset_history_rate"] = 0.5
        self.assertEqual(classify_task_conflict(overall, user), "TASK_CONFLICT_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
