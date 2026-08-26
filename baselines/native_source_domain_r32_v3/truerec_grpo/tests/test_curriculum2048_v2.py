from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "data/build_curriculum2048_v2.py"
SPEC = importlib.util.spec_from_file_location("build_curriculum2048_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(group_id: str, domain: str, gold: list[tuple[int, int, int]], context: int = 1000):
    row = {
        "recommendation_group_id": group_id,
        "split": "train_pool",
        "target_domain": domain,
        "system": "system",
        "user_content_nothink": "prompt/no_think",
        "fixed_domain_token": f"<|{domain}_begin|>",
        "all_gold_abc": [f"<s_a_{a}><s_b_{b}><s_c_{c}>" for a, b, c in gold],
        "all_gold_sids": [f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>" for a, b, c in gold],
    }
    enriched, reasons = MODULE.static_eligibility(row, {group_id})
    assert not reasons
    enriched.update({
        "context_token_count": context,
        "context_length_percentile": 0.5,
        "context_distance_from_domain_median": 0.0,
        "context_outlier": False,
    })
    return enriched


def category_record(group_id: str, domain: str, category: str, index: int):
    if category == "A_RICH":
        gold = [(index * 2 + 1, 1, 1), (index * 2 + 2, 2, 2)]
    elif category == "B_RICH":
        gold = [(1, index * 2 + 1, 1), (1, index * 2 + 2, 2)]
    elif category == "C_RICH":
        gold = [(1, 1, index * 2 + 1), (1, 1, index * 2 + 2)]
    else:
        gold = [(index + 1, 1, 1)]
    return record(group_id, domain, gold, 900 + index)


class Curriculum2048Tests(unittest.TestCase):
    def test_frozen_train_split_sha(self):
        self.assertEqual(MODULE.EXPECTED_SPLIT_SHA256, {
            "train_pool_group_ids.json": "ef645cf62cf50f619b62f2d0fbfb267dd4f8df6c5cfc89c70a7977df3aecf724",
            "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
            "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
            "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
        })
        self.assertIn("context_requires_truncation", MODULE.REJECTION_REASONS)

    def test_gold_hierarchy_and_domain_validation(self):
        row = record("g", "video", [(1, 2, 3), (1, 2, 4), (5, 6, 7)])
        self.assertEqual((row["K_A"], row["K_AB"], row["K_ABC"]), (2, 2, 3))
        self.assertEqual(row["hierarchy_class"], "A_RICH")
        self.assertEqual(category_record("b", "video", "B_RICH", 1)["hierarchy_class"], "B_RICH")
        self.assertEqual(category_record("c", "video", "C_RICH", 1)["hierarchy_class"], "C_RICH")
        self.assertEqual(category_record("s", "video", "SINGLETON", 1)["hierarchy_class"], "SINGLETON")
        bad = dict(row)
        bad["all_gold_sids"] = [value.replace("video", "prod") for value in row["all_gold_sids"]]
        _, reasons = MODULE.static_eligibility(bad, {"g"})
        self.assertIn("gold_domain_mismatch", reasons)

    def test_stage_quota_contract(self):
        for domain in MODULE.DOMAINS:
            for stage in MODULE.STAGES:
                self.assertEqual(sum(MODULE.STAGE_CATEGORY_QUOTAS[domain][stage].values()), 128)
            self.assertEqual(MODULE.STAGE_CATEGORY_QUOTAS[domain]["stage1"]["A_RICH"], 128)
            self.assertGreater(MODULE.STAGE_CATEGORY_QUOTAS[domain]["stage4"]["A_RICH"], 0)
            self.assertGreater(MODULE.STAGE_CATEGORY_QUOTAS[domain]["stage4"]["B_RICH"], 0)

    def test_exact_every64_balance_and_disjointness(self):
        staged = {}
        train = set()
        for domain in MODULE.DOMAINS:
            pool = []
            totals = {
                category: sum(MODULE.STAGE_CATEGORY_QUOTAS[domain][stage][category] for stage in MODULE.STAGES)
                for category in ("A_RICH", "B_RICH", "C_RICH", "SINGLETON")
            }
            for category, count in totals.items():
                for index in range(count):
                    row = category_record(f"{domain}-{category}-{index}", domain, category, index)
                    pool.append(row)
                    train.add(row["recommendation_group_id"])
            staged[domain] = MODULE.assign_domain_stages(pool, domain)
        ordered = MODULE.build_epoch1_order(staged)
        audit = MODULE.validate_order(ordered, train, set(), set(), set())
        self.assertEqual(audit["every64_domain_balance"], "PASS")
        self.assertEqual(audit["probe_overlap"], 0)
        self.assertEqual([row["stage"] for row in ordered[:512]], ["stage1"] * 512)

    def test_stage1_is_all_a_rich_and_stage4_rehearses(self):
        domain = "living"
        pool = []
        for category, count in {"A_RICH": 143, "B_RICH": 13, "C_RICH": 7, "SINGLETON": 400}.items():
            pool.extend(category_record(f"{category}-{i}", domain, category, i) for i in range(count))
        staged = MODULE.assign_domain_stages(pool, domain)
        self.assertEqual({row["hierarchy_class"] for row in staged["stage1"]}, {"A_RICH"})
        stage4 = {name: sum(row["hierarchy_class"] == name for row in staged["stage4"]) for name in MODULE.HIERARCHY_CLASSES}
        self.assertEqual(stage4["A_RICH"], 2)
        self.assertEqual(stage4["B_RICH"], 2)
        self.assertEqual(stage4["C_RICH"], 3)


if __name__ == "__main__":
    unittest.main()
