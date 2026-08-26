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


class Curriculum2048Tests(unittest.TestCase):
    def test_frozen_train_split_sha(self):
        self.assertEqual(MODULE.EXPECTED_SPLIT_SHA256, {
            "train_pool_group_ids.json": "ef645cf62cf50f619b62f2d0fbfb267dd4f8df6c5cfc89c70a7977df3aecf724",
            "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
            "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
            "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
        })

    def test_gold_hierarchy_and_domain_validation(self):
        row = record("g", "video", [(1, 2, 3), (1, 2, 4), (5, 6, 7)])
        self.assertEqual((row["K_A"], row["K_AB"], row["K_ABC"]), (2, 2, 3))
        bad = dict(row)
        bad["all_gold_sids"] = [value.replace("video", "prod") for value in row["all_gold_sids"]]
        _, reasons = MODULE.static_eligibility(bad, {"g"})
        self.assertIn("gold_domain_mismatch", reasons)

    def test_rich_stage_allocation_is_smooth_and_retained(self):
        for rich_count in (163, 322, 366, 384):
            counts = MODULE.rich_stage_counts(rich_count)
            self.assertEqual(sum(counts), rich_count)
            self.assertTrue(all(0 < value <= 128 for value in counts))
            self.assertGreaterEqual(counts[0], counts[-1])

    def test_exact_every64_balance_and_disjointness(self):
        staged = {}
        train = set()
        for domain_index, domain in enumerate(MODULE.DOMAINS):
            selected = []
            for index in range(512):
                gold = [(index + 1, 2, 3), (index + 2, 4, 5)] if index < 384 else [(index + 1, 2, 3)]
                row = record(f"{domain}-{index}", domain, gold, 900 + index)
                row["quality_rank_within_domain"] = index
                row["selection_tier"] = "prefix_rich" if row["prefix_rich"] else "hard_fill"
                selected.append(row)
                train.add(row["recommendation_group_id"])
            staged[domain] = MODULE.assign_domain_stages(selected)
        ordered = MODULE.build_epoch1_order(staged)
        audit = MODULE.validate_order(ordered, train, set(), set(), set())
        self.assertEqual(audit["every64_domain_balance"], "PASS")
        self.assertEqual(audit["probe_overlap"], 0)
        self.assertEqual([row["stage"] for row in ordered[:512]], ["stage1"] * 512)

    def test_selection_prefers_hierarchy_but_keeps_hard_groups(self):
        rows = []
        for index in range(600):
            gold = [(index + 1, 2, 3), (index + 2, 4, 5)] if index < 400 else [(index + 1, 2, 3)]
            rows.append(record(f"g-{index}", "video", gold))
        selected = MODULE.select_domain(rows)
        self.assertEqual(len(selected), 512)
        self.assertEqual(sum(row["prefix_rich"] for row in selected), 384)
        self.assertEqual(sum(not row["prefix_rich"] for row in selected), 128)


if __name__ == "__main__":
    unittest.main()
