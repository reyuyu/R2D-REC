from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "data" / "audit_gold_k_distribution.py"
SPEC = importlib.util.spec_from_file_location("audit_gold_k_distribution", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def sid(domain: str, value: int) -> str:
    return f"<|{domain}_begin|><s_a_{value}><s_b_{value + 1}><s_c_{value + 2}>"


class GoldKCensusTest(unittest.TestCase):
    def test_merge_group_deduplicates_group_id(self):
        groups = {}
        gold = module.normalize_gold([sid("video", 1)], "video")
        module.merge_group(groups, "g", "think", "video", gold)
        module.merge_group(groups, "g", "no_think", "video", gold)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups["g"]["source_rows"], 2)

    def test_think_nothink_are_one_group(self):
        groups = {}
        gold = module.normalize_gold([sid("ad", 1)], "ad")
        module.merge_group(groups, "g", "think", "ad", gold)
        module.merge_group(groups, "g", "no_think", "ad", gold)
        self.assertEqual(groups["g"]["routes"], {"think", "no_think"})
        self.assertEqual(module.census(list(groups.values()))["overall"]["N"], 1)

    def test_k_uses_unique_valid_gold(self):
        values = [sid("prod", 1), sid("prod", 1), sid("prod", 2)]
        self.assertEqual(len(module.normalize_gold(values, "prod")), 2)
        with self.assertRaises(module.AuditError):
            module.normalize_gold(["bad"], "prod")

    def test_bucket_boundaries(self):
        expected = {
            1: "K=1", 2: "K=2", 3: "K=3-5", 5: "K=3-5",
            6: "K=6-10", 10: "K=6-10", 11: "K=11+", 20: "K=11+",
        }
        self.assertEqual({k: module.k_bucket(k) for k in expected}, expected)

    def test_domain_counts_sum_to_overall(self):
        groups = []
        for index, domain in enumerate(module.DOMAINS):
            gold = module.normalize_gold([sid(domain, index + 1)], domain)
            groups.append({"target_domain": domain, "gold": gold})
        result = module.census(groups)
        self.assertEqual(result["overall"]["N"], 4)
        self.assertEqual(sum(value["N"] for value in result["domains"].values()), 4)

    def test_old_unique_group_count_contract(self):
        rows = []
        for group_id, domain in (("g1", "video"), ("g2", "living")):
            for route in ("think", "no_think"):
                rows.append({
                    "recommendation_group_id": group_id,
                    "route": route,
                    "target_domain": domain,
                    "all_gold_sids": [sid(domain, 1)],
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            groups, source = module.load_old(path, expected_groups=2)
            self.assertEqual(len(groups), 2)
            self.assertEqual(source["total_rows"], 4)
            with self.assertRaises(module.AuditError):
                module.load_old(path, expected_groups=3)


if __name__ == "__main__":
    unittest.main()
