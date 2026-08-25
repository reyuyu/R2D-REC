from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "data" / "build_pilot4096.py"
SPEC = importlib.util.spec_from_file_location("build_pilot4096", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def record(index: int, domain: str, novelty: str, k: int):
    bucket = "K=1" if k == 1 else "K=2" if k == 2 else "K=3-5" if k <= 5 else "K=6-10" if k <= 10 else "K=11+"
    return {
        "recommendation_group_id": f"g{index:06d}", "split": "train_pool",
        "target_domain": domain, "novelty": novelty, "K": k, "K_bucket": bucket,
        "system": "s", "user_content_nothink": "u /no_think",
        "fixed_domain_token": f"<|{domain}_begin|>",
        "all_gold_sids": [f"<|{domain}_begin|><s_a_1><s_b_2><s_c_3>"],
        "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"], "history_sids": [],
    }


def fixture():
    rows = []
    index = 0
    for domain in module.DOMAINS:
        for offset in range(1100):
            novelty = module.NOVELTY_CLASSES[offset % 4]
            k = 1 + (offset % 12)
            rows.append(record(index, domain, novelty, k))
            index += 1
    return rows


class Pilot4096Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = fixture()
        cls.pilot, cls.quota = module.select_pilot(cls.train)

    def test_exact_4096_groups(self):
        self.assertEqual(len(self.pilot), 4096)

    def test_exact_1024_per_domain(self):
        for domain in module.DOMAINS:
            self.assertEqual(sum(row["target_domain"] == domain for row in self.pilot), 1024)

    def test_domain_novelty_largest_remainder(self):
        for domain in module.DOMAINS:
            self.assertEqual(sum(self.quota["pilot_cell_quota"][domain].values()), 1024)
            self.assertEqual(self.quota["pilot_cell_quota"][domain], {"H": 256, "N2": 256, "N1": 256, "N0": 256})

    def test_k_weight_formula(self):
        self.assertEqual(module.k_weight(1), 1.0)
        self.assertAlmostEqual(module.k_weight(2), 2**0.5)
        self.assertAlmostEqual(module.k_weight(3), 3**0.5)
        self.assertEqual(module.k_weight(4), 2.0)
        self.assertEqual(module.k_weight(100), 2.0)

    def test_deterministic_sha_uniform(self):
        first = module.deterministic_uniform("g")
        self.assertEqual(first, module.deterministic_uniform("g"))
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)

    def test_deterministic_weighted_priority(self):
        row = record(1, "video", "N0", 3)
        self.assertEqual(module.weighted_priority(row), module.weighted_priority(dict(row)))

    def test_no_duplicate_groups(self):
        self.assertEqual(len(module.id_set(self.pilot)), 4096)

    def test_pilot_subset_train(self):
        self.assertTrue(module.id_set(self.pilot) <= module.id_set(self.train))

    def test_no_heldout_contamination(self):
        audit = module.safety_audit(self.train, self.pilot, {"dev"}, {"final"}, {"probe"}, self.quota)
        self.assertEqual(audit["pilot_dev_overlap"], 0)
        self.assertEqual(audit["pilot_final_overlap"], 0)
        self.assertEqual(audit["pilot_probe_overlap"], 0)

    def test_k_weighting_does_not_change_novelty_quota(self):
        for domain in module.DOMAINS:
            actual = {
                novelty: sum(row["target_domain"] == domain and row["novelty"] == novelty for row in self.pilot)
                for novelty in module.NOVELTY_CLASSES
            }
            self.assertEqual(actual, self.quota["pilot_cell_quota"][domain])

    def test_k1_remains_eligible(self):
        self.assertTrue(any(row["K"] == 1 for row in self.pilot))

    def test_record_payload_no_mutation(self):
        by_id = {row["recommendation_group_id"]: row for row in self.train}
        self.assertTrue(all(
            module.record_payload_sha(row) == module.record_payload_sha(by_id[row["recommendation_group_id"]])
            for row in self.pilot
        ))

    def test_rerun_manifest_sha_identical(self):
        second, second_quota = module.select_pilot(list(reversed(self.train)))
        self.assertEqual(
            module.value_sha([row["recommendation_group_id"] for row in self.pilot]),
            module.value_sha([row["recommendation_group_id"] for row in second]),
        )
        self.assertEqual(self.quota, second_quota)


if __name__ == "__main__":
    unittest.main()
