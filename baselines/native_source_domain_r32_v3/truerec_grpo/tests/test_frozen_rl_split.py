from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "data" / "build_frozen_rl_split.py"
SPEC = importlib.util.spec_from_file_location("build_frozen_rl_split", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def group(index: int, domain: str = "video", novelty: str = "N0", k: int = 1):
    return {
        "recommendation_group_id": f"g{index:05d}",
        "target_domain": domain,
        "novelty": novelty,
        "K": k,
        "K_bucket": module.K_BUCKETS[0] if k == 1 else module.K_BUCKETS[1] if k == 2 else "K=3-5",
    }


def split_fixture():
    return [
        group(index, module.DOMAINS[index % 4], module.NOVELTY_CLASSES[index % 4], 1 if index % 3 else 2)
        for index in range(100)
    ]


def probe_fixture():
    rows = []
    index = 1000
    for domain in module.DOMAINS:
        for novelty, count in {"N0": 3, "N1": 2, "N2": 2, "H": 2}.items():
            for offset in range(count):
                rows.append(group(index, domain, novelty, offset + 1))
                index += 1
    return rows


class FrozenRLSplitTest(unittest.TestCase):
    def test_stable_sha_ordering_is_deterministic(self):
        ids = ["c", "a", "b"]
        first = sorted(ids, key=module.stable_score)
        second = sorted(reversed(ids), key=module.stable_score)
        self.assertEqual(first, second)

    def test_largest_remainder_exact_size(self):
        allocation = module.largest_remainder({"a": 5, "b": 3, "c": 2}, 7)
        self.assertEqual(sum(allocation.values()), 7)
        self.assertTrue(all(allocation[key] <= value for key, value in {"a": 5, "b": 3, "c": 2}.items()))

    def test_groups_do_not_cross_splits(self):
        split, _ = module.build_split(split_fixture(), final_size=20, dev_size=10)
        ids = {name: module.id_set(rows) for name, rows in split.items()}
        self.assertFalse(ids["train"] & ids["dev"])
        self.assertFalse(ids["train"] & ids["final"])
        self.assertFalse(ids["dev"] & ids["final"])

    def test_all_groups_covered_once(self):
        rows = split_fixture()
        split, _ = module.build_split(rows, final_size=20, dev_size=10)
        combined = [row["recommendation_group_id"] for values in split.values() for row in values]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(set(combined), module.id_set(rows))

    def test_k1_can_enter_train(self):
        split, _ = module.build_split(split_fixture(), final_size=20, dev_size=10)
        self.assertTrue(any(row["K"] == 1 for row in split["train"]))

    def test_probe_is_dev_subset(self):
        dev = probe_fixture()
        probe, _ = module.select_probe(dev)
        self.assertTrue(module.id_set(probe) <= module.id_set(dev))

    def test_probe_has_five_groups_per_domain(self):
        probe, _ = module.select_probe(probe_fixture())
        for domain in module.DOMAINS:
            self.assertEqual(sum(row["target_domain"] == domain for row in probe), 5)

    def test_probe_novelty_target_selection(self):
        probe, fallback = module.select_probe(probe_fixture())
        self.assertEqual(fallback, [])
        counts = {name: sum(row["novelty"] == name for row in probe) for name in module.NOVELTY_CLASSES}
        self.assertEqual(counts, {"H": 4, "N2": 4, "N1": 4, "N0": 8})

    def test_low_k_preference_is_deterministic(self):
        dev = probe_fixture()
        first, _ = module.select_probe(dev)
        second, _ = module.select_probe(list(reversed(dev)))
        self.assertEqual(first, second)
        for domain in module.DOMAINS:
            n0 = [row for row in first if row["target_domain"] == domain and row["novelty"] == "N0"]
            self.assertEqual([row["K"] for row in n0], [1, 2])

    def test_rerun_manifest_sha_identical(self):
        rows = split_fixture()
        split1, allocations1 = module.build_split(rows, final_size=20, dev_size=10)
        split2, allocations2 = module.build_split(list(reversed(rows)), final_size=20, dev_size=10)
        empty_probe = []
        source = {"sha256": "x", "path": "fixture"}
        one = module.manifest_payloads(split1, empty_probe, allocations1, source)
        two = module.manifest_payloads(split2, empty_probe, allocations2, source)
        self.assertEqual({k: module.value_sha(v) for k, v in one.items()}, {k: module.value_sha(v) for k, v in two.items()})


if __name__ == "__main__":
    unittest.main()
