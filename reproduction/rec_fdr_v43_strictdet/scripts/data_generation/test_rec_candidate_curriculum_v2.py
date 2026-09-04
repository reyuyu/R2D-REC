#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("rec_candidate_curriculum_v2_fullft_sft.py")
SPEC = importlib.util.spec_from_file_location("rec_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CandidateLossTests(unittest.TestCase):
    def test_single_positive_set_equals_ce(self):
        logits = torch.tensor([1.0, 2.0, -1.0])
        log_probs = torch.log_softmax(logits, dim=0)
        actual = MODULE.multi_positive_set_loss(log_probs, [1])
        self.assertTrue(torch.allclose(actual, -log_probs[1]))

    def test_multi_positive_set_reduces_loss(self):
        log_probs = torch.log_softmax(torch.tensor([1.0, 2.0, -1.0]), dim=0)
        single = MODULE.multi_positive_set_loss(log_probs, [1])
        multiple = MODULE.multi_positive_set_loss(log_probs, [0, 1])
        self.assertLess(float(multiple), float(single))

    def test_hit32_is_monotonic_in_positive_score(self):
        negatives = torch.linspace(3.0, -3.0, 96)
        low, _ = MODULE.sid_hit32_surrogate(torch.tensor([-2.0]), negatives)
        high, _ = MODULE.sid_hit32_surrogate(torch.tensor([2.0]), negatives)
        self.assertLess(float(high), float(low))

    def test_semantic_order_has_lower_loss(self):
        relevance = [1.0, .4, .1, 0.0]
        pairs = [(i, j) for i in range(4) for j in range(4) if relevance[i] > relevance[j]]
        ordered = MODULE.semantic_pair_loss(torch.tensor([3.0, 2.0, 1.0, 0.0]), relevance, pairs)
        reversed_loss = MODULE.semantic_pair_loss(torch.tensor([0.0, 1.0, 2.0, 3.0]), relevance, pairs)
        self.assertLess(float(ordered), float(reversed_loss))


class DatasetContractTests(unittest.TestCase):
    def test_catalog_and_bucket_contract(self):
        root = Path("/data/LLm-8B/code/train/data/rec_candidate_curriculum_v2")
        report = json.loads((root / "build_report.json").read_text())
        self.assertEqual(report["counts"]["rec_full_k1_train"] + report["counts"]["rec_full_k2_train"] + report["counts"]["rec_full_k3_train"], 45780)
        self.assertEqual(report["max_k"], 23)
        catalog = json.loads((root / "rec_group_catalog.json").read_text())
        self.assertEqual(len(catalog["groups"]), report["group_count"])
        for item in catalog["groups"]:
            self.assertEqual(len(item["positive_future_sids"]), item["k"])
            self.assertIn(item["entropy_bucket"], {"k1", "k2", "k3"})


if __name__ == "__main__":
    unittest.main()
