from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from beta_single_group_rollout import independently_reconstruct_old_logps, select_single_group, selector_digest
from rollout_runtime_v1 import capture_sampled_logps, trim_generated_completion


class BetaSingleGroupRolloutTest(unittest.TestCase):
    def test_01_eos_list_handling(self): self.assertEqual(trim_generated_completion([7, 151645, 9], [151645, 151643], 151643), [7, 151645])
    def test_02_pad_also_eos_handling(self): self.assertEqual(trim_generated_completion([7, 151643, 151643], [151645, 151643], 151643), [7, 151643])
    def test_03_pad_trim_without_eos(self): self.assertEqual(trim_generated_completion([7, 8, 0], None, 0), [7, 8])
    def test_04_deterministic_selector(self):
        rows = [{"recommendation_group_id": f"g{index}"} for index in range(4096)]
        expected = min(rows, key=lambda row: (hashlib.sha256(("phase1.2b|20260825|" + row["recommendation_group_id"]).encode()).hexdigest(), row["recommendation_group_id"]))
        self.assertEqual(select_single_group(rows), expected)
    def test_05_selector_order_independent(self):
        rows = [{"recommendation_group_id": f"g{index}"} for index in range(4096)]
        self.assertEqual(select_single_group(rows)["recommendation_group_id"], select_single_group(list(reversed(rows)))["recommendation_group_id"])
    def test_06_independent_reconstruction(self):
        torch.manual_seed(12); scores = torch.randn(8, 3, 24); ids = [[1, 2, 3]] * 8
        reference = independently_reconstruct_old_logps(scores, ids); runtime = capture_sampled_logps(scores, ids)
        self.assertEqual(reference, runtime)
    def test_07_variable_completion_reconstruction(self):
        torch.manual_seed(13); scores = torch.randn(8, 3, 24); ids = [[1], [1, 2], [1, 2, 3], [2], [2, 3], [3, 4, 5], [4], [5, 6]]
        self.assertEqual(independently_reconstruct_old_logps(scores, ids), capture_sampled_logps(scores, ids))
    def test_08_phase12a_init_unchanged(self):
        expected = "5bd72d6c1b9a51d89148a7e1ac4f2b37f66cd7e03ef53d14168a35d690dc84ec"
        blob = __import__("subprocess").check_output(["git", "-C", str(ROOT.parents[2]), "show", "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/initialization/beta_baseline_init.py"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), expected)


if __name__ == "__main__": unittest.main()
