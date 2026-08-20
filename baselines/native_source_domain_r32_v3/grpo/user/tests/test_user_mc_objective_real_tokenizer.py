import math
import sys
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_real_smoke import (  # noqa: E402
    BASE_MODEL,
    TRAIN_DATA,
    prepare_fixed_data,
    read_jsonl,
)
from user_mc_objective import mc_unit_credit_loss  # noqa: E402


@unittest.skipUnless(
    Path(BASE_MODEL).is_dir() and Path(TRAIN_DATA).is_file(),
    "production tokenizer and frozen train_3000 are required",
)
class RealTokenizerChainOverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, local_files_only=True
        )
        cls.fixed = prepare_fixed_data(cls.tokenizer, read_jsonl(TRAIN_DATA))

    def test_fixed_chain_overlap_is_algebraically_supported(self):
        rollout = self.fixed["rollouts"]["chain"]
        batch = self.fixed["batches"]["chain"]
        mask = batch["completion_mask"].to(dtype=torch.float64)
        logps = torch.linspace(
            -4.0,
            -1.0,
            steps=mask.numel(),
            dtype=torch.float64,
        ).reshape_as(mask)
        logps.requires_grad_(True)
        loss, metadata = mc_unit_credit_loss(
            logps, rollout["credit_units_per_candidate"], mask
        )
        loss.backward()

        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(torch.isfinite(logps.grad).all())
        self.assertEqual(metadata["overlap_token_count"], 4)
        self.assertEqual(metadata["same_sign_overlap_token_count"], 3)
        self.assertEqual(metadata["mixed_sign_overlap_token_count"], 1)
        self.assertEqual(metadata["max_active_units_per_token"], 2)
        token_76 = next(
            record
            for record in metadata["overlap_records"]
            if record["candidate_index"] == 1 and record["token_index"] == 76
        )
        self.assertEqual(token_76["unit_indices"], [0, 1])
        mixed = [
            record
            for record in metadata["overlap_records"]
            if min(record["deltas"]) < 0.0 < max(record["deltas"])
        ]
        self.assertEqual(len(mixed), 1)
        self.assertTrue(math.isfinite(mixed[0]["net_logp_coefficient"]))


if __name__ == "__main__":
    unittest.main()
