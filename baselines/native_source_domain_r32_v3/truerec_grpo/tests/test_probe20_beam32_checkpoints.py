from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_probe20_beam32_checkpoints import (  # noqa: E402
    CONTRACT, checkpoint_inventory, generation_kwargs, summarize_groups,
)


class Probe20Beam32CheckpointTests(unittest.TestCase):
    def test_frozen_nothink_fixed_domain_abc3_contract(self):
        kwargs = generation_kwargs()
        self.assertEqual((kwargs["num_beams"], kwargs["num_return_sequences"]), (32, 32))
        self.assertEqual((kwargs["min_new_tokens"], kwargs["max_new_tokens"]), (3, 3))
        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["eos_token_id"], [151645, 151643])
        self.assertEqual(kwargs["pad_token_id"], 151643)
        self.assertEqual(CONTRACT["route"], "NoThink")
        self.assertIs(CONTRACT["empty_think"], True)
        self.assertIs(CONTRACT["fixed_domain_in_context"], True)
        self.assertIs(CONTRACT["bridge"], False)
        self.assertEqual(CONTRACT["action"], ["A", "B", "C"])

    def test_checkpoint_inventory_and_group_any_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for step in (512, 256):
                checkpoint = run / "checkpoints" / f"checkpoint-step-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "state.pt").write_bytes(b"state")
                (checkpoint / "metadata.json").write_text(
                    json.dumps({"model_state_sha256": f"sha-{step}"}), encoding="utf-8",
                )
            self.assertEqual([row["step"] for row in checkpoint_inventory(run)], [256, 512])

        groups = []
        for index in range(20):
            candidates = [{
                "format_valid": True, "A_hit": index < 15, "AB_hit": index < 10,
                "exact": index < 5,
            } for _ in range(32)]
            groups.append({
                "ANY_A_HIT": index < 15, "ANY_AB_HIT": index < 10,
                "ANY_EXACT": index < 5, "unique_valid_ABC_count": 4,
                "candidates": candidates,
            })
        summary = summarize_groups(groups, 12.5)
        self.assertEqual(summary["group_count"], 20)
        self.assertEqual(summary["candidate_count"], 640)
        self.assertEqual(summary["group_any_A_rate"], .75)
        self.assertEqual(summary["group_any_AB_rate"], .5)
        self.assertEqual(summary["group_any_exact_rate"], .25)
        self.assertEqual(summary["mean_unique_sid_at_32"], 4)


if __name__ == "__main__":
    unittest.main()
