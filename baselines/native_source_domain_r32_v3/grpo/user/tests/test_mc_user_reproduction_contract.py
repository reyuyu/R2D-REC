import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MC_USER_REPRO_ROOT", "/tmp/mc-user-repro")
os.environ.setdefault("MC_USER_PARENT_ADAPTER", "/tmp/mc-user-repro/parent")
os.environ.setdefault(
    "MC_USER_TRAIN_DATA",
    "/root/reproduce_datasets/onereason_final_chain_20260901/03_user_grpo/train_3000.jsonl",
)

from run_mc_user_formal_hybrid_k4_ddp_v1 import REPRO_STEP100_CONFIG, load_k4_config


class ReproductionContractTest(unittest.TestCase):
    def test_step100_contract(self):
        self.assertEqual(REPRO_STEP100_CONFIG["prompt_count"], 100)
        self.assertEqual(REPRO_STEP100_CONFIG["action_count"], 50)
        self.assertEqual(REPRO_STEP100_CONFIG["chain_count"], 50)
        self.assertEqual(REPRO_STEP100_CONFIG["checkpoint_steps"], [25, 50, 75, 100])
        self.assertEqual(REPRO_STEP100_CONFIG["selection_seed"], 20260823)
        self.assertEqual(REPRO_STEP100_CONFIG["learning_rate"], 3e-7)
        self.assertEqual(REPRO_STEP100_CONFIG["sequence_weight"], 1.0)
        self.assertEqual(REPRO_STEP100_CONFIG["local_weight"], 0.3)

    def test_dynamic_paths_are_frozen_by_rendered_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(REPRO_STEP100_CONFIG), encoding="utf-8")
            self.assertEqual(load_k4_config(path), REPRO_STEP100_CONFIG)
            self.assertTrue(REPRO_STEP100_CONFIG["train_data"].startswith("/root/reproduce_datasets/"))


if __name__ == "__main__":
    unittest.main()
