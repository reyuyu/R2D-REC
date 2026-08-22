import json
import sys
import tempfile
import unittest
from pathlib import Path


USER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = USER_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_seen_replay100_k4_ddp_v1 import (  # noqa: E402
    FROZEN_CONFIG,
    MCReplayError,
    build_manifest,
    load_replay_config,
    select_parent_prefix_rows,
)


CONFIG_PATH = USER_DIR / "configs" / "mc_user_seen_replay100_k4_ddp.json"


class Replay100ContractTests(unittest.TestCase):
    def test_frozen_config(self):
        self.assertEqual(load_replay_config(CONFIG_PATH), FROZEN_CONFIG)
        self.assertEqual(FROZEN_CONFIG["optimizer_state_policy"], "fresh_adamw")
        self.assertEqual(FROZEN_CONFIG["expected_optimizer_updates"], 100)

    def test_parent_prefix_is_exactly_first_100_in_original_order(self):
        parent_prompts = [
            {"prompt_step": index, "route": "action" if index % 2 else "chain", "sample_id": f"sample-{index:04d}"}
            for index in range(1, 513)
        ]
        train_rows = [
            {"sample_id": item["sample_id"], "route": item["route"], "payload": item["prompt_step"]}
            for item in reversed(parent_prompts)
        ]
        selected = select_parent_prefix_rows(train_rows, {"prompts": parent_prompts})
        self.assertEqual([row["sample_id"] for row in selected], [row["sample_id"] for row in parent_prompts[:100]])
        self.assertEqual([row["payload"] for row in selected], list(range(1, 101)))
        self.assertEqual(sum(row["route"] == "action" for row in selected), 50)
        self.assertEqual(sum(row["route"] == "chain" for row in selected), 50)

    def test_missing_or_reordered_source_fails_closed(self):
        parent = {"prompts": [{"sample_id": f"sample-{index}", "route": "action" if index % 2 else "chain"} for index in range(1, 101)]}
        rows = [{"sample_id": item["sample_id"], "route": item["route"]} for item in parent["prompts"][:-1]]
        with self.assertRaisesRegex(MCReplayError, "cannot be reconstructed"):
            select_parent_prefix_rows(rows, parent)
        rows.append({"sample_id": "sample-100", "route": "action"})
        with self.assertRaisesRegex(MCReplayError, "strict alternating"):
            select_parent_prefix_rows(rows, parent)

    def test_manifest_freezes_comparison_and_parent_contract(self):
        rows = [
            {"sample_id": f"sample-{index}", "route": "action" if index % 2 else "chain"}
            for index in range(1, 101)
        ]
        manifest = build_manifest("REPLAY", Path("config.json"), "c" * 64, FROZEN_CONFIG, rows, "d" * 40)
        self.assertEqual(manifest["parent_run_id"], FROZEN_CONFIG["parent_run_id"])
        self.assertTrue(manifest["replay_of_seen_samples"])
        self.assertEqual(manifest["optimizer_state_policy"], "fresh_adamw")
        self.assertEqual([row["source_prompt_step"] for row in manifest["prompts"]], list(range(1, 101)))
        self.assertEqual(manifest["comparison_contract"], "same_sample_order_same_prompt_step_same_candidate_seed_rule")

    def test_config_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            changed = dict(FROZEN_CONFIG, prompt_count=101)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(MCReplayError, "frozen Replay100 config mismatch"):
                load_replay_config(path)


if __name__ == "__main__":
    unittest.main()
