import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_user_pilot150 import (
    ACTION_BUCKETS,
    CHAIN_BUCKETS,
    build_training_plan,
    continuation_decision,
    load_resume_contract,
    select_pilot_rows,
    validate_restored_optimizer,
)


def fixture_rows():
    rows = []
    action_values = {"1-5": 3, "6-10": 8, "11-20": 15, "21-30": 25, "31-40": 35, "41+": 45}
    action_totals = {"1-5": 45, "6-10": 60, "11-20": 90, "21-30": 60, "31-40": 30, "41+": 15}
    for label, _low, _high, _count in ACTION_BUCKETS:
        rows.extend({
            "sample_id": f"a-{label}-{index:03d}",
            "route": "action",
            "gold_sid_count": action_values[label],
            "gold_event_count": 0,
            "prompt_token_count": 100 + index * 31,
        } for index in range(action_totals[label]))
    chain_totals = {"2": 45, "3": 165, "4": 75, "5": 15}
    for label, events, _count in CHAIN_BUCKETS:
        rows.extend({
            "sample_id": f"c-{label}-{index:03d}",
            "route": "chain",
            "gold_sid_count": 0,
            "gold_event_count": events,
            "prompt_token_count": 200 + index * 37,
        } for index in range(chain_totals[label]))
    return rows


class Pilot300ContinuationTests(unittest.TestCase):
    def test_second_selection_excludes_first_and_reaches_300_unique(self):
        rows = fixture_rows()
        first, _ = select_pilot_rows(rows, 20260820)
        first_ids = {row["sample_id"] for route in first.values() for row in route}
        second, audit = select_pilot_rows(rows, 20260840, first_ids)
        second_ids = {row["sample_id"] for route in second.values() for row in route}
        self.assertEqual(len(first_ids), 150)
        self.assertEqual(len(second_ids), 150)
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(len(first_ids | second_ids), 300)
        self.assertEqual(audit["selected_excluded_overlap"], 0)
        self.assertEqual(audit["bucket_counts"]["action"], {
            "1-5": 15, "6-10": 15, "11-20": 15, "21-30": 14, "31-40": 13, "41+": 3,
        })
        self.assertEqual(audit["bucket_counts"]["chain"], {"2": 25, "3": 25, "4": 25, "5": 0})
        quota = audit["bucket_exhaustion_and_redistribution"]
        self.assertEqual(quota["action"]["exhausted_buckets"], ["41+"])
        self.assertEqual(quota["chain"]["exhausted_buckets"], ["5"])

    def test_continuation_plan_uses_global_steps_21_through_40(self):
        selected, _ = select_pilot_rows(fixture_rows(), 20260820)
        plan = build_training_plan(selected, starting_step=20)
        self.assertEqual([item["step"] for item in plan], list(range(21, 41)))
        self.assertEqual([item["route"] for item in plan], ["action", "chain"] * 10)

    def test_resume_contract_requires_150_prompts_and_step20(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("adapter_model.safetensors", "optimizer.pt"):
                (checkpoint / name).write_bytes(b"fixture")
            (checkpoint / "metadata.json").write_text(json.dumps({
                "processed_unique_prompts": 150,
                "optimizer_steps": 20,
            }), encoding="utf-8")
            ids = {"action": [f"a-{i}" for i in range(75)], "chain": [f"c-{i}" for i in range(75)]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"selected_sample_ids": ids}), encoding="utf-8")
            info = load_resume_contract({"resume": {
                "run_id": "first",
                "checkpoint": str(checkpoint),
                "source_manifest": str(manifest),
                "cumulative_start_prompts": 150,
                "cumulative_start_step": 20,
            }})
            self.assertEqual(len(info["first_flat_ids"]), 150)

    def test_optimizer_state_step_is_validated(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        for _ in range(20):
            optimizer.zero_grad()
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        audit = validate_restored_optimizer(optimizer, 20)
        self.assertEqual(audit["steps"], [20])
        with self.assertRaises(RuntimeError):
            validate_restored_optimizer(optimizer, 19)

    def test_direction_gate_includes_probe_and_action_mismatch(self):
        segments = {
            "action": {
                "early": {"f1": 0.6, "wrong_selection": 0.7},
                "late": {"f1": 0.61, "wrong_selection": 0.69},
            },
            "chain": {
                "early": {"reward": 0.4, "logic_alignment": 0.2, "action_mismatch": 0.2},
                "late": {"reward": 0.42, "logic_alignment": 0.21, "action_mismatch": 0.22},
            },
        }
        metrics = {"chain": {"action_mismatch": 0.2}}
        probes = {
            "0": {"action_f1": 0.5, "chain_reward": 0.3, "chain_logic_alignment": 0.1},
            "40": {"action_f1": 0.55, "chain_reward": 0.35, "chain_logic_alignment": 0.12},
        }
        decision = continuation_decision(
            segments, metrics, metrics, probes, {"nan_or_inf": False, "base_delta": 0.0}
        )
        self.assertEqual(decision["decision"], "HEALTHY_TO_FULL_EPOCH")

    def test_frozen_continuation_config(self):
        config = json.loads((ROOT / "config" / "pilot300_cont_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["resume"]["cumulative_start_step"], 20)
        self.assertEqual(config["resume"]["cumulative_end_step"], 40)
        self.assertEqual(config["full_epoch_cadence_contract"]["checkpoint_every"], 80)
        self.assertEqual((config["G"], config["lambda"], config["learning_rate"]), (4, 0.5, 1e-6))


if __name__ == "__main__":
    unittest.main()
