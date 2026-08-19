import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_monitor_adapter import (
    UserMonitorAdapter,
    build_user_step_event,
    user_run_manifest,
)


class RecordingWriter:
    def __init__(self):
        self.manifests = []
        self.steps = []
        self.rollouts = []
        self.traces = []

    def write_manifest(self, value):
        self.manifests.append(value)
        return True

    def write_step(self, value):
        self.steps.append(value)
        return True

    def write_rollout(self, value):
        self.rollouts.append(value)
        return True

    def write_trace(self, value):
        self.traces.append(value)
        return True


class UserMonitorAdapterTests(unittest.TestCase):
    def test_manifest_has_frozen_user_contract(self):
        manifest = user_run_manifest(dataset="train_3000", world_size=4, max_steps=20)
        self.assertEqual(manifest["run_kind"], "user_grpo")
        self.assertEqual(manifest["G"], 4)
        self.assertEqual(manifest["token_penalty"], {"strategy": "sqrt", "lambda": 0.5})
        self.assertEqual(manifest["reward"]["action"], "set_f1")

    def test_action_event_selects_existing_metrics_without_recomputation(self):
        rollout = {
            "task_reward_mean": 0.7,
            "masked_token_rate": 0.02,
            "f1_mean": 0.7,
            "wrong_selection_candidate_rate": 0.6,
            "beam32": "must-not-leak",
        }
        event = build_user_step_event(
            step=3,
            route="action",
            rollout_metrics=rollout,
            policy_metrics={"loss": 0.1, "ratio_mean": 1.0},
        )
        self.assertEqual(event["f1_mean"], 0.7)
        self.assertEqual(event["wrong_selection_candidate_rate"], 0.6)
        self.assertNotIn("beam32", event)
        self.assertEqual(rollout["beam32"], "must-not-leak")

    def test_chain_event_and_writer_are_passive(self):
        writer = RecordingWriter()
        adapter = UserMonitorAdapter(writer)
        self.assertTrue(adapter.write_manifest(dataset="pilot"))
        self.assertTrue(
            adapter.write_step(
                step=4,
                route="chain",
                rollout_metrics={
                    "total_reward_mean": 0.5,
                    "action_alignment_mean": 0.8,
                    "logic_alignment_mean": 0.2,
                    "date_mismatch_candidate_rate": 0.3,
                    "action_mismatch_candidate_rate": 0.1,
                    "violation_counts": {"date_mismatch": 3},
                },
                policy_metrics={"grad_norm": 0.9},
            )
        )
        self.assertEqual(writer.steps[0]["logic_alignment_mean"], 0.2)
        self.assertEqual(writer.steps[0]["date_mismatch_candidate_rate"], 0.3)
        self.assertEqual(writer.steps[0]["violation_counts"], {"date_mismatch": 3})
        self.assertTrue(adapter.write_rollout({"rollout_id": 1}))
        self.assertTrue(adapter.write_trace({"rollout_id": 1, "candidates": []}))

    def test_unknown_route_fails_closed(self):
        with self.assertRaises(ValueError):
            build_user_step_event(step=1, route="think", rollout_metrics={})


if __name__ == "__main__":
    unittest.main()
