"""CPU-only MC_USER monitor compatibility tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from monitor.server import create_app


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MCUserMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.runs = base / "runs"
        self.outputs = base / "outputs"
        self.user_runs = base / "user-runs"
        self.runs.mkdir()
        self.outputs.mkdir()
        self.user_runs.mkdir()

        recommendation = self.runs / "recommendation"
        write_json(recommendation / "manifest.json", {"run_id": "recommendation", "max_steps": 10})
        write_jsonl(recommendation / "metrics.jsonl", [{"step": 7, "route": "think"}])

        legacy = self.runs / "legacy-user"
        write_json(legacy / "manifest.json", {"run_id": "legacy-user", "run_kind": "user_grpo", "max_steps": 20})
        write_jsonl(legacy / "metrics.jsonl", [{"step": 12, "route": "action"}])

        self.mc = self.user_runs / "mc_user_v1_pilot32" / "MC-USER-PILOT32-REAL"
        prompts = [
            {"prompt_step": index, "route": "action" if index % 2 else "chain", "sample_id": f"sample-{index}"}
            for index in range(1, 33)
        ]
        write_json(
            self.mc / "manifest.json",
            {
                "status": "READY_TO_EXECUTE",
                "run_id": self.mc.name,
                "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
                "train_sha256": "a" * 64,
                "config": {"K": 2, "prompt_count": 32},
                "prompts": prompts,
            },
        )
        candidates = {
            "action": [
                {"format_valid": True, "reward": 0.8, "projection_required": False, "negative_unit_count": 1, "positive_credit_mass": 0.5, "negative_credit_mass": 0.2, "overlap_token_count": 0, "predicted_sid_unit_count": 3},
                {"format_valid": False, "reward": 0.0, "projection_required": True, "negative_unit_count": 0, "positive_credit_mass": 0.0, "negative_credit_mass": 0.0, "overlap_token_count": 1, "predicted_sid_unit_count": 0},
            ],
            "chain": [
                {"format_valid": True, "reward": 0.6, "projection_required": False, "negative_unit_count": 0, "positive_credit_mass": 0.4, "negative_credit_mass": 0.0, "overlap_token_count": 0, "predicted_event_count": 2, "full_action_alignment": 0.7, "full_logic_alignment": 0.5},
                {"format_valid": True, "reward": 0.4, "projection_required": False, "negative_unit_count": 1, "positive_credit_mass": 0.2, "negative_credit_mass": 0.1, "overlap_token_count": 0, "predicted_event_count": 4, "full_action_alignment": 0.5, "full_logic_alignment": 0.3},
            ],
        }
        write_jsonl(
            self.mc / "metrics.jsonl",
            [
                {"prompt_step": 1, "optimizer_step": 1, "route": "action", "skipped_update": False, "optimizer_step_performed": True, "candidates": candidates["action"]},
                {"prompt_step": 32, "optimizer_step": 31, "route": "chain", "skipped_update": True, "optimizer_step_performed": False, "candidates": candidates["chain"]},
            ],
        )
        for step in (8, 16, 32):
            checkpoint = self.mc / "checkpoints" / f"prompt-step-{step:04d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        self.client = TestClient(
            create_app(runs_dir=self.runs, outputs_dir=self.outputs, user_runs_dir=self.user_runs)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_existing_pilot_recognition_and_dual_step(self):
        runs = {row["run_id"]: row for row in self.client.get("/api/runs").json()}
        self.assertEqual(runs[self.mc.name]["run_kind"], "user_grpo")
        self.assertEqual(runs[self.mc.name]["algorithm"], "mc_user_v1")
        self.assertEqual(runs[self.mc.name]["latest_step"], 32)
        manifest = self.client.get(f"/api/manifest?run_id={self.mc.name}").json()
        self.assertEqual((manifest["run_kind"], manifest["algorithm"], manifest["display_name"]), ("user_grpo", "mc_user_v1", "MC_USER_v1"))
        metrics = self.client.get(f"/api/metrics?run_id={self.mc.name}").json()
        self.assertEqual(metrics[-1]["step"], 32)
        self.assertEqual(metrics[-1]["optimizer_step"], 31)

    def test_checkpoint_discovery_and_legacy_regressions(self):
        checkpoints = self.client.get(f"/api/checkpoints?run_id={self.mc.name}").json()
        self.assertEqual([row["step"] for row in checkpoints], [8, 16, 32])
        self.assertEqual(self.client.get("/api/metrics?run_id=legacy-user").json()[-1]["step"], 12)
        self.assertEqual(self.client.get("/api/manifest?run_id=recommendation").json()["run_kind"], "recommendation_grpo")

    def test_mc_aggregate_and_placeholders(self):
        summary = self.client.get(f"/api/mc-user/summary?run_id={self.mc.name}").json()
        self.assertEqual((summary["prompt_step"], summary["optimizer_step"]), (32, 31))
        self.assertAlmostEqual(summary["valid_candidate_rate"], 0.75)
        self.assertAlmostEqual(summary["no_credit_prompt_rate"], 0.5)
        self.assertAlmostEqual(summary["projection_required_rate"], 0.25)
        self.assertAlmostEqual(summary["overlap_candidate_rate"], 0.25)
        self.assertAlmostEqual(summary["action"]["negative_candidate_rate"], 0.5)
        self.assertAlmostEqual(summary["chain"]["mean_predicted_event_count"], 3.0)
        self.assertEqual(self.client.get(f"/api/rollouts?run_id={self.mc.name}").json(), [])
        self.assertEqual(self.client.get(f"/api/user-light-probe?run_id={self.mc.name}").json(), {"available": False})
        self.assertEqual(self.client.get(f"/api/recommendation-guard?run_id={self.mc.name}").json(), {"available": False})

    def test_future_rollout_is_read_without_recomputation(self):
        row = {"prompt_step": 2, "optimizer_step": 2, "route": "action", "candidate_index": 0, "credit_units": [{"delta": -0.25}]}
        write_jsonl(self.mc / "rollouts.jsonl", [row])
        returned = self.client.get(f"/api/rollouts?run_id={self.mc.name}").json()
        self.assertEqual(returned[0]["step"], 2)
        self.assertEqual(returned[0]["credit_units"], [{"delta": -0.25}])

    def test_dashboard_uses_mc_dual_step_and_read_only_endpoints(self):
        html = self.client.get("/").text
        javascript = self.client.get("/static/user_dashboard.js").text
        self.assertIn("20260821-mc1a", html)
        self.assertIn("Prompt Step", javascript)
        self.assertIn("Optimizer Step", javascript)
        self.assertIn("/api/mc-user/summary", javascript)
        self.assertIn("/api/user-light-probe", javascript)
        self.assertIn("/api/recommendation-guard", javascript)


if __name__ == "__main__":
    unittest.main()
