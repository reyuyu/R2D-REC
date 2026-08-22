"""CPU-only MC_USER monitor compatibility tests."""
from __future__ import annotations

import json
import hashlib
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
                {"prompt_step": 1, "optimizer_step": 1, "route": "action", "skipped_update": False, "optimizer_step_performed": True, "active_unit_count": 3, "active_token_count": 9, "generation_peak_vram_mib": 17200, "candidates": candidates["action"]},
                {"prompt_step": 32, "optimizer_step": 31, "route": "chain", "skipped_update": True, "optimizer_step_performed": False, "active_unit_count": 0, "active_token_count": 0, "training_peak_vram_mib": 18400, "candidates": candidates["chain"]},
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
        self.assertAlmostEqual(summary["action"]["mean_f1"], 0.4)
        self.assertAlmostEqual(summary["chain"]["mean_predicted_event_count"], 3.0)
        self.assertEqual(summary["peak_vram_mib"], 18400)
        self.assertEqual(self.client.get(f"/api/rollouts?run_id={self.mc.name}").json(), [])
        self.assertEqual(self.client.get(f"/api/user-light-probe?run_id={self.mc.name}").json(), {"available": False})
        self.assertEqual(self.client.get(f"/api/recommendation-guard?run_id={self.mc.name}").json(), {"available": False})

    def test_future_rollout_is_read_without_recomputation(self):
        row = {
            "prompt_step": 2,
            "optimizer_step": 2,
            "route": "action",
            "candidate_index": 0,
            "completion": "A B",
            "credit_units": [{"unit_index": 0, "delta": -0.25, "credit_type": "negative", "char_start": 0, "char_end": 1, "generated_token_indices": [0]}],
            "overlap_metadata": [{"token_index": 0, "unit_indices": [0], "mixed_sign": False, "net_coefficient": -0.25}],
        }
        write_jsonl(self.mc / "rollouts.jsonl", [row])
        returned = self.client.get(f"/api/rollouts?run_id={self.mc.name}").json()
        self.assertEqual(returned[0]["step"], 2)
        self.assertEqual(returned[0]["credit_units"], row["credit_units"])
        self.assertEqual(returned[0]["overlap_metadata"], row["overlap_metadata"])

    def test_formal_train_contract_exposes_sample_context(self):
        sample_id = "b" * 64
        dataset = Path(self.temporary.name) / "formal_train.jsonl"
        write_jsonl(dataset, [{"sample_id": sample_id, "prompt": "fixed train prompt", "gold_sids": ["video A1 B2 C3"], "gold_events": []}])
        formal = self.user_runs / "mc_user_v1_formal" / "MC-USER-FORMAL-TEST"
        write_json(formal / "manifest.json", {
            "run_id": formal.name,
            "run_kind": "user_grpo",
            "algorithm": "mc_user_v1",
            "train_data": str(dataset),
            "train_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        })
        returned = self.client.get(f"/api/sample-context?run_id={formal.name}&sample_id={sample_id}")
        self.assertEqual(returned.status_code, 200)
        self.assertEqual(returned.json()["prompt"], "fixed train prompt")

    def test_future_probe_and_recommendation_guard_are_read_only(self):
        probe = {
            "status": "PASS",
            "checkpoints": [
                {"step": step, "summary": {"action": {"f1": 0.4 + step / 1000, "precision": 0.5, "recall": 0.4}, "chain": {"total_reward": 0.3, "action_alignment": 0.4, "logic_alignment": 0.2}, "overall_user_proxy": 0.7 + step / 1000}}
                for step in (0, 8, 16, 32)
            ],
        }
        guard = {"status": "PASS", "checkpoints": [{"step": 0, "hit_rate": 0.2, "history_copy_rate": 0.3, "unique_history_copy_rate": 0.25}, {"step": 32, "hit_rate": 0.21, "history_copy_rate": 0.32, "unique_history_copy_rate": 0.27}]}
        write_json(self.mc / "evaluations" / "user_light_probe" / "results.json", probe)
        write_json(self.mc / "evaluations" / "recommendation_guard" / "results.json", guard)
        returned_probe = self.client.get(f"/api/user-light-probe?run_id={self.mc.name}").json()
        returned_guard = self.client.get(f"/api/recommendation-guard?run_id={self.mc.name}").json()
        self.assertTrue(returned_probe["available"])
        self.assertEqual([row["step"] for row in returned_probe["checkpoints"]], [0, 8, 16, 32])
        self.assertTrue(returned_guard["available"])
        self.assertEqual(returned_guard["checkpoints"][-1]["history_copy_rate"], 0.32)

    def test_probe_queue_is_visible_before_sidecar_results(self):
        queue = {
            "status": "PENDING_TRAINING_CHECKPOINTS",
            "items": [
                {"step": 0, "status": "pending", "available_for_probe": True},
                {"step": 128, "status": "ready", "available_for_probe": True},
                {"step": 256, "status": "waiting", "available_for_probe": False},
            ],
        }
        write_json(self.mc / "evaluations" / "user_light_probe" / "probe_queue.json", queue)
        returned = self.client.get(f"/api/user-light-probe?run_id={self.mc.name}").json()
        self.assertTrue(returned["available"])
        self.assertEqual(returned["checkpoint_schedule"], [0, 128, 256])
        self.assertEqual(returned["items"][1]["status"], "ready")
        self.assertEqual(returned["checkpoints"], [])

    def test_live_probe_partial_result_preserves_waiting_schedule_and_samples(self):
        sample = {
            "sample_id": "fixed-action-1",
            "route": "action",
            "seed": 20260820,
            "prompt": "fixed input",
            "gold_sids": ["video A1 B2 C3"],
            "mean_reward": 0.5,
            "relative_to_beta_delta": 0.0,
            "candidates": [
                {
                    "candidate_id": index,
                    "completion": "video A1 B2 C3",
                    "completion_length": 4,
                    "reward": 0.5,
                    "f1": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "gold_sids": ["video A1 B2 C3"],
                    "pred_sids": ["video A1 B2 C3"],
                }
                for index in range(4)
            ],
        }
        probe = {
            "status": "WAITING_FOR_CHECKPOINTS",
            "mode": "FIXED_SAMPLES_INFERENCE_ONLY",
            "checkpoint_schedule": [0, 128, 256, 384, 512],
            "evaluated_steps": [0],
            "waiting_steps": [128, 256, 384, 512],
            "checkpoints": [
                {
                    "step": 0,
                    "summary": {
                        "action": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                        "chain": {"total_reward": 0.4, "action_alignment": 0.4, "logic_alignment": 0.4},
                        "overall_user_proxy": 0.9,
                    },
                    "samples": [sample],
                }
            ],
        }
        write_json(self.mc / "evaluations" / "user_light_probe" / "results.json", probe)
        returned = self.client.get(f"/api/user-light-probe?run_id={self.mc.name}").json()
        self.assertTrue(returned["available"])
        self.assertEqual(returned["status"], "WAITING_FOR_CHECKPOINTS")
        self.assertEqual(returned["evaluated_steps"], [0])
        self.assertEqual(returned["waiting_steps"], [128, 256, 384, 512])
        self.assertEqual(returned["checkpoints"][0]["samples"][0]["candidates"][3]["completion"], "video A1 B2 C3")

    def test_dashboard_uses_mc_dual_step_and_read_only_endpoints(self):
        html = self.client.get("/").text
        javascript = self.client.get("/static/user_dashboard.js").text
        self.assertIn("20260822-rollout-context", html)
        self.assertIn("Prompt Step", javascript)
        self.assertIn("Optimizer Step", javascript)
        self.assertIn("tokenTab.textContent = mc ? 'Marginal Credit' : 'Token Advantage'", javascript)
        self.assertIn("tokenTab.hidden = !user", javascript)
        for title in ("Marginal Credit Mass", "Negative Candidate Rate", "Training Signal", "Pipeline Health"):
            self.assertIn(title, javascript)
        self.assertIn("Action · SID Marginal Credit", javascript)
        self.assertIn("Chain · Event Marginal Credit", javascript)
        self.assertIn("该历史 run 未落盘 unit-level marginal trace", javascript)
        self.assertIn("generated_token_indices", javascript)
        self.assertIn("固定 3+3 inference-only sidecar 尚未写入 BETA", javascript)
        self.assertIn("当前训练样本 / on-policy / 参与训练", javascript)
        self.assertIn("checkpoint_schedule", javascript)
        self.assertIn("Waiting for adapter-only checkpoint", javascript)
        self.assertIn("mcProbeSampleDetail", javascript)
        self.assertIn("relative_to_beta_delta", javascript)
        self.assertIn("Action Mean F1", javascript)
        self.assertIn("state.manifest.K ?? 2", javascript)
        self.assertIn("userRolloutKey", javascript)
        self.assertIn("完整样本", javascript)
        self.assertIn("仅汇总", javascript)
        self.assertIn("summary-only", javascript)
        self.assertIn("trace-lane-compact", javascript)
        self.assertIn("completeCount", javascript)
        self.assertIn("平均 F1", javascript)
        self.assertIn("平均 Action / Logic", javascript)
        self.assertIn("没有可恢复的输入样本与 Ground Truth", javascript)
        self.assertIn("button.dataset.rolloutId!=null", html)
        self.assertIn("full_action_alignment", javascript)
        self.assertIn("尚未执行 Recommendation guard evaluation", javascript)
        self.assertIn("state.checkpoints", javascript)
        self.assertIn("`ckpt ${step}`", javascript)
        self.assertIn("/api/mc-user/summary", javascript)
        self.assertIn("/api/user-light-probe", javascript)
        self.assertIn("/api/recommendation-guard", javascript)


if __name__ == "__main__":
    unittest.main()
