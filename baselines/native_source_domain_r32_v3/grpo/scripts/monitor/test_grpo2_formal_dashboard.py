"""CPU-only contract tests for the GRPO-2 formal dashboard adapter."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from monitor.server import GRPO2_CHECKPOINT_FILES, create_app


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_grpo2_formal_status_tracks_training_checkpoints_and_offline_results(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    outputs = tmp_path / "outputs"
    run_id = "formal-grpo2"
    monitor_run = runs / run_id
    output_run = outputs / run_id
    write_json(monitor_run / "manifest.json", {
        "run_id": run_id,
        "schema": "grpo2_think_continued_adapter_v1",
        "effective_max_steps": 300,
        "checkpoint_steps": [100, 150, 200, 250, 300],
        "adapter_inheritance": "CONTINUE_PARENT_ADAPTER",
        "adapter_initialization": "INHERITED_FROM_GRPO1",
        "fresh_lora": False,
        "fresh_optimizer": True,
        "base_full_model_sha256": "a" * 64,
        "grpo1_parent_checkpoint": 500,
        "grpo1_parent_adapter_sha256": "b" * 64,
        "git_commit": "c" * 40,
        "training_routes": ["think"],
    })
    monitor_run.mkdir(parents=True, exist_ok=True)
    (monitor_run / "metrics.jsonl").write_text(
        json.dumps({"step": 23, "loss": -0.1, "grad_norm": 1.2, "reward_mean": 0.5}) + "\n",
        encoding="utf-8",
    )
    write_json(output_run / "startup-audit-rank0.json", {"status": "PASS"})
    client = TestClient(create_app(runs_dir=runs, outputs_dir=outputs))

    training = client.get(f"/api/grpo2-formal/status?run_id={run_id}")
    assert training.status_code == 200
    payload = training.json()
    assert payload["phase"] == "training" and payload["current_step"] == 23
    assert [row["step"] for row in payload["checkpoints"]] == [100, 150, 200, 250, 300]
    assert not any(row["complete"] for row in payload["checkpoints"])
    assert str(tmp_path) not in json.dumps(payload)

    checkpoint = output_run / "checkpoint-100"
    checkpoint.mkdir()
    for name in GRPO2_CHECKPOINT_FILES:
        if name == "lineage.json":
            write_json(checkpoint / name, {"adapter_sha256": "d" * 64})
        elif name == "trainer_state.json":
            write_json(checkpoint / name, {"global_step": 100})
        else:
            (checkpoint / name).write_bytes(b"test")
    checkpoint_payload = client.get(f"/api/grpo2-formal/status?run_id={run_id}").json()
    assert checkpoint_payload["checkpoints"][0]["complete"] is True
    assert checkpoint_payload["checkpoints"][0]["adapter_sha256"] == "d" * 64

    write_json(output_run / "training_summary.json", {"status": "PASS", "global_step": 300})
    probe = output_run / "evidence" / "offline-monitor" / "OFFLINE-GRPO1-step500" / "probes.jsonl"
    probe.parent.mkdir(parents=True)
    probe.write_text('{"step":0}\n', encoding="utf-8")
    offline = client.get(f"/api/grpo2-formal/status?run_id={run_id}").json()
    assert offline["phase"] == "offline_probes"
    assert offline["offline_probes"][0]["complete"] is True

    write_json(output_run / "offline_checkpoint_comparison.json", {
        "status": "PASS",
        "models": [{
            "model": "GRPO1-step500",
            "metrics": {
                "think": {"mean_reward": 1.0, "success_at_k": {"value": 0.5}, "success_at_32": {"value": 0.75}},
                "nothink": {"mean_reward": 0.25, "success_at_k": {"value": 0.2}, "positive_candidate_rate": {"value": 0.3}},
            },
        }],
    })
    write_json(output_run / "FORMAL_GRPO2_REPORT.json", {"STATUS": "READY_FOR_GRPO2_CHECKPOINT_EVALUATION"})
    ready = client.get(f"/api/grpo2-formal/status?run_id={run_id}").json()
    assert ready["phase"] == "ready"
    assert ready["comparison"][0]["think_success_at_32"] == 0.75


def test_dashboard_loads_grpo2_adapter_assets(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_json(run / "manifest.json", {"schema": "grpo2_think_continued_adapter_v1"})
    client = TestClient(create_app(run_dir=run))
    html = client.get("/").text
    assert "grpo2_formal_dashboard.css" in html
    assert "grpo2_formal_dashboard.js" in html
    assert client.get("/static/grpo2_formal_dashboard.js").status_code == 200
