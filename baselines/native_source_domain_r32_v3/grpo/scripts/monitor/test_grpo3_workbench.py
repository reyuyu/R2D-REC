"""CPU-only contracts for GRPO3 checkpoint download and cumulative-adapter merge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from monitor.server import create_app


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_grpo3_external_checkpoint_download_and_publish_contract(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    checkpoints = tmp_path / "checkpoints"
    run_id = "formal-grpo3"
    run = runs / run_id
    run.mkdir(parents=True)

    base = tmp_path / "full-sft"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"verified-full-sft-parent")
    base_sha = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
    parent_adapter_sha = "b" * 64
    config_sha = "c" * 64
    dataset_sha = "d" * 64
    commit = "e" * 40
    write_json(run / "manifest.json", {
        "run_id": run_id,
        "stage": "grpo3_user_from_grpo2_step250_formal_v1",
        "algorithm": "mc_user_hybrid_grpo_v1",
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
        "fresh_lora": False,
        "parent_experiment": "GRPO2_REC_THINK_CONTINUED_SINGLE_ADAPTER",
        "parent_adapter_sha256": parent_adapter_sha,
        "parent_checkpoint_step": 250,
        "parent_lineage": {
            "status": "PASS",
            "contains_grpo1_and_grpo2_effect": True,
            "adapter_sha256": parent_adapter_sha,
            "grpo2_step": 250,
        },
        "base_model": str(base),
        "checkpoint_root": str(checkpoints),
        "checkpoint_steps": [25, 50, 100, 200],
        "config_sha256": config_sha,
        "train_sha256": dataset_sha,
        "git_commit": commit,
        "prompt_count": 200,
    })
    (run / "metrics.jsonl").write_text(
        json.dumps({"prompt_step": 25, "optimizer_step": 25, "route": "action"}) + "\n",
        encoding="utf-8",
    )

    checkpoint = checkpoints / run_id / "checkpoints" / "prompt-step-0025"
    checkpoint.mkdir(parents=True)
    adapter_bytes = b"cumulative-grpo1-grpo2-grpo3-adapter"
    (checkpoint / "adapter_model.safetensors").write_bytes(adapter_bytes)
    (checkpoint / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    write_json(checkpoint / "lineage.json", {
        "schema": "grpo3_user_continued_adapter_lineage_v1",
        "adapter_only": True,
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
        "contains_grpo1_grpo2_and_grpo3_effect": True,
        "fresh_lora": False,
        "parent_adapter_sha256": parent_adapter_sha,
        "adapter_weight_parent": "GRPO2 checkpoint-250",
        "grpo3_prompt_step": 25,
        "grpo3_optimizer_step": 25,
        "dataset_sha256": dataset_sha,
        "config_sha256": config_sha,
        "code_commit": commit,
        "base_model": str(base),
        "adapter_sha256": adapter_sha,
    })

    token_file = tmp_path / "modelscope.token"
    token_file.write_text("test-token-must-not-leak", encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o600)
    app = create_app(
        runs_dir=runs,
        checkpoint_outputs_dirs=[checkpoints],
        publish_root=tmp_path / "publish",
        publish_python=Path(os.sys.executable),
        publish_base_models={base_sha: base},
        publish_token_file=token_file,
    )
    client = TestClient(app)

    listed = client.get("/api/runs").json()
    assert listed[0]["display_name"] == "GRPO-3 User GRPO"
    catalog = client.get(f"/api/checkpoints?run_id={run_id}").json()
    assert catalog == [{
        "checkpoint": "prompt-step-0025",
        "step": 25,
        "files": {
            "adapter_config.json": len('{"r":32}'.encode()),
            "adapter_model.safetensors": len(adapter_bytes),
        },
    }]
    downloaded = client.get(
        f"/api/checkpoints/prompt-step-0025/download?run_id={run_id}&file=adapter_model.safetensors"
    )
    assert downloaded.status_code == 200 and downloaded.content == adapter_bytes
    assert "attachment" in downloaded.headers["content-disposition"]
    assert downloaded.headers["content-type"] == "application/octet-stream"

    capability = client.get(f"/api/model-publish/capabilities?run_id={run_id}").json()
    assert capability["enabled"] is True
    assert capability["parent_sha256"] == base_sha
    assert "test-token" not in json.dumps(capability)

    request = {
        "checkpoint": "prompt-step-0025",
        "model_id": "owner/grpo3-step25",
        "visibility": "private",
        "confirmation": f"PUBLISH {run_id} prompt-step-0025 owner/grpo3-step25",
    }
    with patch("monitor.server.subprocess.Popen", return_value=Mock(pid=os.getpid())) as launch:
        response = client.post(f"/api/model-publish/jobs?run_id={run_id}", json=request)
    assert response.status_code == 200
    command = launch.call_args.args[0]
    assert str(base) in command and str(checkpoint) in command
    assert "test-token-must-not-leak" not in " ".join(command)

    html = client.get("/").text
    assert "下载 Adapter" in html
    assert "downloadSelectedCheckpoint" in html
    assert "showDirectoryPicker" in html
    user_dashboard = client.get("/static/user_dashboard.js").text
    assert "'/api/model-publish/capabilities'" in user_dashboard
    assert "'/api/model-publish/jobs'" in user_dashboard
    assert "recommendationGuard, modelPublish, publishJobs" in user_dashboard

    lineage = json.loads((checkpoint / "lineage.json").read_text(encoding="utf-8"))
    lineage["contains_grpo1_grpo2_and_grpo3_effect"] = False
    write_json(checkpoint / "lineage.json", lineage)
    invalid = client.post(f"/api/model-publish/jobs?run_id={run_id}", json=request)
    assert invalid.status_code == 409
