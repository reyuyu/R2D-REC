"""CPU-only correctness tests for GRPO Monitoring Phase 1."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from monitor.generate_demo_run import generate
from monitor.server import create_app, read_jsonl
from monitor.writer import MonitorWriter, monitor_from_env


def parse_every_line(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)

    disabled = MonitorWriter(False, root, "disabled", rank=0)
    disabled.write_manifest({"seed": 1})
    disabled.write_step({"step": 1})
    assert not (root / "disabled").exists()
    with patch.dict(os.environ, {"GRPO_MONITOR_DIR": str(root)}, clear=True):
        assert monitor_from_env("env-disabled", 0).enabled is False
        assert not (root / "env-disabled").exists()
    print("[PASS] disabled monitor creates no files")

    rank0 = MonitorWriter(True, root, "writer-test", rank=0, trace_every=2)
    rank1 = MonitorWriter(True, root, "writer-test", rank=1, trace_every=2)
    assert rank0.write_manifest({"seed": 7, "unsafe_nan": float("nan")})
    assert rank0.write_step({"step": 1, "loss": 0.25})
    assert rank0.write_step({"step": 2, "loss": float("inf")})
    assert rank0.write_rollout({"rollout_id": 2, "step": 2, "route": "think"})
    assert rank0.write_rank({"rollout_id": 2, "route": "think", "beam_exec_wall_sec": 4.0})
    assert rank1.write_rank({"rollout_id": 2, "route": "think", "beam_exec_wall_sec": 5.0})
    assert rank0.write_trace({"rollout_id": 2, "step": 2, "route": "think", "candidates": []})
    assert parse_every_line(rank0.run_dir / "metrics.jsonl")[1]["loss"] is None
    assert json.loads((rank0.run_dir / "manifest.json").read_text())["unsafe_nan"] is None
    assert parse_every_line(rank0.run_dir / "ranks/rank0.jsonl")[0]["rank"] == 0
    assert parse_every_line(rank0.run_dir / "ranks/rank1.jsonl")[0]["rank"] == 1
    print("[PASS] append-only JSONL is strict JSON and rank files are isolated")

    # Simulate a writer interrupted in the middle of its next line. The server
    # must continue returning every preceding complete event.
    metrics_path = rank0.run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"step","step":3')
    assert [row["step"] for row in read_jsonl(metrics_path)] == [1, 2]
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('\n{"type":"step","step":4}\n')
    assert [row["step"] for row in read_jsonl(metrics_path)] == [1, 2, 4]
    print("[PASS] concurrent partial append cannot break JSONL reads")

    app = create_app(rank0.run_dir)
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/manifest").json()["seed"] == 7
    assert [row["step"] for row in client.get("/api/metrics?from_step=2").json()] == [2, 4]
    assert len(client.get("/api/rollouts?route=think&rollout_id=2").json()) == 1
    assert len(client.get("/api/ranks?rank=1").json()) == 1
    assert len(client.get("/api/traces?rollout_id=2").json()) == 1
    assert client.get("/").status_code == 200
    print("[PASS] FastAPI endpoints and query filters")

    isolated = MonitorWriter(True, root, "isolated-run", rank=0)
    assert isolated.write_manifest({"seed": 99, "max_steps": 120})
    assert isolated.write_step({"step": 17, "loss": 9.9})
    outputs = root / "_outputs"
    checkpoint = outputs / "isolated-run" / "checkpoint-17"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"private-training-state")
    multi_client = TestClient(create_app(runs_dir=root, outputs_dir=outputs))
    runs = multi_client.get("/api/runs").json()
    assert {item["run_id"] for item in runs} >= {"writer-test", "isolated-run"}
    assert multi_client.get("/api/manifest?run_id=writer-test").json()["seed"] == 7
    assert multi_client.get("/api/manifest?run_id=isolated-run").json()["seed"] == 99
    assert [row["step"] for row in multi_client.get("/api/metrics?run_id=isolated-run").json()] == [17]
    assert all(row["step"] != 17 for row in multi_client.get("/api/metrics?run_id=writer-test").json())
    assert multi_client.get("/api/metrics").status_code == 400
    assert multi_client.get("/api/metrics?run_id=..").status_code == 400
    assert multi_client.get("/api/metrics?run_id=missing").status_code == 404
    checkpoints = multi_client.get("/api/checkpoints?run_id=isolated-run").json()
    assert checkpoints[0]["checkpoint"] == "checkpoint-17"
    assert set(checkpoints[0]["files"]) == {"adapter_config.json", "adapter_model.safetensors"}
    download = multi_client.get("/api/checkpoints/checkpoint-17/download?run_id=isolated-run&file=adapter_config.json")
    assert download.status_code == 200 and download.json() == {"r": 32}
    assert multi_client.get("/api/checkpoints/checkpoint-17/download?run_id=isolated-run&file=optimizer.pt").status_code == 404
    assert multi_client.get("/api/checkpoints/../download?run_id=isolated-run&file=adapter_config.json").status_code == 404
    print("[PASS] experiment list and run-scoped APIs keep datasets isolated")

    demo_dir = Path(generate(str(root), "demo"))
    assert len(parse_every_line(demo_dir / "metrics.jsonl")) == 100
    assert len(parse_every_line(demo_dir / "rollouts.jsonl")) == 50
    assert all(len(parse_every_line(demo_dir / f"ranks/rank{rank}.jsonl")) == 50 for rank in range(4))
    traces = parse_every_line(demo_dir / "traces/traces.jsonl")
    assert len(traces) == 5
    assert len(traces[0]["candidates"][0]["beam_sids"]) == 32
    assert traces[0]["candidates"][0]["beam_sids"][0] == traces[0]["gold_sids"][0]
    nothink_trace = next(trace for trace in traces if trace["route"] == "no_think")
    assert len(nothink_trace["candidates"]) == 8
    assert all(candidate["reward"] is not None for candidate in nothink_trace["candidates"])
    html = client.get("/").text
    assert all(label in html for label in ("训练总览", "性能分析", "Rollout 检视", "选择实验"))
    assert all(label in html for label in ("查看 32 条 Beam SID", "最近 20", "Gold SID"))
    assert all(label in html for label in ("名词解释", "健康趋势", "奖励档位", "candidate-count"))
    print("[PASS] 100-step synthetic run, four rank streams, traces, and dashboard shell")

print("ALL MONITOR CPU TESTS PASSED")
