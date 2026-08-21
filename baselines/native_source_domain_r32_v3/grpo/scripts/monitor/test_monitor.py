"""CPU-only correctness tests for GRPO Monitoring Phase 1."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from checkpoint_eval import build_cohort, load_validation_pool, route_prompt, wilson_interval
from monitor.generate_demo_run import generate
from monitor.generate_user_demo_run import generate as generate_user
from monitor.server import checkpoint_eval_launch_command, create_app, read_jsonl
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
    assert rank0.write_probe({"step": 0, "group_id": "probe-a", "think": {"reward_mean": 1.0}})
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

    eval_validation = root / "eval-validation.jsonl"
    eval_train = root / "eval-train.jsonl"
    eval_leakage = root / "eval-leakage.json"
    validation_rows = []
    for index, domain in enumerate(("video", "prod", "ad", "living")):
        group_id = hashlib.sha256(f"validation-{index}".encode()).hexdigest()
        sid = f"<|{domain}_begin|><s_a_{index + 1}><s_b_2><s_c_3>"
        validation_rows.append({
            "instruction": f"旧提示\n历史 {index}/{domain}/think", "input": "",
            "source_segment": "recommendation_cot",
            "aux_metadata_json": json.dumps({
                "recommendation_group_id": group_id,
                "recommendation_all_gold_sids": [sid],
            }, ensure_ascii=False),
        })
    eval_validation.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in validation_rows) + "\n", encoding="utf-8")
    eval_train.write_text(json.dumps({"recommendation_group_id": hashlib.sha256(b"train").hexdigest()}) + "\n", encoding="utf-8")
    eval_leakage.write_text(json.dumps({
        "validation_safe": True,
        "dev": {
            "recommendation_group_id_overlap": 0,
            "recommendation_history_domain_overlap": 0,
            "exact_full_row_overlap": 0,
            "canonical_prompt_overlap": 0,
        },
    }), encoding="utf-8")
    eval_pool = load_validation_pool(eval_validation, eval_leakage, eval_train)
    assert len(eval_pool) == 4 and len(build_cohort(eval_pool, 4, 20260822)) == 4
    assert route_prompt(eval_pool[0]["base_prompt"], eval_pool[0]["domain"], "think").endswith("/think")
    assert wilson_interval(3, 4)[0] < 0.75 < wilson_interval(3, 4)[1]
    evaluator_source = (Path(__file__).parent.parent / "checkpoint_eval.py").read_text(encoding="utf-8")
    assert "optimizer.step" not in evaluator_source and "scheduler.step" not in evaluator_source
    launch_command = checkpoint_eval_launch_command(Path("/tmp/checkpoint_eval.py"), 29617)
    assert launch_command[1:4] == ["-m", "torch.distributed.run", "--nproc_per_node=4"]
    assert launch_command[-2:] == ["--master_port=29617", "/tmp/checkpoint_eval.py"]
    print("[PASS] frozen disjoint cohort, route prompts, Wilson interval, and inference-only evaluator")

    app = create_app(rank0.run_dir)
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/manifest").json()["seed"] == 7
    assert [row["step"] for row in client.get("/api/metrics?from_step=2").json()] == [2, 4]
    assert len(client.get("/api/rollouts?route=think&rollout_id=2").json()) == 1
    assert len(client.get("/api/ranks?rank=1").json()) == 1
    assert len(client.get("/api/traces?rollout_id=2").json()) == 1
    assert client.get("/api/probes?group_id=probe-a").json()[0]["step"] == 0
    assert client.get("/api/capabilities").json()["dsr"] is False
    assert client.get("/api/dsr/metrics").json() == []
    assert client.get("/api/dsr/steps").json() == []
    assert client.get("/api/dsr/traces").json() == []
    assert client.get("/").status_code == 200
    print("[PASS] FastAPI endpoints and query filters")

    probe_source = root / "probe_light_v1.jsonl"
    source_row = {
        "sample_id": "probe-a",
        "prompt": "history input /no_think",
        "history_sids": ["sid-a"],
        "gold_events": [{"date": "2026-01-01", "action": "watch", "logic": "interest"}],
    }
    probe_source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    source_sha = hashlib.sha256(probe_source.read_bytes()).hexdigest()
    manifest_path = rank0.run_dir / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest["fixed_probe"] = {"dataset": str(probe_source), "sha256": source_sha, "enabled": True}
    train_source = root / "train_3000.jsonl"
    train_sample_id = "a" * 64
    train_row = {
        "sample_id": train_sample_id,
        "prompt": "training history input /no_think",
        "gold_sids": ["sid-gold"],
        "history_sids": ["sid-history"],
        "private_field": "must-not-leak",
    }
    train_source.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    source_manifest["dataset"] = str(train_source)
    source_manifest["dataset_sha"] = {
        train_source.name: hashlib.sha256(train_source.read_bytes()).hexdigest()
    }
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    rank0.write_probe({"step": 10, "group_id": "probe-a", "think": {"reward_mean": 1.0}})
    rank0.write_trace({"rollout_id": 4, "step": 3, "route": "action", "group_id": train_sample_id, "candidates": []})
    enriched_rows = client.get("/api/probes?group_id=probe-a").json()
    enriched = enriched_rows[0]
    assert enriched["prompt"] == source_row["prompt"]
    assert enriched["history_sids"] == source_row["history_sids"]
    assert enriched["gold_events"] == source_row["gold_events"]
    assert sum(row.get("prompt") == source_row["prompt"] for row in enriched_rows) == 1
    enriched_trace = client.get("/api/traces?rollout_id=4").json()[0]
    assert enriched_trace["prompt"] == train_row["prompt"]
    assert enriched_trace["gold_sids"] == train_row["gold_sids"]
    assert "private_field" not in enriched_trace
    context_response = client.get(f"/api/sample-context?sample_id={train_sample_id}")
    assert context_response.status_code == 200
    assert context_response.json()["prompt"] == train_row["prompt"]
    assert "private_field" not in context_response.json()
    assert client.get("/api/sample-context?sample_id=../private").status_code == 404
    source_manifest["fixed_probe"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    assert "prompt" not in client.get("/api/probes?group_id=probe-a").json()[0]
    print("[PASS] Probe and rollout source enrichment is SHA-gated and field-limited")

    isolated = MonitorWriter(True, root, "isolated-run", rank=0)
    outputs = root / "_outputs"
    user_outputs = root / "_user_runs"
    assert isolated.write_manifest({"seed": 99, "max_steps": 120, "fixed_probe": None, "output_dir": str(user_outputs / "isolated-run")})
    assert isolated.write_step({"step": 17, "loss": 9.9})
    checkpoint = outputs / "isolated-run" / "checkpoint-17"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"private-training-state")
    multi_client = TestClient(create_app(runs_dir=root, outputs_dir=outputs, user_runs_dir=user_outputs))
    runs = multi_client.get("/api/runs").json()
    assert {item["run_id"] for item in runs} >= {"writer-test", "isolated-run"}
    assert multi_client.get("/api/manifest?run_id=writer-test").json()["seed"] == 7
    assert multi_client.get("/api/manifest?run_id=isolated-run").json()["seed"] == 99
    assert [row["step"] for row in multi_client.get("/api/metrics?run_id=isolated-run").json()] == [17]
    assert all(row["step"] != 17 for row in multi_client.get("/api/metrics?run_id=writer-test").json())
    assert len(multi_client.get("/api/probes?run_id=writer-test").json()) == 2
    assert multi_client.get("/api/probes?run_id=isolated-run").json() == []
    assert multi_client.get("/api/capabilities?run_id=isolated-run").json()["probes"] is False
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
    rejected_delete = multi_client.delete("/api/checkpoints/checkpoint-17?run_id=isolated-run&confirm=wrong")
    assert rejected_delete.status_code == 400 and checkpoint.is_dir()
    deleted = multi_client.delete(
        "/api/checkpoints/checkpoint-17?run_id=isolated-run&confirm=checkpoint-17"
    )
    assert deleted.status_code == 200
    assert deleted.json()["released_bytes"] >= len(b"safe-adapter")
    assert not checkpoint.exists()
    assert multi_client.get("/api/checkpoints?run_id=isolated-run").json() == []
    assert multi_client.delete(
        "/api/checkpoints/checkpoint-17?run_id=isolated-run&confirm=checkpoint-17"
    ).status_code == 404

    formal_checkpoint = user_outputs / "isolated-run" / "checkpoint-step80"
    formal_checkpoint.mkdir(parents=True)
    (formal_checkpoint / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    (formal_checkpoint / "adapter_model.safetensors").write_bytes(b"formal-safe-adapter")
    formal_checkpoints = multi_client.get("/api/checkpoints?run_id=isolated-run").json()
    assert formal_checkpoints == [{
        "checkpoint": "checkpoint-step80",
        "step": 80,
        "files": {
            "adapter_config.json": len(b'{"r":32}'),
            "adapter_model.safetensors": len(b"formal-safe-adapter"),
        },
    }]
    formal_download = multi_client.get(
        "/api/checkpoints/checkpoint-step80/download?run_id=isolated-run&file=adapter_model.safetensors"
    )
    assert formal_download.status_code == 200 and formal_download.content == b"formal-safe-adapter"

    final_checkpoint = user_outputs / "isolated-run" / "full-epoch-final"
    final_checkpoint.mkdir(parents=True)
    (final_checkpoint / "metadata.json").write_text('{"global_optimizer_step":378}', encoding="utf-8")
    (final_checkpoint / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    (final_checkpoint / "adapter_model.safetensors").write_bytes(b"final-safe-adapter")
    final_checkpoints = multi_client.get("/api/checkpoints?run_id=isolated-run").json()
    assert [item["checkpoint"] for item in final_checkpoints] == ["checkpoint-step80", "full-epoch-final"]
    assert final_checkpoints[-1]["step"] == 378
    final_download = multi_client.get(
        "/api/checkpoints/full-epoch-final/download?run_id=isolated-run&file=adapter_model.safetensors"
    )
    assert final_download.status_code == 200 and final_download.content == b"final-safe-adapter"
    print("[PASS] experiment list and run-scoped APIs keep datasets isolated")

    demo_dir = Path(generate(str(root), "demo"))
    assert len(parse_every_line(demo_dir / "metrics.jsonl")) == 100
    assert len(parse_every_line(demo_dir / "rollouts.jsonl")) == 50
    assert all(len(parse_every_line(demo_dir / f"ranks/rank{rank}.jsonl")) == 50 for rank in range(4))
    traces = parse_every_line(demo_dir / "traces/traces.jsonl")
    probes = parse_every_line(demo_dir / "probes.jsonl")
    assert len(probes) == 12
    assert all(len(row["think"]["candidates"]) == 4 for row in probes)
    assert all(len(row["nothink"]["candidates"]) == 8 for row in probes)
    assert len(traces) == 5
    assert len(traces[0]["candidates"][0]["beam_sids"]) == 32
    assert traces[0]["candidates"][0]["beam_sids"][0] == traces[0]["gold_sids"][0]
    nothink_trace = next(trace for trace in traces if trace["route"] == "no_think")
    assert len(nothink_trace["candidates"]) == 8
    assert all(candidate["reward"] is not None for candidate in nothink_trace["candidates"])
    html = client.get("/").text
    assert all(label in html for label in ("训练总览", "性能分析", "采样检视", "固定探针", "选择实验"))
    assert all(label in html for label in ("查看 32 条 Beam SID", "最近 20", "Gold SID"))
    assert all(label in html for label in ("名词解释", "健康趋势", "奖励档位", "candidate-count"))
    assert all(label in html for label in (
        "DSR 诊断", "信号救援", "错误 A 轮换", "思考辅助原始损失",
        "Raw N", "Grounding Coverage"
    ))
    assert all(label in html for label in (
        "实验名称", "选择 D:\\model\\GRPO", "下载两个文件", "刷新列表", "删除检查点"
    ))
    assert all(label in html for label in (
        "renderRolloutCredit", "ensureExplorerAdvantages", "Sequence Advantage",
        "Token Credit", "查看完整优势详情", "优势载入中", "复算"
    ))
    assert all(label in html for label in (
        "function rolloutKey(row)", "function selectedExplorerRollout()",
        "function selectedExplorerTrace()", "data-rollout-key", "data-trace-key",
        "from_step=${encodeURIComponent(trace.step)}", "to_step=${encodeURIComponent(trace.step)}",
        "Number(item.step)===Number(trace.step)", "rollouts:['step','rollout_id','route']",
        "traces:['step','rollout_id','route','group_id']",
    ))
    assert all(label in html for label in (
        "历史实验奖励对比", "历史实验 1", "历史实验 2", "清空对比",
        "updateRewardComparison", "historicalMetricCache", "当前粗实线，历史细虚线"
    ))
    assert all(label in html for label in (
        "放大查看", "原始曲线（保留尖峰）", "attachChartControls",
        "chart-smoothing-control", "下方均为原始采样值", "最近 100", "最近 200"
    ))
    assert all(label in html for label in (
        "Checkpoint 验证", "冻结验证集上的 Checkpoint 横向评测", "启动 4 卡验证",
        "思考路线 Hit@32", "直答路线 Hit@32", "95% CI", "function renderCheckpointEval()",
        "/api/checkpoint-eval/catalog", "/api/checkpoint-eval/jobs",
    ))
    assert 'id="dsrTab" data-view="dsr" hidden' in html
    print("[PASS] 100-step synthetic run, four rank streams, traces, and dashboard shell")

    dsr_demo_dir = Path(generate(str(root), "demo-dsr", dsr=True))
    assert json.loads((dsr_demo_dir / "manifest.json").read_text(encoding="utf-8"))["max_steps"] == 400
    assert len(parse_every_line(dsr_demo_dir / "metrics.jsonl")) == 400
    assert len(parse_every_line(dsr_demo_dir / "dsr_steps.jsonl")) == 400
    assert len(parse_every_line(dsr_demo_dir / "dsr_metrics.jsonl")) == 200
    dsr_probes = parse_every_line(dsr_demo_dir / "probes.jsonl")
    assert {row["step"] for row in dsr_probes} == {0, 200, 400}
    assert all("dsr" in row for row in dsr_probes)
    assert all(row["reason"] == "final" for row in dsr_probes if row["step"] == 400)

    with (dsr_demo_dir / "dsr_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type":"dsr_rollout","step":401')
    dsr_client = TestClient(create_app(runs_dir=root))
    assert dsr_client.get("/api/capabilities?run_id=demo-dsr").json()["dsr"] is True
    assert dsr_client.get("/api/capabilities?run_id=demo").json()["dsr"] is False
    assert len(dsr_client.get("/api/dsr/metrics?run_id=demo-dsr&route=think").json()) > 0
    assert dsr_client.get("/api/dsr/metrics?run_id=demo").json() == []
    filtered_steps = dsr_client.get(
        "/api/dsr/steps?run_id=demo-dsr&from_step=200&to_step=205&route=think"
    ).json()
    assert filtered_steps and all(200 <= row["step"] <= 205 and row["route"] == "think" for row in filtered_steps)
    assert dsr_client.get("/api/dsr/metrics?run_id=..").status_code == 400
    assert dsr_client.get("/api/dsr/metrics?run_id=missing").status_code == 404
    print("[PASS] DSR demo, optional APIs, run isolation, filters, and incomplete-tail tolerance")

    user_demo_dir = Path(generate_user(str(root), "demo-user-grpo", steps=40))
    user_manifest = json.loads((user_demo_dir / "manifest.json").read_text(encoding="utf-8"))
    user_metrics = parse_every_line(user_demo_dir / "metrics.jsonl")
    user_rollouts = parse_every_line(user_demo_dir / "rollouts.jsonl")
    user_traces = parse_every_line(user_demo_dir / "traces/traces.jsonl")
    user_probes = parse_every_line(user_demo_dir / "probes.jsonl")
    assert user_manifest["run_kind"] == "user_grpo" and user_manifest["demo"] is True
    assert user_manifest["G"] == 4 and user_manifest["token_penalty"]["lambda"] == 0.5
    assert len(user_metrics) == 40 and {row["route"] for row in user_metrics} == {"action", "chain"}
    assert len(user_rollouts) == 8 and {row["route"] for row in user_rollouts} == {"action", "chain"}
    assert len(user_traces) == 8 and all(len(row["candidates"]) == 4 for row in user_traces)
    assert any(candidate.get("masked_spans") for row in user_traces for candidate in row["candidates"])
    assert len(user_probes) == 60 and {row["step"] for row in user_probes} == {0, 20, 40}
    assert {row["route"] for row in user_probes} == {"action", "chain"}
    assert any(candidate.get("match_spans") for row in user_probes for candidate in row["candidates"])

    with (user_demo_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type":"step","step":41')
    user_client = TestClient(create_app(runs_dir=root))
    all_runs = user_client.get("/api/runs").json()
    by_id = {run["run_id"]: run for run in all_runs}
    assert by_id["demo-user-grpo"]["run_kind"] == "user_grpo"
    assert by_id["demo-user-grpo"]["demo"] is True
    assert by_id["demo"]["run_kind"] == "recommendation_grpo"
    assert {run["run_id"] for run in user_client.get("/api/runs?run_kind=user_grpo").json()} == {"demo-user-grpo"}
    assert user_client.get("/api/runs?run_kind=unknown").status_code == 400
    assert user_client.get("/api/manifest?run_id=demo").json()["run_kind"] == "recommendation_grpo"
    assert user_client.get("/api/capabilities?run_id=demo-user-grpo").json()["user_grpo"] is True
    assert user_client.get("/api/capabilities?run_id=demo-user-grpo").json()["probes"] is True
    assert user_client.get("/api/capabilities?run_id=demo").json()["user_grpo"] is False
    assert len(user_client.get("/api/metrics?run_id=demo-user-grpo").json()) == 40
    assert user_client.get("/static/user_dashboard.js").status_code == 200
    assert user_client.get("/static/user_dashboard.css").status_code == 200
    user_js = user_client.get("/static/user_dashboard.js").text
    assert all(label in user_js for label in (
        "懂推荐 GRPO", "懂用户 GRPO", "Action Set-F1", "Chain Alignment",
        "Token Advantage", "Rollout 样本", "wrong_selection_candidate_rate",
        "stageRunKind", "切换实验中", "Prob / 固定探针", "match-mark",
        "userProbeActionChart",
        "输入样本", "Ground Truth ·", "selected.prompt", "selected.gold_events",
        "renderUserSampleContext(trace, trace.route)",
        "latestTrace", "· 有样本",
        "/api/sample-context", "candidate trace 未落盘", "代表输入 1/",
        "captureMonitorScrollState", "restoreMonitorScrollState", "document.scrollingElement",
        "userRefreshInFlight", "populateRewardCompareOptions", "attachChartControls",
    ))
    print("[PASS] User run-kind routing, fixed probes, green matches, and six dashboard views")

print("ALL MONITOR CPU TESTS PASSED")
