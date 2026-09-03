#!/usr/bin/env python3
"""Read-only local dashboard for sequential STABLE553-A/B runs."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


LABELS = ("STABLE553-A", "STABLE553-B")
TRAINING_MARKER = "***** Running training *****"
DOWNLOAD_FILES = ("adapter_model.safetensors", "adapter_config.json")
STEP_RE = re.compile(r"(\d+)/1106")
EPOCH2_RUN_RE = re.compile(r"^EPOCH2-S([0-9]{1,10})-([0-9]{8}-[0-9]{6})$")
MAX_SEED = 2**31 - 1
TRAIN_ACTION_LOCK = threading.Lock()
ERROR_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|non-finite|"
    r"(?:^|[^A-Za-z])(?:nan|inf)(?:[^A-Za-z]|$)",
    re.IGNORECASE,
)


def scalar_rows(log_text: str, *, start_step: int = 0) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    inferred_step = start_step
    for raw in log_text.replace("\r", "\n").splitlines():
        text = raw.strip()
        if not (text.startswith("{") and text.endswith("}") and "'loss'" in text):
            continue
        try:
            row = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict):
            inferred_step += 5
            step = row.get("step", inferred_step)
            if not isinstance(step, int):
                continue
            values = {key: row.get(key) for key in ("loss", "grad_norm", "learning_rate", "epoch")}
            rows[step] = {
                "step": step,
                **{
                    key: float(value) if isinstance(value, str) and value else value
                    for key, value in values.items()
                },
            }
    return [rows[step] for step in sorted(rows)]


def split_training_logs(log_text: str) -> dict[str, str]:
    """Map sequential Trainer log sections to their fixed A/B run labels."""
    sections = log_text.split(TRAINING_MARKER)[1:]
    return {
        label: TRAINING_MARKER + sections[index] if index < len(sections) else ""
        for index, label in enumerate(LABELS)
    }


def nvidia_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == 5:
            rows.append(
                dict(zip(("index", "memory_used", "memory_total", "utilization", "temperature"), map(int, values)))
            )
    return rows


def run_status(root: Path, label: str, log_text: str) -> dict[str, Any]:
    evidence_dir = root / "runs" / label / "evidence"
    evidence = []
    for path in sorted(evidence_dir.glob("runtime_rank*.json")):
        try:
            evidence.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    checkpoint = root / "runs" / label / "output" / "checkpoint-553"
    required = (
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
        "trainer_state.json", "training_args.bin", "rng_state_0.pth", "rng_state_1.pth",
        "rng_state_2.pth", "rng_state_3.pth",
    )
    complete = checkpoint.is_dir() and all((checkpoint / name).is_file() for name in required)
    statuses = [row.get("status") for row in evidence]
    if complete and len(evidence) == 4 and all(status == "PASS" for status in statuses):
        status = "PASS"
        step = 553
    elif evidence and any(status == "RUNNING" for status in statuses):
        status = "RUNNING"
        step = max((int(value) for value in STEP_RE.findall(log_text)), default=0)
        step = min(step, 553)
    else:
        status = "WAITING"
        step = 0
    return {
        "label": label,
        "status": status,
        "step": step,
        "target_step": 553,
        "rank_evidence_count": len(evidence),
        "checkpoint_complete": complete,
        "download_ready": all((checkpoint / name).is_file() for name in DOWNLOAD_FILES),
    }


def checkpoint_download_path(root: Path, label: str, filename: str) -> Path | None:
    if label not in LABELS or filename not in DOWNLOAD_FILES:
        return None
    path = root / "runs" / label / "output" / "checkpoint-553" / filename
    return path if path.is_file() else None


def epoch2_checkpoint_download_path(root: Path, run_id: str, filename: str) -> Path | None:
    if EPOCH2_RUN_RE.fullmatch(run_id) is None or filename not in DOWNLOAD_FILES:
        return None
    path = root / "epoch2_runs" / run_id / "output" / "checkpoint-1106" / filename
    return path if path.is_file() else None


def process_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def manual_scores(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "manual_scores.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_manual_score(root: Path, run_id: str, score: float | None, notes: str) -> dict[str, Any]:
    if (
        not isinstance(run_id, str)
        or EPOCH2_RUN_RE.fullmatch(run_id) is None
        or not (root / "epoch2_runs" / run_id).is_dir()
    ):
        raise ValueError("unknown epoch-2 run")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
        raise ValueError("score must be numeric or null")
    if not isinstance(notes, str) or len(notes) > 500:
        raise ValueError("notes must be a string of at most 500 characters")
    values = manual_scores(root)
    record = {"score": float(score) if score is not None else None, "notes": notes.strip()}
    values[run_id] = record
    path = root / "manual_scores.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def epoch2_run_status(root: Path, run_dir: Path, scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_id = run_dir.name
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"run_id": run_id, "status": "INVALID", "step": 553, "target_step": 1106}
    log_path = run_dir / "train.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    pid_path = run_dir / "launcher.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = None
    evidence = []
    for path in sorted((run_dir / "evidence").glob("runtime_rank*.json")):
        try:
            evidence.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    checkpoint = run_dir / "output/checkpoint-1106"
    complete = checkpoint.is_dir() and all((checkpoint / name).is_file() for name in DOWNLOAD_FILES)
    evidence_pass = len(evidence) == 4 and all(row.get("status") == "PASS" for row in evidence)
    alive = process_alive(pid)
    steps = [int(value) for value in STEP_RE.findall(log_text)]
    scalar_series = scalar_rows(log_text, start_step=553)
    step = max([553, *steps, *(int(row["step"]) for row in scalar_series)], default=553)
    step = min(step, 1106)
    errors = [line.strip() for line in log_text.replace("\r", "\n").splitlines() if ERROR_RE.search(line)][-5:]
    if complete and evidence_pass:
        status = "PASS"
        step = 1106
    elif alive:
        status = "RUNNING"
    elif pid is None:
        status = "READY"
    else:
        status = "STOPPED"
    return {
        "run_id": run_id,
        "seed": int(manifest["seed"]),
        "source_label": manifest.get("source_label"),
        "status": status,
        "step": step,
        "target_step": 1106,
        "launcher_pid": pid,
        "checkpoint_complete": complete,
        "download_ready": complete,
        "evidence_count": len(evidence),
        "latest_scalar": scalar_series[-1] if scalar_series else None,
        "series": scalar_series,
        "errors": errors,
        "manual_score": scores.get(run_id, {"score": None, "notes": ""}),
    }


def epoch2_runs(root: Path) -> list[dict[str, Any]]:
    runs_root = root / "epoch2_runs"
    scores = manual_scores(root)
    if not runs_root.is_dir():
        return []
    rows = [
        epoch2_run_status(root, path, scores)
        for path in sorted(runs_root.iterdir(), reverse=True)
        if path.is_dir() and EPOCH2_RUN_RE.fullmatch(path.name)
    ]
    return rows


def gpu_launch_ready() -> tuple[bool, str]:
    gpus = nvidia_snapshot()
    if len(gpus) != 4 or any(row["memory_used"] >= 1024 for row in gpus):
        return False, "四张 GPU 必须全部低于 1024 MiB"
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    try:
        processes = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, "无法读取 GPU compute process"
    if processes:
        return False, "GPU 上已有 compute process"
    return True, "READY"


def start_epoch2_run(root: Path, seed: int) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
    existing = epoch2_runs(root)
    if any(row["status"] == "RUNNING" for row in existing):
        raise RuntimeError("已有 epoch-2 续训正在运行")
    ready, reason = gpu_launch_ready()
    if not ready:
        raise RuntimeError(reason)
    run_id = f"EPOCH2-S{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    support = Path(__file__).resolve().parent
    prepare = support / "prepare_stable_epoch2.py"
    launcher = support / "launch_stable_epoch2.sh"
    subprocess.run(
        [
            "/data/venvs/llamafactory-01398eb-liger081/bin/python",
            str(prepare),
            "prepare",
            "--root",
            str(root),
            "--run-id",
            run_id,
            "--seed",
            str(seed),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    run_dir = root / "epoch2_runs" / run_id
    log_handle = (run_dir / "train.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            ["bash", str(launcher), str(root), run_id],
            cwd=support,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    (run_dir / "launcher.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return {"status": "STARTED", "run_id": run_id, "seed": seed, "pid": process.pid}


def comparison_status(root: Path) -> dict[str, Any]:
    """Return the small, public subset needed by the live dashboard."""
    path = root / "public" / "stable553_result.json"
    if not path.is_file():
        return {"status": "WAITING"}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        pairwise = result["pairwise_repeatability"]
        historical = result["historical553_comparison"]
        return {
            "status": "READY",
            "verdict": result["verdict"],
            "first_epoch_repeatability": result["first_epoch_repeatability"],
            "pairwise": {
                key: pairwise[key]
                for key in (
                    "adapter_file_sha_exact",
                    "adapter_canonical_sha_exact",
                    "optimizer_exact",
                    "scheduler_exact",
                    "rng_exact",
                    "raw_lora",
                    "effective_ba",
                )
            },
            "historical": {
                "raw_lora": historical["raw_lora"],
                "effective_ba": historical["effective_ba"],
                "layer_summary": historical["layer_summary"],
                "projections": historical["projections"],
            },
            "external_evaluation": result.get("external_evaluation", {}).get("status"),
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "ERROR", "error": str(exc)}


def snapshot(root: Path) -> dict[str, Any]:
    log_path = root / "sequence.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    run_logs = split_training_logs(log_text)
    preliminary = [run_status(root, label, run_logs[label]) for label in LABELS]
    active = next((row["label"] for row in preliminary if row["status"] == "RUNNING"), None)
    runs = [run_status(root, label, run_logs[label]) for label in LABELS]
    series = {label: scalar_rows(run_logs[label]) for label in LABELS}
    scalars = series.get(active, []) if active else []
    overall = sum(row["step"] for row in runs)
    errors = [
        line.strip() for line in log_text.replace("\r", "\n").splitlines() if ERROR_RE.search(line)
    ][-10:]
    continuations = epoch2_runs(root)
    continuation_errors = [message for row in continuations for message in row.get("errors", [])]
    errors = (errors + continuation_errors)[-10:]
    all_series = dict(series)
    all_series.update({row["run_id"]: row["series"] for row in continuations})
    return {
        "recipe": "BATA-STABLE-V0",
        "scope": "base to checkpoint-553, two independent sequential runs",
        "active_run": active,
        "runs": runs,
        "overall_step": overall,
        "overall_target": 1106,
        "latest_scalar": scalars[-1] if scalars else None,
        "scalars": scalars,
        "series": all_series,
        "epoch2_runs": continuations,
        "epoch2_active_run": next((row["run_id"] for row in continuations if row["status"] == "RUNNING"), None),
        "epoch2_launch_ready": gpu_launch_ready()[0],
        "comparison": comparison_status(root),
        "gpus": nvidia_snapshot(),
        "errors": errors,
        "healthy": not errors,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BATA Stable553 Monitor</title><style>
:root{color-scheme:light;--ink:#17232a;--muted:#66757d;--line:#d6dfe1;--green:#07865f;--blue:#2369a9;--amber:#aa6710;--red:#b42318;--bg:#f3f6f5;--panel:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Segoe UI,Microsoft YaHei,sans-serif;letter-spacing:0}
header{background:#14242a;color:#fff;border-bottom:3px solid #13a77b;padding:16px 28px;display:flex;justify-content:space-between;gap:20px;align-items:center}
h1{font-size:19px;margin:0}.sub{color:#b9c7ca;font-size:12px}.live{font-variant-numeric:tabular-nums;color:#91e4c8}
main{max-width:1500px;margin:auto;padding:22px 28px}.band{background:var(--panel);border:1px solid var(--line);border-radius:6px;margin-bottom:14px;padding:18px}
.topline{display:flex;justify-content:space-between;gap:18px;align-items:center}.status{font-size:18px;font-weight:700}.healthy{color:var(--green)}.bad{color:var(--red)}
.progress{height:12px;background:#e7eceb;margin:14px 0 6px;overflow:hidden}.progress>i{display:block;height:100%;background:var(--green);width:0;transition:width .35s}
.grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line)}
.metric{padding:13px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);min-height:76px}.label{font-size:12px;color:var(--muted)}.value{font-size:21px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
.runs{display:grid;grid-template-columns:1fr 1fr;gap:14px}.run{border-left:4px solid var(--line);padding:13px 14px;background:#f8faf9;min-height:102px}.run.running{border-color:var(--blue)}.run.pass{border-color:var(--green)}.runhead{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}.runmeta{font-variant-numeric:tabular-nums}.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.button,button,select{font:inherit;border:1px solid #aebcc0;background:#fff;color:var(--ink);border-radius:4px;padding:7px 10px}.button,button{cursor:pointer}.button:hover,button:hover{border-color:var(--green);color:var(--green)}button:disabled{cursor:not-allowed;color:#9aa5a9;border-color:#d7dfe1;background:#f1f4f3}.download-state{font-size:11px;color:var(--muted);width:100%;text-align:right}.download-progress{height:5px;background:#dfe7e5;width:180px;margin-left:auto;overflow:hidden}.download-progress i{display:block;height:100%;background:var(--green);transition:width .2s}
.gpurow{display:grid;grid-template-columns:70px 1fr 90px 90px;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);align-items:center}.bar{height:8px;background:#e4e9e8}.bar i{display:block;height:100%;background:var(--blue)}
.charthead{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:12px}.chartcontrols{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:18px;height:3px;margin:0 5px 3px 0}.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}.chartbox{border:1px solid var(--line);padding:12px;background:#fbfcfc}.plot{position:relative;margin-top:8px}.plot canvas{display:block;width:100%;height:300px;cursor:crosshair;touch-action:none}.tooltip{display:none;position:absolute;pointer-events:none;background:#17232a;color:#fff;border-radius:4px;padding:7px 9px;font-size:12px;line-height:1.55;white-space:nowrap;z-index:2;box-shadow:0 4px 14px #0003}.zoomreadout{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}h2{font-size:15px;margin:0}.hint{font-size:12px;color:var(--muted)}
.compare-summary{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line);margin-top:14px}.compare-cell{padding:13px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.compare-cell .value{font-size:18px}.verdict{display:inline-flex;align-items:center;border:1px solid #8bcab4;background:#edf8f4;color:#056b4d;border-radius:4px;padding:5px 9px;font-weight:700}.compare-table{width:100%;border-collapse:collapse;margin-top:14px;font-variant-numeric:tabular-nums}.compare-table th,.compare-table td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}.compare-table th{font-size:12px;color:var(--muted);font-weight:600}.definitions{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.definition{border-left:3px solid var(--blue);background:#f7f9fa;padding:10px 12px}.definition b{display:block;margin-bottom:3px}.waiting{padding:16px 0;color:var(--muted)}
.launchbar{display:flex;align-items:flex-end;gap:10px;flex-wrap:wrap}.field{display:flex;flex-direction:column;gap:4px}.field input{font:inherit;border:1px solid #aebcc0;border-radius:4px;padding:8px 10px;width:180px}.primary{background:var(--green);border-color:var(--green);color:#fff}.primary:hover{background:#067653;color:#fff}.launch-message{min-height:20px;margin-top:8px;font-size:12px}.tablewrap{overflow-x:auto;margin-top:14px}.run-table{width:100%;border-collapse:collapse;min-width:980px}.run-table th,.run-table td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:middle}.run-table th{font-size:12px;color:var(--muted)}.run-table input{font:inherit;border:1px solid #bdc8cb;border-radius:3px;padding:6px 7px}.score-input{width:90px}.notes-input{width:210px}.run-status{font-weight:700}.run-status.running{color:var(--blue)}.run-status.pass{color:var(--green)}.run-status.stopped,.run-status.invalid{color:var(--red)}
@media(max-width:900px){.grid,.compare-summary{grid-template-columns:repeat(2,1fr)}.runs,.charts,.definitions{grid-template-columns:1fr}.gpurow{grid-template-columns:50px 1fr 70px 70px}header{align-items:flex-start;flex-direction:column}}
</style></head><body><header><div><h1>BATA Stable553 首轮重复性监控</h1><div class="sub">Production-like · Base → 553 · A/B 顺序执行 · Scheduler horizon 1106</div></div><div id="clock" class="live">连接中</div></header>
<main><section class="band"><div class="topline"><div><div class="label">当前状态</div><div id="status" class="status">读取中</div></div><div id="overallText" class="value"></div></div><div class="progress"><i id="overallBar"></i></div><div class="hint">总体进度按两个独立 run 共 1106 个 optimizer updates 计算。</div></section>
<section class="band"><div id="metrics" class="grid"></div></section>
<section class="band"><h2>A/B 执行状态</h2><div id="runs" class="runs"></div></section>
<section class="band"><div class="charthead"><div><h2>Epoch 2 续训</h2><div class="hint">固定从 STABLE553-A 恢复 optimizer、scheduler 与 RNG，四卡训练至 step 1106。修改 seed 属于受控随机种子消融。</div></div><div class="launchbar"><label class="field"><span class="label">随机种子</span><input id="seedInput" type="number" min="0" max="2147483647" step="1" value="20260806"></label><button id="launchEpoch2" class="primary" type="button">启动四卡续训</button></div></div><div id="launchMessage" class="launch-message hint"></div><div class="tablewrap"><table class="run-table"><thead><tr><th>Seed</th><th>Run</th><th>状态</th><th>Step</th><th>Loss / Grad</th><th>手工成绩</th><th>备注</th><th>操作</th></tr></thead><tbody id="epoch2Rows"><tr><td colspan="8" class="hint">尚无续训记录</td></tr></tbody></table></div></section>
<section class="band"><div class="charthead"><div><h2>A/B 与 Epoch 2 训练曲线</h2><div class="legend"><span><i style="background:#07865f"></i>STABLE553-A</span><span><i style="background:#2369a9"></i>STABLE553-B</span><span id="epoch2Legend"></span></div></div><div class="chartcontrols"><label class="label" for="smooth">平滑</label><select id="smooth"><option value="1">原始</option><option value="3">3 点</option><option value="5">5 点</option><option value="9">9 点</option></select><button id="resetZoom" type="button">重置视图</button><span id="zoomReadout" class="zoomreadout"></span></div></div><div class="charts"><div class="chartbox"><div class="label">Loss</div><div class="plot"><canvas id="loss"></canvas><div class="tooltip"></div></div></div><div class="chartbox"><div class="label">Gradient norm</div><div class="plot"><canvas id="grad"></canvas><div class="tooltip"></div></div></div></div><div class="hint">滚轮缩放时间轴，按住拖动平移，双击重置；续训曲线接在 step 553 后。数据来自 logging_steps=5。</div></section>
<section class="band"><div class="topline"><div><h2>Checkpoint-553 权重比较</h2><div class="hint">A/B 独立复现相互比较，并分别以本轮 A 对历史 2026-08-12 checkpoint-553。</div></div><div id="compareVerdict"></div></div><div id="comparison" class="waiting">正在读取比较结果…</div></section>
<section class="band"><h2>GPU</h2><div id="gpus"></div></section>
</main><script>
const COLORS={'STABLE553-A':'#07865f','STABLE553-B':'#2369a9'},BASE_LABELS=['STABLE553-A','STABLE553-B'];
const f=(v,n=6)=>v==null?'—':Number(v).toFixed(n);let state=null,smoothN=1,view={min:0,max:25,auto:true},drag=null,launchMessage='';const downloadStates={},scoreDrafts={};
function metric(label,value){return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`}
function smooth(rows,key,n){return rows.map((row,i)=>{const from=Math.max(0,i-n+1),slice=rows.slice(from,i+1).map(x=>Number(x[key])).filter(Number.isFinite);return {...row,[key]:slice.length?slice.reduce((a,b)=>a+b,0)/slice.length:null}})}
function seriesLabels(){const latest=state?.epoch2_runs?.[0]?.run_id;return latest?[...BASE_LABELS,latest]:BASE_LABELS}
function colorFor(label){return COLORS[label]||(label.startsWith('EPOCH2-')?'#aa6710':'#6f42c1')}
function allSeries(key){const src=state?.series||{};return seriesLabels().map(label=>({label,rows:smooth(src[label]||[],key,smoothN)}))}
function resetView(){const labels=seriesLabels(),steps=labels.flatMap(label=>(state?.series?.[label]||[]).map(x=>Number(x.step))).filter(Number.isFinite),limit=state?.epoch2_runs?.length?1106:553,mx=Math.max(25,...steps);view={min:0,max:Math.min(limit,Math.max(25,mx)),auto:true};drawAll()}
function drawChart(id,key){const c=document.getElementById(id),box=c.parentElement,tip=box.querySelector('.tooltip'),d=devicePixelRatio||1,w=Math.max(320,c.clientWidth),h=300;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,w,h);const m={l:54,r:16,t:16,b:34},pw=w-m.l-m.r,ph=h-m.t-m.b,series=allSeries(key),visible=series.map(s=>({...s,rows:s.rows.filter(r=>r.step>=view.min&&r.step<=view.max&&Number.isFinite(Number(r[key])))})),vals=visible.flatMap(s=>s.rows.map(r=>Number(r[key])));x.strokeStyle='#e0e6e7';x.fillStyle='#718087';x.font='11px Segoe UI';for(let i=0;i<=4;i++){const py=m.t+ph*i/4;x.beginPath();x.moveTo(m.l,py);x.lineTo(w-m.r,py);x.stroke()}if(!vals.length){x.fillText('等待标量数据',m.l+12,m.t+24);return}let ymin=Math.min(...vals),ymax=Math.max(...vals);const pad=(ymax-ymin||Math.max(Math.abs(ymax),1))*.08;ymin-=pad;ymax+=pad;const xp=step=>m.l+pw*(step-view.min)/(view.max-view.min||1),yp=value=>m.t+ph*(1-(value-ymin)/(ymax-ymin||1));for(let i=0;i<=4;i++){const val=ymax-(ymax-ymin)*i/4;x.fillText(val.toFixed(key==='loss'?2:3),4,m.t+ph*i/4+4)}[view.min,(view.min+view.max)/2,view.max].forEach((step,i)=>x.fillText(Math.round(step),m.l+pw*i/2-(i===0?0:i===1?8:18),h-10));visible.forEach(s=>{if(!s.rows.length)return;x.strokeStyle=colorFor(s.label);x.lineWidth=2;x.beginPath();s.rows.forEach((r,i)=>{const px=xp(r.step),py=yp(Number(r[key]));i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});if(c._hoverStep!=null){const hs=c._hoverStep,px=xp(hs);x.strokeStyle='#7b898f';x.setLineDash([4,4]);x.beginPath();x.moveTo(px,m.t);x.lineTo(px,m.t+ph);x.stroke();x.setLineDash([]);const lines=['Step '+Math.round(hs)];seriesLabels().forEach(label=>{const rows=(state.series?.[label]||[]),near=rows.reduce((best,r)=>Math.abs(r.step-hs)<Math.abs((best?.step??1e9)-hs)?r:best,null);if(near&&Math.abs(near.step-hs)<=3)lines.push(`${label.startsWith('EPOCH2-')?'E2':label.slice(-1)} · ${near.step}: ${f(near[key],key==='loss'?4:5)}`)});tip.innerHTML=lines.join('<br>');tip.style.display='block';tip.style.left=Math.min(w-tip.offsetWidth-6,Math.max(4,px+10))+'px';tip.style.top='8px'}else tip.style.display='none'}
function drawAll(){if(!state)return;drawChart('loss','loss');drawChart('grad','grad_norm');document.getElementById('zoomReadout').textContent=`Step ${Math.round(view.min)}–${Math.round(view.max)}`}
function bindChart(id){const c=document.getElementById(id);c.addEventListener('wheel',e=>{e.preventDefault();const rect=c.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(e.clientX-rect.left-54)/(rect.width-70))),span=view.max-view.min,factor=e.deltaY<0?.78:1.28,newSpan=Math.max(15,Math.min(553,span*factor)),center=view.min+span*ratio;view.min=Math.max(0,Math.min(553-newSpan,center-newSpan*ratio));view.max=view.min+newSpan;view.auto=false;drawAll()},{passive:false});c.addEventListener('pointerdown',e=>{drag={x:e.clientX,min:view.min,max:view.max};c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{const rect=c.getBoundingClientRect();if(drag){const delta=-(e.clientX-drag.x)*(drag.max-drag.min)/(rect.width-70),span=drag.max-drag.min;view.min=Math.max(0,Math.min(553-span,drag.min+delta));view.max=view.min+span;view.auto=false;drawAll()}else{const ratio=Math.max(0,Math.min(1,(e.clientX-rect.left-54)/(rect.width-70)));c._hoverStep=view.min+(view.max-view.min)*ratio;drawChart(id,id==='loss'?'loss':'grad_norm')}});c.addEventListener('pointerup',()=>drag=null);c.addEventListener('pointercancel',()=>drag=null);c.addEventListener('pointerleave',()=>{if(!drag){c._hoverStep=null;drawChart(id,id==='loss'?'loss':'grad_norm')}});c.addEventListener('dblclick',resetView)}
function formatBytes(value){if(value<1024)return value+' B';if(value<1024*1024)return (value/1024).toFixed(1)+' KiB';return (value/1024/1024).toFixed(1)+' MiB'}
function setDownloadState(label,patch){downloadStates[label]={...(downloadStates[label]||{}),...patch};renderRuns();renderEpoch2()}
function renderRuns(){if(!state)return;document.getElementById('runs').innerHTML=state.runs.map(x=>{const d=downloadStates[x.label]||{},busy=!!d.busy,ready=x.download_ready&&!busy,text=d.text||(x.download_ready?'权重 + config':'checkpoint 完成后可下载'),progress=Number.isFinite(d.progress)?`<div class="download-progress"><i style="width:${d.progress}%"></i></div>`:'';return `<div class="run ${x.status.toLowerCase()}"><div class="runhead"><div class="runmeta"><b>${x.label}</b><div>${x.status} · ${x.step}/553</div><div class="hint">4-rank evidence ${x.rank_evidence_count}/4 · checkpoint ${x.checkpoint_complete?'完整':'未生成'}</div></div><div class="actions"><button type="button" ${ready?'':'disabled'} onclick="downloadAdapter('${x.label}')">${busy?'正在下载…':'下载 Adapter 两文件'}</button><div class="download-state">${text}</div>${progress}</div></div></div>`}).join('')}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function renderEpoch2(){if(!state)return;const rows=state.epoch2_runs||[],body=document.getElementById('epoch2Rows'),active=rows.some(x=>x.status==='RUNNING'),button=document.getElementById('launchEpoch2');button.disabled=active||!state.epoch2_launch_ready;document.getElementById('launchMessage').textContent=launchMessage||(active?'续训进行中；完成或停止后才能启动下一组 seed。':state.epoch2_launch_ready?'四张 GPU 空闲，可以启动。':'GPU 当前不可用于新续训。');document.getElementById('epoch2Legend').innerHTML=rows.length?`<i style="background:#aa6710"></i>${esc(rows[0].run_id)}`:'';if(!rows.length){body.innerHTML='<tr><td colspan="8" class="hint">尚无续训记录</td></tr>';return}rows.forEach(x=>{if(!scoreDrafts[x.run_id])scoreDrafts[x.run_id]={score:x.manual_score?.score??'',notes:x.manual_score?.notes??''}});body.innerHTML=rows.map(x=>{const d=downloadStates[x.run_id]||{},draft=scoreDrafts[x.run_id],latest=x.latest_scalar||{},busy=!!d.busy;return `<tr><td>${x.seed}</td><td><code>${esc(x.run_id)}</code></td><td><span class="run-status ${x.status.toLowerCase()}">${x.status}</span></td><td>${x.step}/1106</td><td>${f(latest.loss,4)} / ${f(latest.grad_norm,4)}</td><td><input class="score-input" type="number" step="0.0001" value="${esc(draft.score)}" oninput="scoreDrafts['${x.run_id}'].score=this.value"></td><td><input class="notes-input" maxlength="500" value="${esc(draft.notes)}" oninput="scoreDrafts['${x.run_id}'].notes=this.value"></td><td><button type="button" onclick="saveScore('${x.run_id}')">保存成绩</button> <button type="button" ${x.download_ready&&!busy?'':'disabled'} onclick="downloadEpoch2Adapter('${x.run_id}')">${busy?'下载中…':'下载 Adapter'}</button><div class="download-state">${esc(d.text||'')}</div></td></tr>`}).join('')}
async function postJson(url,value){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Stable553-Action':'1'},body:JSON.stringify(value)}),payload=await response.json();if(!response.ok)throw new Error(payload.error||('HTTP '+response.status));return payload}
async function launchEpoch2(){const seed=Number(document.getElementById('seedInput').value);if(!Number.isInteger(seed)||seed<0||seed>2147483647){launchMessage='Seed 必须是 0 到 2147483647 的整数。';renderEpoch2();return}if(!confirm(`从 STABLE553-A 启动四卡续训到 step 1106？\nSeed: ${seed}`))return;launchMessage='正在执行合同检查并启动…';renderEpoch2();try{const result=await postJson('/api/epoch2/start',{seed});launchMessage='已启动 '+result.run_id;await refresh()}catch(e){launchMessage='启动失败：'+e.message;renderEpoch2()}}
async function saveScore(runId){const draft=scoreDrafts[runId],score=draft.score===''?null:Number(draft.score);if(score!==null&&!Number.isFinite(score)){launchMessage='成绩必须是数字或留空。';renderEpoch2();return}try{await postJson('/api/epoch2/score',{run_id:runId,score,notes:draft.notes});launchMessage='成绩已保存：'+runId;await refresh()}catch(e){launchMessage='成绩保存失败：'+e.message;renderEpoch2()}}
function pct(v){return v==null?'—':(Number(v)*100).toFixed(4)+'%'}
function yes(v){return v?'一致':'不一致'}
function renderComparison(){const c=state?.comparison||{status:'WAITING'},box=document.getElementById('comparison'),badge=document.getElementById('compareVerdict');if(c.status!=='READY'){badge.innerHTML='';box.className='waiting';box.textContent=c.status==='ERROR'?'比较结果读取失败：'+c.error:'比较尚未完成';return}const p=c.pairwise,h=c.historical,ls=h.layer_summary.cosine,projections=Object.entries(h.projections).sort(([a],[b])=>a.localeCompare(b));badge.innerHTML=`<span class="verdict">${c.verdict}</span>`;box.className='';box.innerHTML=`<div class="compare-summary"><div class="compare-cell"><div class="label">A ↔ B Raw LoRA cosine</div><div class="value">${f(p.raw_lora.cosine,9)}</div></div><div class="compare-cell"><div class="label">A ↔ B Raw relative L2</div><div class="value">${pct(p.raw_lora.relative_l2)}</div></div><div class="compare-cell"><div class="label">A ↔ B Effective B@A cosine</div><div class="value">${f(p.effective_ba.cosine,9)}</div></div><div class="compare-cell"><div class="label">A ↔ B Effective relative L2</div><div class="value">${pct(p.effective_ba.relative_l2)}</div></div><div class="compare-cell"><div class="label">Adapter bytes / tensors</div><div class="value">${yes(p.adapter_file_sha_exact)} / ${yes(p.adapter_canonical_sha_exact)}</div></div><div class="compare-cell"><div class="label">Optimizer / Scheduler / RNG</div><div class="value">${yes(p.optimizer_exact)} / ${yes(p.scheduler_exact)} / ${yes(p.rng_exact)}</div></div><div class="compare-cell"><div class="label">A ↔ 历史 Raw cosine</div><div class="value">${f(h.raw_lora.cosine,6)}</div></div><div class="compare-cell"><div class="label">A ↔ 历史 Effective cosine</div><div class="value">${f(h.effective_ba.cosine,6)}</div></div></div><table class="compare-table"><thead><tr><th>历史对比维度</th><th>Cosine</th><th>Relative L2</th></tr></thead><tbody><tr><td>Raw LoRA</td><td>${f(h.raw_lora.cosine,6)}</td><td>${pct(h.raw_lora.relative_l2)}</td></tr><tr><td>Effective B@A</td><td>${f(h.effective_ba.cosine,6)}</td><td>${pct(h.effective_ba.relative_l2)}</td></tr>${projections.map(([name,v])=>`<tr><td>Projection · ${name}</td><td>${f(v.cosine,6)}</td><td>${pct(v.relative_l2)}</td></tr>`).join('')}<tr><td>Layer cosine min / mean / max</td><td colspan="2">${f(ls.min,6)} / ${f(ls.mean,6)} / ${f(ls.max,6)}</td></tr></tbody></table><div class="definitions"><div class="definition"><b>Raw LoRA</b>直接比较 checkpoint 中保存的 LoRA A/B 张量。Cosine 越接近 1、Relative L2 越接近 0，参数越一致。</div><div class="definition"><b>Effective B@A</b>比较 LoRA 实际施加到基础权重上的低秩更新矩阵，比单独比较 A/B 因子更接近功能差异。A/B 完全一致只证明训练波动已消除，不代表恢复了历史效果。</div></div><div class="hint" style="margin-top:10px">外部评测：${c.external_evaluation||'NOT_EXECUTED'}。本页权重比较不替代效果评测。</div>`}
async function downloadBundle(label,urlFor){const files=['adapter_model.safetensors','adapter_config.json'];let writer=null;setDownloadState(label,{busy:true,progress:0,text:'请选择保存目录'});try{if(window.showDirectoryPicker){const parent=await window.showDirectoryPicker({mode:'readwrite'}),dir=await parent.getDirectoryHandle(label,{create:true});for(const name of files){const response=await fetch(urlFor(name));if(!response.ok)throw new Error('HTTP '+response.status);const total=Number(response.headers.get('Content-Length'))||0,handle=await dir.getFileHandle(name,{create:true}),reader=response.body.getReader();writer=await handle.createWritable();let received=0,lastPercent=-1;while(true){const {done,value}=await reader.read();if(done)break;await writer.write(value);received+=value.byteLength;const percent=total?Math.min(100,Math.floor(received*100/total)):0;if(percent!==lastPercent){lastPercent=percent;setDownloadState(label,{progress:percent,text:`${name} · ${formatBytes(received)} / ${total?formatBytes(total):'未知大小'} · ${percent}%`})}}await writer.close();writer=null}setDownloadState(label,{busy:false,progress:100,text:'下载完成 · 已保存至 '+label+' 文件夹'})}else{files.forEach((name,i)=>setTimeout(()=>{const a=document.createElement('a');a.href=urlFor(name);a.download=name;document.body.appendChild(a);a.click();a.remove()},i*500));setDownloadState(label,{busy:false,progress:null,text:'已提交两个文件下载'})}}catch(e){if(writer){try{await writer.abort()}catch{}}setDownloadState(label,{busy:false,progress:null,text:e.name==='AbortError'?'已取消选择':'下载失败：'+e.message})}}
function downloadAdapter(label){return downloadBundle(label,name=>`/api/checkpoint/${encodeURIComponent(label)}/${name}`)}
function downloadEpoch2Adapter(runId){return downloadBundle(runId,name=>`/api/epoch2/checkpoint/${encodeURIComponent(runId)}/${name}`)}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),d=await r.json(),continuation=d.epoch2_runs?.find(x=>x.status==='RUNNING'),s=continuation?.latest_scalar||d.latest_scalar||{},previousMax=view.max;state=d;document.getElementById('clock').textContent='实时 · '+new Date().toLocaleTimeString();document.getElementById('status').textContent=d.healthy?(continuation?`Epoch 2 续训中 · Seed ${continuation.seed}`:d.comparison?.status==='READY'?'A/B 已完成 · 权重比较已生成':d.overall_step===1106?'A/B 已完成':'等待启动'):'检测到错误';document.getElementById('status').className='status '+(d.healthy?'healthy':'bad');const displayedStep=continuation?.step??d.overall_step,displayedTarget=continuation?1106:d.overall_target;document.getElementById('overallText').textContent=displayedStep+' / '+displayedTarget;document.getElementById('overallBar').style.width=(100*displayedStep/displayedTarget)+'%';document.getElementById('metrics').innerHTML=metric('当前 optimizer step',continuation?.step??'—')+metric('最新 logging step',s.step??'—')+metric('Loss',f(s.loss))+metric('Grad norm',f(s.grad_norm))+metric('Learning rate',s.learning_rate==null?'—':Number(s.learning_rate).toExponential(4))+metric('Epoch',f(s.epoch,4));renderRuns();renderEpoch2();renderComparison();document.getElementById('gpus').innerHTML=d.gpus.map(g=>`<div class="gpurow"><b>GPU ${g.index}</b><div class="bar"><i style="width:${100*g.memory_used/g.memory_total}%"></i></div><span>${g.memory_used} MiB</span><span>${g.utilization}%</span></div>`).join('');if(view.auto){const labels=seriesLabels(),steps=labels.flatMap(label=>(d.series?.[label]||[]).map(x=>x.step)),limit=d.epoch2_runs?.length?1106:553,mx=Math.max(25,...steps);view.max=Math.min(limit,Math.max(previousMax,mx))}drawAll()}catch(e){document.getElementById('status').textContent='监控连接失败';document.getElementById('status').className='status bad'}}
bindChart('loss');bindChart('grad');document.getElementById('smooth').addEventListener('change',e=>{smoothN=Number(e.target.value);drawAll()});document.getElementById('resetZoom').addEventListener('click',resetView);document.getElementById('launchEpoch2').addEventListener('click',launchEpoch2);refresh();setInterval(refresh,5000);addEventListener('resize',drawAll);
</script></body></html>"""


def create_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, value: dict[str, Any]) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, path: Path) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)

        def do_GET(self):
            request_path = unquote(urlparse(self.path).path)
            parts = request_path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "checkpoint"]:
                path = checkpoint_download_path(root, parts[2], parts[3])
                if path is None:
                    self.send_error(404)
                    return
                self.send_file(path)
                return
            if len(parts) == 5 and parts[:3] == ["api", "epoch2", "checkpoint"]:
                path = epoch2_checkpoint_download_path(root, parts[3], parts[4])
                if path is None:
                    self.send_error(404)
                    return
                self.send_file(path)
                return
            if request_path == "/api/status":
                payload = json.dumps(snapshot(root), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif request_path in ("/", "/index.html"):
                payload = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif request_path == "/healthz":
                payload = b'{"status":"ok"}'
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            request_path = unquote(urlparse(self.path).path)
            if self.headers.get("X-Stable553-Action") != "1":
                self.send_json(403, {"error": "missing action confirmation header"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= 16 * 1024:
                self.send_json(400, {"error": "invalid request body size"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                with TRAIN_ACTION_LOCK:
                    if request_path == "/api/epoch2/start":
                        result = start_epoch2_run(root, body.get("seed"))
                        self.send_json(202, result)
                    elif request_path == "/api/epoch2/score":
                        record = save_manual_score(
                            root,
                            body.get("run_id"),
                            body.get("score"),
                            body.get("notes", ""),
                        )
                        self.send_json(200, {"status": "SAVED", **record})
                    else:
                        self.send_json(404, {"error": "unknown endpoint"})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.send_json(409, {"error": str(exc)})
            except (OSError, subprocess.CalledProcessError) as exc:
                self.send_json(500, {"error": f"launch preparation failed: {type(exc).__name__}"})

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), create_handler(args.root)).serve_forever()


if __name__ == "__main__":
    main()
