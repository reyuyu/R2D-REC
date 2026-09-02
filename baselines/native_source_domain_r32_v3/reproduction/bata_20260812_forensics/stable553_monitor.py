#!/usr/bin/env python3
"""Read-only local dashboard for sequential STABLE553-A/B runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LABELS = ("STABLE553-A", "STABLE553-B")
STEP_RE = re.compile(r"(\d+)/1106")
ERROR_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|non-finite|"
    r"(?:^|[^A-Za-z])(?:nan|inf)(?:[^A-Za-z]|$)",
    re.IGNORECASE,
)


def scalar_rows(log_text: str) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    inferred_step = 0
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
    }


def snapshot(root: Path) -> dict[str, Any]:
    log_path = root / "sequence.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    preliminary = [run_status(root, label, "") for label in LABELS]
    active = next((row["label"] for row in preliminary if row["status"] == "RUNNING"), None)
    marker = "***** Running training *****"
    active_log = log_text[log_text.rfind(marker):] if active and marker in log_text else log_text
    runs = [run_status(root, label, active_log if label == active else log_text) for label in LABELS]
    scalars = scalar_rows(active_log)
    overall = sum(row["step"] for row in runs)
    errors = [
        line.strip() for line in log_text.replace("\r", "\n").splitlines() if ERROR_RE.search(line)
    ][-10:]
    return {
        "recipe": "BATA-STABLE-V0",
        "scope": "base to checkpoint-553, two independent sequential runs",
        "active_run": active,
        "runs": runs,
        "overall_step": overall,
        "overall_target": 1106,
        "latest_scalar": scalars[-1] if scalars else None,
        "scalars": scalars,
        "gpus": nvidia_snapshot(),
        "errors": errors,
        "healthy": not errors,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BATA Stable553 Monitor</title><style>
:root{color-scheme:light;--ink:#162128;--muted:#68767d;--line:#d8e0e2;--green:#07865f;--blue:#2267a8;--red:#b42318;--bg:#f4f7f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Segoe UI,Microsoft YaHei,sans-serif;letter-spacing:0}
header{background:#14242a;color:#fff;border-bottom:3px solid #13a77b;padding:16px 28px;display:flex;justify-content:space-between;gap:20px;align-items:center}
h1{font-size:19px;margin:0}.sub{color:#b9c7ca;font-size:12px}.live{font-variant-numeric:tabular-nums;color:#91e4c8}
main{max-width:1440px;margin:auto;padding:22px 28px}.band{background:#fff;border:1px solid var(--line);border-radius:6px;margin-bottom:14px;padding:18px}
.topline{display:flex;justify-content:space-between;gap:18px;align-items:center}.status{font-size:18px;font-weight:700}.healthy{color:var(--green)}.bad{color:var(--red)}
.progress{height:12px;background:#e7eceb;margin:14px 0 6px;overflow:hidden}.progress>i{display:block;height:100%;background:var(--green);width:0;transition:width .35s}
.grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line)}
.metric{padding:13px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);min-height:76px}.label{font-size:12px;color:var(--muted)}.value{font-size:21px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
.runs{display:grid;grid-template-columns:1fr 1fr;gap:14px}.run{border-left:4px solid var(--line);padding:10px 14px;background:#f8faf9}.run.running{border-color:var(--blue)}.run.pass{border-color:var(--green)}
.gpurow{display:grid;grid-template-columns:70px 1fr 90px 90px;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);align-items:center}.bar{height:8px;background:#e4e9e8}.bar i{display:block;height:100%;background:var(--blue)}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}canvas{width:100%;height:260px;border-top:1px solid var(--line)}h2{font-size:15px;margin:0 0 12px}.hint{font-size:12px;color:var(--muted)}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}.runs,.charts{grid-template-columns:1fr}.gpurow{grid-template-columns:50px 1fr 70px 70px}header{align-items:flex-start;flex-direction:column}}
</style></head><body><header><div><h1>BATA Stable553 首轮重复性监控</h1><div class="sub">Production-like · Base → 553 · A/B 顺序执行 · Scheduler horizon 1106</div></div><div id="clock" class="live">连接中</div></header>
<main><section class="band"><div class="topline"><div><div class="label">当前状态</div><div id="status" class="status">读取中</div></div><div id="overallText" class="value"></div></div><div class="progress"><i id="overallBar"></i></div><div class="hint">总体进度按两个独立 run 共 1106 个 optimizer updates 计算。</div></section>
<section class="band"><div id="metrics" class="grid"></div></section>
<section class="band"><h2>A/B 执行状态</h2><div id="runs" class="runs"></div></section>
<section class="band"><h2>训练曲线</h2><div class="charts"><div><div class="label">Loss</div><canvas id="loss"></canvas></div><div><div class="label">Gradient norm</div><canvas id="grad"></canvas></div></div><div class="hint">曲线使用正常 Trainer logging_steps=5 的标量，不执行额外 forward/backward。</div></section>
<section class="band"><h2>GPU</h2><div id="gpus"></div></section>
</main><script>
const f=(v,n=6)=>v==null?'—':Number(v).toFixed(n);function draw(id,rows,key,color){const c=document.getElementById(id),d=devicePixelRatio||1,w=c.clientWidth,h=260;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);x.strokeStyle='#d8e0e2';x.beginPath();x.moveTo(42,10);x.lineTo(42,h-28);x.lineTo(w-10,h-28);x.stroke();const pts=rows.filter(r=>Number.isFinite(Number(r[key])));if(!pts.length)return;const vals=pts.map(r=>Number(r[key])),mn=Math.min(...vals),mx=Math.max(...vals),span=mx-mn||1;x.strokeStyle=color;x.lineWidth=2;x.beginPath();pts.forEach((r,i)=>{const px=42+(w-56)*(pts.length===1?0:i/(pts.length-1)),py=10+(h-42)*(1-(Number(r[key])-mn)/span);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle='#68767d';x.font='11px Segoe UI';x.fillText(mx.toFixed(3),2,17);x.fillText(mn.toFixed(3),2,h-28);x.fillText('step '+pts.at(-1).step,w-75,h-8)}
function metric(label,value){return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),d=await r.json(),s=d.latest_scalar||{};document.getElementById('clock').textContent='实时 · '+new Date().toLocaleTimeString();document.getElementById('status').textContent=d.healthy?(d.active_run?d.active_run+' 训练中':d.overall_step===1106?'A/B 已完成':'等待启动'):'检测到错误';document.getElementById('status').className='status '+(d.healthy?'healthy':'bad');document.getElementById('overallText').textContent=d.overall_step+' / '+d.overall_target;document.getElementById('overallBar').style.width=(100*d.overall_step/d.overall_target)+'%';document.getElementById('metrics').innerHTML=metric('当前 optimizer step',d.runs.find(x=>x.label===d.active_run)?.step??'—')+metric('最新 logging step',s.step??'—')+metric('Loss',f(s.loss))+metric('Grad norm',f(s.grad_norm))+metric('Learning rate',s.learning_rate==null?'—':Number(s.learning_rate).toExponential(4))+metric('Epoch',f(s.epoch,4));document.getElementById('runs').innerHTML=d.runs.map(x=>`<div class="run ${x.status.toLowerCase()}"><b>${x.label}</b><div>${x.status} · ${x.step}/553</div><div class="hint">4-rank evidence ${x.rank_evidence_count}/4 · checkpoint ${x.checkpoint_complete?'完整':'未生成'}</div></div>`).join('');document.getElementById('gpus').innerHTML=d.gpus.map(g=>`<div class="gpurow"><b>GPU ${g.index}</b><div class="bar"><i style="width:${100*g.memory_used/g.memory_total}%"></i></div><span>${g.memory_used} MiB</span><span>${g.utilization}%</span></div>`).join('');draw('loss',d.scalars,'loss','#07865f');draw('grad',d.scalars,'grad_norm','#b66b16')}catch(e){document.getElementById('status').textContent='监控连接失败';document.getElementById('status').className='status bad'}}refresh();setInterval(refresh,5000);addEventListener('resize',refresh);
</script></body></html>"""


def create_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                payload = json.dumps(snapshot(root), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                payload = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif self.path == "/healthz":
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
