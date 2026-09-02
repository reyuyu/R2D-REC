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
from urllib.parse import unquote, urlparse


LABELS = ("STABLE553-A", "STABLE553-B")
TRAINING_MARKER = "***** Running training *****"
DOWNLOAD_FILES = ("adapter_model.safetensors", "adapter_config.json")
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
    return {
        "recipe": "BATA-STABLE-V0",
        "scope": "base to checkpoint-553, two independent sequential runs",
        "active_run": active,
        "runs": runs,
        "overall_step": overall,
        "overall_target": 1106,
        "latest_scalar": scalars[-1] if scalars else None,
        "scalars": scalars,
        "series": series,
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
.button,button,select{font:inherit;border:1px solid #aebcc0;background:#fff;color:var(--ink);border-radius:4px;padding:7px 10px}.button,button{cursor:pointer}.button:hover,button:hover{border-color:var(--green);color:var(--green)}button:disabled{cursor:not-allowed;color:#9aa5a9;border-color:#d7dfe1;background:#f1f4f3}.download-state{font-size:11px;color:var(--muted);width:100%;text-align:right}
.gpurow{display:grid;grid-template-columns:70px 1fr 90px 90px;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);align-items:center}.bar{height:8px;background:#e4e9e8}.bar i{display:block;height:100%;background:var(--blue)}
.charthead{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:12px}.chartcontrols{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:18px;height:3px;margin:0 5px 3px 0}.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}.chartbox{border:1px solid var(--line);padding:12px;background:#fbfcfc}.plot{position:relative;margin-top:8px}.plot canvas{display:block;width:100%;height:300px;cursor:crosshair;touch-action:none}.tooltip{display:none;position:absolute;pointer-events:none;background:#17232a;color:#fff;border-radius:4px;padding:7px 9px;font-size:12px;line-height:1.55;white-space:nowrap;z-index:2;box-shadow:0 4px 14px #0003}.zoomreadout{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}h2{font-size:15px;margin:0}.hint{font-size:12px;color:var(--muted)}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}.runs,.charts{grid-template-columns:1fr}.gpurow{grid-template-columns:50px 1fr 70px 70px}header{align-items:flex-start;flex-direction:column}}
</style></head><body><header><div><h1>BATA Stable553 首轮重复性监控</h1><div class="sub">Production-like · Base → 553 · A/B 顺序执行 · Scheduler horizon 1106</div></div><div id="clock" class="live">连接中</div></header>
<main><section class="band"><div class="topline"><div><div class="label">当前状态</div><div id="status" class="status">读取中</div></div><div id="overallText" class="value"></div></div><div class="progress"><i id="overallBar"></i></div><div class="hint">总体进度按两个独立 run 共 1106 个 optimizer updates 计算。</div></section>
<section class="band"><div id="metrics" class="grid"></div></section>
<section class="band"><h2>A/B 执行状态</h2><div id="runs" class="runs"></div></section>
<section class="band"><div class="charthead"><div><h2>A/B 训练曲线对照</h2><div class="legend"><span><i style="background:#07865f"></i>STABLE553-A</span><span><i style="background:#2369a9"></i>STABLE553-B</span></div></div><div class="chartcontrols"><label class="label" for="smooth">平滑</label><select id="smooth"><option value="1">原始</option><option value="3">3 点</option><option value="5">5 点</option><option value="9">9 点</option></select><button id="resetZoom" type="button">重置视图</button><span id="zoomReadout" class="zoomreadout"></span></div></div><div class="charts"><div class="chartbox"><div class="label">Loss</div><div class="plot"><canvas id="loss"></canvas><div class="tooltip"></div></div></div><div class="chartbox"><div class="label">Gradient norm</div><div class="plot"><canvas id="grad"></canvas><div class="tooltip"></div></div></div></div><div class="hint">滚轮缩放时间轴，按住拖动平移，双击重置；悬停会同时显示 A/B 同一步数值。数据仍来自 logging_steps=5，不执行额外计算。</div></section>
<section class="band"><h2>GPU</h2><div id="gpus"></div></section>
</main><script>
const COLORS={'STABLE553-A':'#07865f','STABLE553-B':'#2369a9'},LABELS=['STABLE553-A','STABLE553-B'];
const f=(v,n=6)=>v==null?'—':Number(v).toFixed(n);let state=null,smoothN=1,view={min:0,max:25,auto:true},drag=null;
function metric(label,value){return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`}
function smooth(rows,key,n){return rows.map((row,i)=>{const from=Math.max(0,i-n+1),slice=rows.slice(from,i+1).map(x=>Number(x[key])).filter(Number.isFinite);return {...row,[key]:slice.length?slice.reduce((a,b)=>a+b,0)/slice.length:null}})}
function allSeries(key){const src=state?.series||{};return LABELS.map(label=>({label,rows:smooth(src[label]||[],key,smoothN)}))}
function resetView(){const steps=LABELS.flatMap(label=>(state?.series?.[label]||[]).map(x=>Number(x.step))).filter(Number.isFinite),mx=Math.max(25,...steps);view={min:0,max:Math.min(553,Math.max(25,mx)),auto:true};drawAll()}
function drawChart(id,key){const c=document.getElementById(id),box=c.parentElement,tip=box.querySelector('.tooltip'),d=devicePixelRatio||1,w=Math.max(320,c.clientWidth),h=300;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,w,h);const m={l:54,r:16,t:16,b:34},pw=w-m.l-m.r,ph=h-m.t-m.b,series=allSeries(key),visible=series.map(s=>({...s,rows:s.rows.filter(r=>r.step>=view.min&&r.step<=view.max&&Number.isFinite(Number(r[key])))})),vals=visible.flatMap(s=>s.rows.map(r=>Number(r[key])));x.strokeStyle='#e0e6e7';x.fillStyle='#718087';x.font='11px Segoe UI';for(let i=0;i<=4;i++){const py=m.t+ph*i/4;x.beginPath();x.moveTo(m.l,py);x.lineTo(w-m.r,py);x.stroke()}if(!vals.length){x.fillText('等待标量数据',m.l+12,m.t+24);return}let ymin=Math.min(...vals),ymax=Math.max(...vals);const pad=(ymax-ymin||Math.max(Math.abs(ymax),1))*.08;ymin-=pad;ymax+=pad;const xp=step=>m.l+pw*(step-view.min)/(view.max-view.min||1),yp=value=>m.t+ph*(1-(value-ymin)/(ymax-ymin||1));for(let i=0;i<=4;i++){const val=ymax-(ymax-ymin)*i/4;x.fillText(val.toFixed(key==='loss'?2:3),4,m.t+ph*i/4+4)}[view.min,(view.min+view.max)/2,view.max].forEach((step,i)=>x.fillText(Math.round(step),m.l+pw*i/2-(i===0?0:i===1?8:18),h-10));visible.forEach(s=>{if(!s.rows.length)return;x.strokeStyle=COLORS[s.label];x.lineWidth=2;x.beginPath();s.rows.forEach((r,i)=>{const px=xp(r.step),py=yp(Number(r[key]));i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});if(c._hoverStep!=null){const hs=c._hoverStep,px=xp(hs);x.strokeStyle='#7b898f';x.setLineDash([4,4]);x.beginPath();x.moveTo(px,m.t);x.lineTo(px,m.t+ph);x.stroke();x.setLineDash([]);const lines=['Step '+Math.round(hs)];LABELS.forEach(label=>{const rows=(state.series?.[label]||[]),near=rows.reduce((best,r)=>Math.abs(r.step-hs)<Math.abs((best?.step??1e9)-hs)?r:best,null);if(near&&Math.abs(near.step-hs)<=3)lines.push(`${label.slice(-1)} · ${near.step}: ${f(near[key],key==='loss'?4:5)}`)});tip.innerHTML=lines.join('<br>');tip.style.display='block';tip.style.left=Math.min(w-tip.offsetWidth-6,Math.max(4,px+10))+'px';tip.style.top='8px'}else tip.style.display='none'}
function drawAll(){if(!state)return;drawChart('loss','loss');drawChart('grad','grad_norm');document.getElementById('zoomReadout').textContent=`Step ${Math.round(view.min)}–${Math.round(view.max)}`}
function bindChart(id){const c=document.getElementById(id);c.addEventListener('wheel',e=>{e.preventDefault();const rect=c.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(e.clientX-rect.left-54)/(rect.width-70))),span=view.max-view.min,factor=e.deltaY<0?.78:1.28,newSpan=Math.max(15,Math.min(553,span*factor)),center=view.min+span*ratio;view.min=Math.max(0,Math.min(553-newSpan,center-newSpan*ratio));view.max=view.min+newSpan;view.auto=false;drawAll()},{passive:false});c.addEventListener('pointerdown',e=>{drag={x:e.clientX,min:view.min,max:view.max};c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{const rect=c.getBoundingClientRect();if(drag){const delta=-(e.clientX-drag.x)*(drag.max-drag.min)/(rect.width-70),span=drag.max-drag.min;view.min=Math.max(0,Math.min(553-span,drag.min+delta));view.max=view.min+span;view.auto=false;drawAll()}else{const ratio=Math.max(0,Math.min(1,(e.clientX-rect.left-54)/(rect.width-70)));c._hoverStep=view.min+(view.max-view.min)*ratio;drawChart(id,id==='loss'?'loss':'grad_norm')}});c.addEventListener('pointerup',()=>drag=null);c.addEventListener('pointercancel',()=>drag=null);c.addEventListener('pointerleave',()=>{if(!drag){c._hoverStep=null;drawChart(id,id==='loss'?'loss':'grad_norm')}});c.addEventListener('dblclick',resetView)}
async function downloadAdapter(label,button){const files=['adapter_model.safetensors','adapter_config.json'],status=button.parentElement.querySelector('.download-state');button.disabled=true;try{if(window.showDirectoryPicker){status.textContent='请选择保存目录';const parent=await window.showDirectoryPicker({mode:'readwrite'}),dir=await parent.getDirectoryHandle(label,{create:true});for(const name of files){status.textContent='正在下载 '+name;const response=await fetch(`/api/checkpoint/${encodeURIComponent(label)}/${name}`);if(!response.ok)throw new Error('HTTP '+response.status);const handle=await dir.getFileHandle(name,{create:true}),writer=await handle.createWritable();await response.body.pipeTo(writer)}status.textContent='已保存至 '+label+' 文件夹'}else{files.forEach((name,i)=>setTimeout(()=>{const a=document.createElement('a');a.href=`/api/checkpoint/${encodeURIComponent(label)}/${name}`;a.download=name;document.body.appendChild(a);a.click();a.remove()},i*500));status.textContent='已提交两个文件下载'}}catch(e){status.textContent=e.name==='AbortError'?'已取消选择':'下载失败：'+e.message}finally{button.disabled=false}}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),d=await r.json(),s=d.latest_scalar||{},previousMax=view.max;state=d;document.getElementById('clock').textContent='实时 · '+new Date().toLocaleTimeString();document.getElementById('status').textContent=d.healthy?(d.active_run?d.active_run+' 训练中':d.overall_step===1106?'A/B 已完成':'等待启动'):'检测到错误';document.getElementById('status').className='status '+(d.healthy?'healthy':'bad');document.getElementById('overallText').textContent=d.overall_step+' / '+d.overall_target;document.getElementById('overallBar').style.width=(100*d.overall_step/d.overall_target)+'%';document.getElementById('metrics').innerHTML=metric('当前 optimizer step',d.runs.find(x=>x.label===d.active_run)?.step??'—')+metric('最新 logging step',s.step??'—')+metric('Loss',f(s.loss))+metric('Grad norm',f(s.grad_norm))+metric('Learning rate',s.learning_rate==null?'—':Number(s.learning_rate).toExponential(4))+metric('Epoch',f(s.epoch,4));document.getElementById('runs').innerHTML=d.runs.map(x=>`<div class="run ${x.status.toLowerCase()}"><div class="runhead"><div class="runmeta"><b>${x.label}</b><div>${x.status} · ${x.step}/553</div><div class="hint">4-rank evidence ${x.rank_evidence_count}/4 · checkpoint ${x.checkpoint_complete?'完整':'未生成'}</div></div><div class="actions"><button type="button" ${x.download_ready?'':'disabled'} onclick="downloadAdapter('${x.label}',this)">下载 Adapter 两文件</button><div class="download-state">${x.download_ready?'权重 + config':'checkpoint 完成后可下载'}</div></div></div></div>`).join('');document.getElementById('gpus').innerHTML=d.gpus.map(g=>`<div class="gpurow"><b>GPU ${g.index}</b><div class="bar"><i style="width:${100*g.memory_used/g.memory_total}%"></i></div><span>${g.memory_used} MiB</span><span>${g.utilization}%</span></div>`).join('');if(view.auto){const steps=LABELS.flatMap(label=>(d.series?.[label]||[]).map(x=>x.step)),mx=Math.max(25,...steps);view.max=Math.min(553,Math.max(previousMax,mx))}drawAll()}catch(e){document.getElementById('status').textContent='监控连接失败';document.getElementById('status').className='status bad'}}
bindChart('loss');bindChart('grad');document.getElementById('smooth').addEventListener('change',e=>{smoothN=Number(e.target.value);drawAll()});document.getElementById('resetZoom').addEventListener('click',resetView);refresh();setInterval(refresh,5000);addEventListener('resize',drawAll);
</script></body></html>"""


def create_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = unquote(urlparse(self.path).path)
            parts = request_path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "checkpoint"]:
                path = checkpoint_download_path(root, parts[2], parts[3])
                if path is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        self.wfile.write(chunk)
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
