# GRPO Monitoring Phase 1

Passive monitoring data flow:

```text
GRPO ranks -> append-only JSONL -> independent FastAPI -> static dashboard
```

The training writer has no torch, HTTP, database, thread, or fsync dependency.
All write errors are fail-open. Monitoring is disabled unless
`GRPO_MONITOR=1` is explicitly set.

## Training configuration

```bash
export GRPO_MONITOR=1
export GRPO_RUN_ID=mini-fix-u3k-epoch1
export GRPO_MONITOR_DIR=/data/GRPO/runs
export GRPO_MONITOR_STEP_EVERY=1
export GRPO_MONITOR_ROLLOUT_EVERY=1
export GRPO_TRACE_EVERY=20
```

Run layout:

```text
runs/<RUN_ID>/
├── manifest.json
├── metrics.jsonl
├── rollouts.jsonl
├── ranks/
│   ├── rank0.jsonl
│   └── rankN.jsonl
├── traces/
│   └── traces.jsonl
└── probes/                 # reserved for Phase 2
```

## CPU-only demo

From `/data/GRPO/scripts`:

```bash
python monitor/generate_demo_run.py \
  --output-dir /data/GRPO/runs \
  --run-id demo-phase1

python monitor/server.py \
  --run-dir /data/GRPO/runs/demo-phase1 \
  --port 8765
```

Open `http://127.0.0.1:8765`.

The server exposes `/api/manifest`, `/api/metrics`, `/api/rollouts`,
`/api/ranks`, and `/api/traces`. JSONL endpoints accept `from_step`,
`to_step`, `route`, and `rollout_id`; `/api/ranks` additionally accepts
`rank`.

## CPU tests

```bash
cd /data/GRPO/scripts
python -m monitor.test_monitor
python test_beam_prompt_cache.py
python test_correctness_v2.py
```

Phase 2 may add fixed checkpoint probes under `probes/`. Phase 1 never loads a
model or generates an additional completion for monitoring.
