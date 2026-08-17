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
  --runs-dir /data/GRPO/runs \
  --port 8765
```

Open `http://127.0.0.1:8765`.

The Chinese dashboard lists experiments under `--runs-dir`; every data request
is scoped to the selected `run_id`, so metrics and rollout traces cannot mix
between experiments. The older `--run-dir <one-run>` mode remains supported.
The selected run is also stored in the page URL for refresh/share continuity.

The dashboard polls append-only data every three seconds. Clicking a
chart opens an enlarged view with 20/50/all-point ranges. Think traces retain
the 32 SIDs that were already parsed for reward and highlight exact gold, AB
prefix, A prefix, and invalid outputs. Beam SID retention is monitor-only and
does not add generation or parsing work.

NoThink traces reuse the existing group metadata gather to assemble one full
G=8 group with candidate SID, reward level, gold match, and token length. This
adds no DDP collective. Every metric panel has an on-demand definition and a
short description of a healthy trend; metrics without a monotonic optimum are
explicitly described as joint diagnostics rather than "higher is better".

The server exposes `/api/runs`, `/api/manifest`, `/api/metrics`, `/api/rollouts`,
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
