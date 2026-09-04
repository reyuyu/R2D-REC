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
└── probes.jsonl            # optional fixed held-out evaluation
```

DSR runs add optional, append-only streams in the same run directory:

```text
dsr_metrics.jsonl           # rollout-level Think/NoThink diagnostics
dsr_steps.jsonl             # existing Python loss/drift scalars
dsr_traces.jsonl            # sampled DSR trace details
```

Legacy runs do not need these files and require no migration.

## Recommendation and User run kinds

The same monitor serves both GRPO projects. New manifests identify their
contract explicitly:

```json
{"run_kind": "recommendation_grpo"}
{"run_kind": "user_grpo"}
```

An old or missing `run_kind` is always treated as `recommendation_grpo`, so
existing Recommendation and DSR runs keep their original pages and API
behavior without rewriting manifests. `/api/runs` returns normalized run kinds
and accepts an optional `run_kind` filter. User runs add six dashboard views:
overall, Action, Chain, Token Advantage, fixed Probe, and Rollout samples. Optional or
partially written User fields render as missing data rather than breaking the
three-second refresh loop.

## CPU-only demo

From `/data/GRPO/scripts`:

```bash
python monitor/generate_demo_run.py \
  --output-dir /data/GRPO/runs \
  --run-id demo-phase1 \
  --mode legacy

python monitor/generate_demo_run.py \
  --output-dir /data/GRPO/runs \
  --run-id demo-dsr \
  --mode dsr

python monitor/generate_user_demo_run.py \
  --output-dir /data/GRPO/runs \
  --run-id demo-user-grpo \
  --steps 40

python monitor/server.py \
  --runs-dir /data/GRPO/runs \
  --outputs-dir /data/GRPO/outputs/formal \
  --port 8765
```

When the monitor is copied outside the repository, it discovers the formal
GRPO source under `${GRPO_WORK_ROOT:-/data/GRPO/work}` so the read-only
advantage view can reconstruct the exact training credit. Set
`GRPO_FORMAL_SOURCE_ROOT=/path/to/grpo` to select an explicit source tree.
Reconstructed fields are labeled `复算`; captured rollout fields remain
labeled `实采`.

Open `http://127.0.0.1:8765`.

The User demo is synthetic CPU-only UI data and is labeled `DEMO` throughout
the interface. It must not be interpreted as a training result.

The Chinese dashboard lists experiments under `--runs-dir`; every data request
is scoped to the selected `run_id`, so metrics and rollout traces cannot mix
between experiments. The older `--run-dir <one-run>` mode remains supported.
The selected run is also stored in the page URL for refresh/share continuity.
When `--outputs-dir` is configured, the dashboard lists that experiment's
checkpoints. The compatibility download API exposes only `adapter_config.json`
and `adapter_model.safetensors`; optimizer and other training-state files are
not served. The main toolbar uses the validated full-model publish flow below.

## Full-model publish

The checkpoint toolbar can launch a server-side CPU job that validates an
adapter-only checkpoint, merges it into its exact full-SFT parent, and uploads
the resulting full model to ModelScope. The parent is selected only through a
SHA256-to-path allowlist. The adapter SHA256 and `lineage.json` must agree with
that parent before the worker starts. New repositories default to private.

Configure the server with explicit roots and a root-only token file:

```bash
install -d -m 700 /root/.config/grpo-monitor /root/grpo-modelscope-publish
install -m 600 /dev/null /root/.config/grpo-monitor/modelscope.token

python monitor/server.py \
  --runs-dir /data/GRPO/runs \
  --additional-runs-dir /path/to/new/monitor-runs \
  --checkpoint-outputs-dir /path/to/new/checkpoint-outputs \
  --modelscope-publish-root /root/grpo-modelscope-publish \
  --modelscope-publish-python /path/to/python-with-modelscope \
  --modelscope-token-file /root/.config/grpo-monitor/modelscope.token \
  --publish-base-model <MODEL_SAFETENSORS_SHA256>=/path/to/full-sft-parent
```

The token is read by the worker from the protected file. It is never accepted
from the browser, included in process arguments, returned by the API, or
written to job metadata. The UI requires the exact `owner/model-name`, an
explicit visibility choice, and a confirmation checkbox. Successful uploads
retain SHA256/status evidence and remove the temporary merged weight files.
Failed jobs preserve their working directory for diagnosis.

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

The server exposes `/api/runs`, `/api/manifest`, `/api/capabilities`,
`/api/metrics`, `/api/rollouts`, `/api/ranks`, `/api/traces`, `/api/probes`,
and the optional `/api/dsr/metrics`, `/api/dsr/steps`, and `/api/dsr/traces`.
JSONL endpoints accept `from_step`, `to_step`, `route`, and `rollout_id`;
`/api/ranks` additionally accepts `rank`. Missing DSR files return `[]`, and
the dashboard hides DSR-only controls for legacy runs.

## CPU tests

```bash
cd /data/GRPO/scripts
python -m monitor.test_monitor
python monitor/test_publish_merged_model.py
python test_fixed_probe.py
python test_formal_runner.py
python test_beam_prompt_cache.py
python test_correctness_v2.py
```

Fixed Probe evaluation is opt-in at the formal runner. Its four group IDs are
recorded in the manifest and excluded from the training sampler. It runs both
routes with production batch shapes and a fixed seed, restores training RNG,
and writes complete candidates and Beam SIDs separately from passive metrics.
