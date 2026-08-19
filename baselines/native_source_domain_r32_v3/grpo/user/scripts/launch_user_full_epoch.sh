#!/usr/bin/env bash
set -euo pipefail

cd /data/GRPO_USER

RUN_ID="${GRPO_RUN_ID:-GR-USER-FULL-EPOCH-$(date +%Y%m%d-%H%M%S)}"
GIT_COMMIT="${FULL_EPOCH_GIT_COMMIT:?FULL_EPOCH_GIT_COMMIT is required}"
PYTHON=/data/venvs/llamafactory-01398eb-liger081/bin/python
CONFIG=/data/GRPO_USER/config/full_epoch_v1.json
RUNNER=/data/GRPO_USER/scripts/run_user_full_epoch.py
STATIC_PREFLIGHT=/data/GRPO_USER/scripts/preflight_user_full_epoch.py
RESUME_PREFLIGHT=/data/GRPO_USER/scripts/preflight_user_pilot300_resume_gpu.py
BACKFILL_RUNNER=/data/GRPO_USER/scripts/backfill_user_fixed_probe.py
BASE=/data/models/onereason-8b-pretrain-competition
RESUME=/data/GRPO_USER/runs/GR-USER-PILOT300-CONT-20260820-031716/pilot300-final
PROBE=/data/GRPO_USER/data/gr_user_v1/probe_light_v1.jsonl
PROBE_MANIFEST=/data/GRPO_USER/data/gr_user_v1/probe_light_v1_manifest.json
PROBE_SHA=3e383c81fc8662162039f2dc827887656daeda9382b519bbb6f6d9ba4724c2f3
MONITOR_ROOT=/data/GRPO/runs
PREFLIGHT_OUTPUT=/data/GRPO_USER/staging/${RUN_ID}-preflight.json
PROBE_BACKFILL=/data/GRPO_USER/staging/${RUN_ID}-probe-step40.jsonl
LOG_PATH=/data/GRPO_USER/logs/${RUN_ID}.log

check_gpus_idle() {
  if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 4 ]]; then
    echo "Formal User epoch requires exactly four GPUs" >&2
    exit 1
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "GPU compute process detected; refusing to continue" >&2
    exit 1
  fi
  while IFS=, read -r index used; do
    if [[ "${used// /}" -gt 1024 ]]; then
      echo "GPU ${index// /} is not idle" >&2
      exit 1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

for required in "$CONFIG" "$RUNNER" "$STATIC_PREFLIGHT" "$RESUME_PREFLIGHT" "$BACKFILL_RUNNER" \
  "$RESUME/adapter_model.safetensors" "$RESUME/optimizer.pt" "$RESUME/metadata.json" \
  "$PROBE" "$PROBE_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "Required formal-epoch artifact missing: $required" >&2
    exit 1
  fi
done
for rank in 0 1 2 3; do
  if [[ ! -f "$RESUME/rng_rank${rank}.pt" ]]; then
    echo "Missing resume RNG state for rank ${rank}" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$PROBE" | awk '{print $1}')" != "$PROBE_SHA" ]]; then
  echo "Frozen light-probe SHA mismatch" >&2
  exit 1
fi
if [[ -e "/data/GRPO_USER/runs/$RUN_ID" || -e "$MONITOR_ROOT/$RUN_ID" || \
      -e "$PREFLIGHT_OUTPUT" || -e "$PROBE_BACKFILL" ]]; then
  echo "RUN_ID or staging output already exists: $RUN_ID" >&2
  exit 1
fi
if ! curl --fail --silent http://127.0.0.1:8878/api/health >/dev/null; then
  echo "User GRPO monitor on port 8878 is unavailable" >&2
  exit 1
fi

mkdir -p /data/GRPO_USER/logs /data/GRPO_USER/staging
check_gpus_idle
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/data/GRPO_USER/scripts:/data/GRPO/scripts \
  "$PYTHON" "$STATIC_PREFLIGHT" --config "$CONFIG" --output "$PREFLIGHT_OUTPUT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/GRPO_USER/scripts:/data/GRPO/scripts
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

check_gpus_idle
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$RESUME_PREFLIGHT" --base-model "$BASE" --checkpoint "$RESUME" --expected-step 40

check_gpus_idle
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$BACKFILL_RUNNER" \
  --base-model "$BASE" --adapter "$RESUME" --probe "$PROBE" \
  --output "$PROBE_BACKFILL" --step 40 --reason pilot300_resume \
  --seed 20260820 --expected-sha "$PROBE_SHA"

check_gpus_idle
export GRPO_MONITOR=1
export GRPO_MONITOR_DIR="$MONITOR_ROOT"
export GRPO_MONITOR_STEP_EVERY=1
export GRPO_MONITOR_ROLLOUT_EVERY=5
export GRPO_TRACE_EVERY=10
export GRPO_RUN_ID="$RUN_ID"

echo "RUN_ID=$RUN_ID"
echo "LOG_PATH=$LOG_PATH"
echo "PROBE_BACKFILL=$PROBE_BACKFILL"

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$RUNNER" \
  --run-id "$RUN_ID" \
  --git-commit "$GIT_COMMIT" \
  --config "$CONFIG" \
  --preflight "$PREFLIGHT_OUTPUT" \
  --probe-backfill "$PROBE_BACKFILL" \
  --result-output /data/GRPO_USER/results/full_epoch_v1_summary.json \
  --docs-output /data/GRPO_USER/docs/full_epoch_v1.md \
  2>&1 | tee "$LOG_PATH"
