#!/usr/bin/env bash
set -euo pipefail

cd /data/GRPO_USER

RUN_ID="${GRPO_RUN_ID:-GR-USER-PILOT300-CONT-$(date +%Y%m%d-%H%M%S)}"
GIT_COMMIT="${PILOT_GIT_COMMIT:?PILOT_GIT_COMMIT is required}"
CONFIG="/data/GRPO_USER/config/pilot300_cont_v1.json"
RUNNER="/data/GRPO_USER/scripts/run_user_pilot300_cont.py"
BACKFILL_RUNNER="/data/GRPO_USER/scripts/backfill_user_fixed_probe.py"
AUDIT_RUNNER="/data/GRPO_USER/scripts/audit_user_pilot300_cont.py"
RESUME_PREFLIGHT="/data/GRPO_USER/scripts/preflight_user_pilot300_resume_gpu.py"
BASE="/data/models/onereason-8b-pretrain-competition"
PARENT="/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
RESUME="/data/GRPO_USER/runs/GR-USER-PILOT150-20260820-004130/pilot150-final"
PROBE="/data/GRPO_USER/data/gr_user_v1/probe_v1.jsonl"
MONITOR_ROOT="/data/GRPO/runs"
PROBE_BACKFILL="/data/GRPO_USER/staging/${RUN_ID}-probes-step0-step20.jsonl"
PREFLIGHT_OUTPUT="/data/GRPO_USER/staging/${RUN_ID}-preflight.json"

check_gpus_idle() {
  if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 4 ]]; then
    echo "Pilot300 continuation requires exactly four GPUs" >&2
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

for required in "$CONFIG" "$RUNNER" "$BACKFILL_RUNNER" "$AUDIT_RUNNER" "$RESUME_PREFLIGHT" "$PARENT/adapter_model.safetensors" \
  "$RESUME/adapter_model.safetensors" "$RESUME/optimizer.pt" "$RESUME/metadata.json" "$PROBE"; do
  if [[ ! -f "$required" ]]; then
    echo "Required Pilot300 artifact missing: $required" >&2
    exit 1
  fi
done
if [[ -e "/data/GRPO_USER/runs/$RUN_ID" || -e "$MONITOR_ROOT/$RUN_ID" || -e "$PROBE_BACKFILL" || -e "$PREFLIGHT_OUTPUT" ]]; then
  echo "RUN_ID or staging output already exists: $RUN_ID" >&2
  exit 1
fi
if ! curl --fail --silent "http://127.0.0.1:8878/api/health" >/dev/null; then
  echo "User GRPO monitor on port 8878 is unavailable" >&2
  exit 1
fi
if [[ "$(sha256sum "$PROBE" | awk '{print $1}')" != "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61" ]]; then
  echo "Frozen fixed-probe SHA mismatch" >&2
  exit 1
fi

check_gpus_idle
mkdir -p /data/GRPO_USER/logs /data/GRPO_USER/staging
LOG_PATH="/data/GRPO_USER/logs/${RUN_ID}.log"

CUDA_VISIBLE_DEVICES='' /data/venvs/llamafactory-01398eb-liger081/bin/python \
  "$AUDIT_RUNNER" --config "$CONFIG" --output "$PREFLIGHT_OUTPUT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/GRPO_USER/scripts:/data/GRPO/scripts
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

/data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "$RESUME_PREFLIGHT" --base-model "$BASE" --checkpoint "$RESUME" --expected-step 20

check_gpus_idle
/data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "$BACKFILL_RUNNER" \
  --base-model "$BASE" --adapter "$PARENT" --probe "$PROBE" \
  --output "$PROBE_BACKFILL" --step 0 --reason parent_baseline --seed 20260820

check_gpus_idle
/data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "$BACKFILL_RUNNER" \
  --base-model "$BASE" --adapter "$RESUME" --probe "$PROBE" \
  --output "$PROBE_BACKFILL" --step 20 --reason pilot150_resume --seed 20260820 --append

check_gpus_idle
export GRPO_MONITOR=1
export GRPO_MONITOR_DIR="$MONITOR_ROOT"
export GRPO_MONITOR_STEP_EVERY=1
export GRPO_MONITOR_ROLLOUT_EVERY=5
export GRPO_TRACE_EVERY=10
export GRPO_RUN_ID="$RUN_ID"
export GRPO_PROBE_BACKFILL_PATH="$PROBE_BACKFILL"

echo "RUN_ID=$RUN_ID"
echo "LOG_PATH=$LOG_PATH"
echo "PROBE_BACKFILL=$PROBE_BACKFILL"

/data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "$RUNNER" \
  --run-id "$RUN_ID" \
  --git-commit "$GIT_COMMIT" \
  --config "$CONFIG" \
  --result-output /data/GRPO_USER/results/pilot300_cont_v1_summary.json \
  --docs-output /data/GRPO_USER/docs/pilot300_cont_v1.md \
  2>&1 | tee "$LOG_PATH"
