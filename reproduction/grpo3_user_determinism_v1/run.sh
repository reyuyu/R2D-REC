#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${GRPO3_DETERMINISM_ROOT:-/root/grpo3_user_determinism_20260906}"
SOURCE="$ROOT/source"
RUNS="$ROOT/runs"
CHECKPOINTS="$ROOT/checkpoints"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
RUNNER="$SOURCE/baselines/native_source_domain_r32_v3/grpo/user/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py"
COMPARE="$SOURCE/baselines/native_source_domain_r32_v3/grpo/user/scripts/compare_grpo3_user_determinism.py"
CONFIG="$SOURCE/baselines/native_source_domain_r32_v3/grpo/user/configs/grpo3_user_from_grpo2_step300_determinism_v1.json"
RUN_A="GRPO3-USER-DETERMINISM-SMOKE-A"
RUN_B="GRPO3-USER-DETERMINISM-SMOKE-B"

mkdir -p "$RUNS" "$CHECKPOINTS" "$STATE" "$LOGS"

write_state() {
  /usr/local/bin/python3 - "$STATE/status.json" "$1" "${2:-}" <<'PY'
import json
import sys
from pathlib import Path

path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path.write_text(json.dumps({"state": state, "detail": detail}, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  write_state STOPPED "command failed at line $1"
}
trap 'on_error "$LINENO"' ERR

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

write_state WAITING_FOR_GPUS "waiting for existing work to release GPU0-3"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do
  sleep 60
done

git -C "$SOURCE" diff --quiet
git -C "$SOURCE" diff --cached --quiet

run_smoke() {
  local run_id="$1"
  write_state "PREFLIGHT_${run_id}" "registered-data and parent contracts"
  /usr/local/bin/python3 "$RUNNER" \
    --config "$CONFIG" \
    --run-id "$run_id" \
    --smoke-prompts 5 \
    --determinism-evidence \
    --output-root "$RUNS" \
    --checkpoint-root "$CHECKPOINTS" \
    >"$LOGS/${run_id}.preflight.log" 2>&1
  write_state "RUNNING_${run_id}" "independent 0-to-5 optimizer-step smoke"
  /usr/local/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$RUNNER" \
    --config "$CONFIG" \
    --run-id "$run_id" \
    --execute \
    --smoke-prompts 5 \
    --determinism-evidence \
    --output-root "$RUNS" \
    --checkpoint-root "$CHECKPOINTS" \
    >"$LOGS/${run_id}.log" 2>&1
}

run_smoke "$RUN_A"
run_smoke "$RUN_B"

write_state COMPARING "step evidence and final adapter bytes"
/usr/local/bin/python3 "$COMPARE" \
  --run-a "$RUNS/$RUN_A" \
  --run-b "$RUNS/$RUN_B" \
  --output "$ROOT/comparison.json" \
  >"$LOGS/comparison.log" 2>&1

write_state COMPLETE "determinism verification finished; formal training was not started"
trap - ERR
