#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=/data/baselines/native_source_domain_r32_v3
CONFIG_PATH=${1:?usage: run_rec_pu_phase4_smoke.sh CONFIG_PATH RUN_ID DEBUG_FLAG}
RUN_ID=${2:?usage: run_rec_pu_phase4_smoke.sh CONFIG_PATH RUN_ID DEBUG_FLAG}
DEBUG_FLAG=${3:-0}
LOG_DIR=/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}
mkdir -p "${LOG_DIR}"

(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 1
  done
) > "${LOG_DIR}/gpu.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  NATIVE_GC_FRACTION=0.4 \
  SOURCE_WEIGHT_SKIP_FINAL_SAVE=1 \
  REC_PU_DEBUG_PROBE="${DEBUG_FLAG}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
