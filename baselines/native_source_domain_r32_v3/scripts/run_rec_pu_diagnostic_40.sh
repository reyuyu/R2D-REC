#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=/data/baselines/native_source_domain_r32_v3
RUN_ID=REC-PU-BETA-MATERIAL-ALIGNED-R32-B005-DIAG40-$(date +%Y%m%d-%H%M%S)
DATASET_ROOT=/data/lf_data_versions/alltrain/BETA_material_aligned_v1
CONFIG=${1:?usage: run_rec_pu_diagnostic_40.sh CONFIG_PATH}
LOG_DIR=/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}
mkdir -p "${LOG_DIR}"

PYTHONPATH=/data/reference/llamafactory-01398eb/src \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" GLOBAL_ITEM_WEIGHT=8 \
  /data/venvs/llamafactory-01398eb-liger081/bin/python \
  "${BASELINE_ROOT}/scripts/validate_beta_material_aligned_sid8.py" \
  --manifest "${DATASET_ROOT}/manifest.json" --config "${CONFIG}" \
  --launcher "${BASELINE_ROOT}/scripts/train_native_source_domain_r32_v3.py" \
  >"${LOG_DIR}/preflight.log" 2>&1

(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 1
  done
) >"${LOG_DIR}/gpu.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

env CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 NATIVE_GC_FRACTION=0.4 \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" GLOBAL_ITEM_WEIGHT=8 \
  REC_PU_DEBUG_PROBE=0 REC_PU_DIAGNOSTICS=1 REC_PU_GRAD_DIAGNOSTICS=1 \
  SOURCE_WEIGHT_SKIP_FINAL_SAVE=1 CONFIG_PATH="${CONFIG}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  >"${LOG_DIR}/train.log" 2>&1
