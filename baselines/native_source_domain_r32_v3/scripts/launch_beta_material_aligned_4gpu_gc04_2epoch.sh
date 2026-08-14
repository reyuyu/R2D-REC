#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
RUN_ID="${RUN_ID:-BETA-MATERIAL-ALIGNED-R32-2E-GC04-4GPU-$(date +%Y%m%d-%H%M%S)}"
SOURCE_CONFIG="${BASELINE_ROOT}/config/train_beta_material_aligned_4gpu_gc04_2epoch.yaml"
DATASET_ROOT="/data/lf_data_versions/alltrain/BETA_material_aligned_v1"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${OUTPUT_DIR}/metadata/training_config.yaml"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/metadata"
cp "${SOURCE_CONFIG}" "${CONFIG_PATH}"
sed -i "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${CONFIG_PATH}"
cp "${DATASET_ROOT}/manifest.json" "${OUTPUT_DIR}/metadata/dataset_manifest.json"

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE=4 \
  NATIVE_GC_FRACTION=0.4 \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" \
  CONFIG_PATH="${CONFIG_PATH}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
