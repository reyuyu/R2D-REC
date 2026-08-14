#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
RUN_ID="${RUN_ID:-ALPHA-MONITOR-ONLY-R32-2E-GC04-4GPU-$(date +%Y%m%d-%H%M%S)}"
SOURCE_CONFIG="${BASELINE_ROOT}/config/train_alpha_jiankong_monitor_only_4gpu_gc04_2epoch.yaml"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${OUTPUT_DIR}/metadata/training_config.yaml"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}/metadata" "${LOG_DIR}"
cp "${SOURCE_CONFIG}" "${CONFIG_PATH}"
sed -i "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${CONFIG_PATH}"
cp /data/lf_data_versions/alltrain/alpha-jiankong/manifest.json "${OUTPUT_DIR}/metadata/dataset_manifest.json"

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE=4 \
  ALLOW_EXTRA_ARGS=1 \
  NATIVE_GC_FRACTION=0.4 \
  GLOBAL_ITEM_WEIGHT=8 \
  MATERIAL_DOMAIN_MANIFEST=/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json \
  CONFIG_PATH="${CONFIG_PATH}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
