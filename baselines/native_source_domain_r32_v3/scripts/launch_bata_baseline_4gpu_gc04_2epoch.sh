#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
RUN_ID="${RUN_ID:-BATA-BASELINE-R32-2E-GC04-4GPU-$(date +%Y%m%d-%H%M%S)}"
DATASET_ROOT="/data/lf_data_versions/alltrain/bata_baseline_v1"
SOURCE_CONFIG="${BASELINE_ROOT}/config/train_bata_baseline_4gpu_gc04_2epoch.yaml"
VALIDATOR="${BASELINE_ROOT}/scripts/validate_bata_baseline.py"
TRAINER="${BASELINE_ROOT}/scripts/train_native_source_domain_r32_v3.py"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${OUTPUT_DIR}/metadata/training_config.yaml"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/metadata"
cp "${SOURCE_CONFIG}" "${CONFIG_PATH}"
sed -i "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${CONFIG_PATH}"
sed -i "s|^logging_dir:.*|logging_dir: ${OUTPUT_DIR}/tb|" "${CONFIG_PATH}"
cp "${DATASET_ROOT}/manifest.json" "${OUTPUT_DIR}/metadata/dataset_manifest.json"

PYTHONPATH="/data/reference/llamafactory-01398eb/src" \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" \
  GLOBAL_ITEM_WEIGHT=8 \
  /data/venvs/llamafactory-01398eb-liger081/bin/python "${VALIDATOR}" \
  --manifest "${DATASET_ROOT}/manifest.json" \
  --config "${CONFIG_PATH}" \
  --launcher "${TRAINER}" \
  > "${LOG_DIR}/preflight.log" 2>&1

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE=4 \
  NATIVE_GC_FRACTION=0.4 \
  GLOBAL_ITEM_WEIGHT=8 \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" \
  CONFIG_PATH="${CONFIG_PATH}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
