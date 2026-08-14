#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="/data/baselines/native_source_domain_r32_v3"
DATASET_ROOT="/data/lf_data_versions/alltrain/BETA_material_aligned_v1"
CONFIG_PATH="${BASELINE_ROOT}/config/train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015_coremetrics_smoke5.yaml"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/PACKRATIO-20452015-4GPU-COREMETRICS-SMOKE5"

mkdir -p "${LOG_DIR}"

exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  NATIVE_GC_FRACTION=0.4 \
  GLOBAL_ITEM_WEIGHT=8 \
  MATERIAL_DOMAIN_MANIFEST="${DATASET_ROOT}/manifest.json" \
  REC_PU_DIAGNOSTICS=0 \
  REC_PU_GRAD_DIAGNOSTICS=0 \
  REC_PU_DEBUG_PROBE=0 \
  PACK_RATIO_STOP_AFTER_STEPS=5 \
  CONFIG_PATH="${CONFIG_PATH}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
