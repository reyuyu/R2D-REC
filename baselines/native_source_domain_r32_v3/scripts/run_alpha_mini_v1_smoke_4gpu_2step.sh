#!/usr/bin/env bash
# Minimal 4-GPU compatibility smoke; never a formal training entrypoint.
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
RUN_ID="${RUN_ID:-ALPHA-MINI-V1-SMOKE-2STEP-$(date +%Y%m%d-%H%M%S)}"
SOURCE_CONFIG="${BASELINE_ROOT}/config/smoke_alpha_mini_v1_4gpu_2step.yaml"
MATERIAL_MANIFEST="/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json"
VENV_ROOT="${VENV_ROOT:-/data/venvs/llamafactory-01398eb-liger081}"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${OUTPUT_DIR}/metadata/training_config.yaml"

mkdir -p "${OUTPUT_DIR}/metadata" "${LOG_DIR}"
"${VENV_ROOT}/bin/python" - "${SOURCE_CONFIG}" "${CONFIG_PATH}" "${OUTPUT_DIR}" "${LOG_DIR}" <<'PY'
import sys
from pathlib import Path
import yaml
source, destination, output_dir, log_dir = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["output_dir"] = str(output_dir)
config["alpha_validation_metrics_path"] = str(log_dir / "alpha_validation_metrics.jsonl")
destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE=4 \
  ALLOW_EXTRA_ARGS=1 \
  PACK_RATIO_STOP_AFTER_STEPS=2 \
  SOURCE_WEIGHT_SKIP_FINAL_SAVE=1 \
  NATIVE_GC_FRACTION=0.4 \
  GLOBAL_ITEM_WEIGHT=8 \
  MATERIAL_DOMAIN_MANIFEST="${MATERIAL_MANIFEST}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" \
  GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}" \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
