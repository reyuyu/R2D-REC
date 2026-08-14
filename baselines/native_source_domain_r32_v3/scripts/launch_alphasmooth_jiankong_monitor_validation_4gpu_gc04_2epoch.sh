#!/usr/bin/env bash
# Formal AlphaSmooth entrypoint. Do not invoke without explicit training approval.
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
RUN_ID="${RUN_ID:-ALPHASMOOTH-SIDLS05-R32-2E-GC04-4GPU-$(date +%Y%m%d-%H%M%S)}"
SOURCE_CONFIG="${BASELINE_ROOT}/config/train_alphasmooth_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${OUTPUT_DIR}/metadata/training_config.yaml"
VENV_ROOT="${VENV_ROOT:-/data/venvs/llamafactory-01398eb-liger081}"
MATERIAL_MANIFEST="/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json"

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
cp /data/lf_data_versions/alltrain/alpha-jiankong-split-v1/manifest.json "${OUTPUT_DIR}/metadata/split_manifest.json"
cp /data/lf_data_versions/alltrain/alpha-jiankong-split-v1/split_audit.json "${OUTPUT_DIR}/metadata/split_audit.json"
cp /data/lf_data_versions/alltrain/alpha-jiankong-split-v1/probe_v1/manifest.json "${OUTPUT_DIR}/metadata/probe_manifest.json"
cp "${MATERIAL_MANIFEST}" "${OUTPUT_DIR}/metadata/material_domain_manifest.json"

env GLOBAL_ITEM_WEIGHT=8 MATERIAL_DOMAIN_MANIFEST="${MATERIAL_MANIFEST}" \
  "${VENV_ROOT}/bin/python" "${BASELINE_ROOT}/scripts/preflight_alpha_jiankong_formal.py" \
  --config "${CONFIG_PATH}" --world-size 4 | tee "${LOG_DIR}/preflight.log"

if [[ "${ALPHA_FORMAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "ALPHASMOOTH_FORMAL_PREFLIGHT_ONLY_PASS run_id=${RUN_ID}"
  exit 0
fi

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE=4 \
  ALLOW_EXTRA_ARGS=1 \
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
