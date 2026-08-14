#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/data/reference/llamafactory-01398eb}"
VENV_ROOT="${VENV_ROOT:-/data/venvs/llamafactory-01398eb-liger081}"
RUN_ID="${RUN_ID:-NSD-R32-V3-2E-GC04-2GPU-20260810}"
LOG_DIR="/data/logs/baselines/native_source_domain_r32_v3/${RUN_ID}"
CONFIG_PATH="${BASELINE_ROOT}/config/train_native_source_domain_r32_v3_2gpu_gc04_2epoch.yaml"
OUTPUT_DIR="/data/outputs/baselines/native_source_domain_r32_v3/${RUN_ID}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/metadata"
cp "${CONFIG_PATH}" "${OUTPUT_DIR}/metadata/training_config.yaml"
{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'native_gc_fraction=0.4\n'
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-2,3}"
  printf 'reference_commit='
  git -C "${REFERENCE_ROOT}" rev-parse HEAD
  "${VENV_ROOT}/bin/python" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "${OUTPUT_DIR}/metadata/runtime_environment.txt"
exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" \
  NPROC_PER_NODE=2 \
  NATIVE_GC_FRACTION=0.4 \
  CONFIG_PATH="${CONFIG_PATH}" \
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" \
  GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}" \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
  > "${LOG_DIR}/train.log" 2>&1
