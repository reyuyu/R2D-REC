#!/usr/bin/env bash
set -euo pipefail

: "${FULLBASE_MODEL:?set FULLBASE_MODEL to the immutable full-SFT directory}"
: "${GRPO_DATA:?set GRPO_DATA to the frozen recommendation GRPO JSONL}"
: "${RUN_ROOT:?set RUN_ROOT to a new output root}"
: "${MONITOR_ROOT:?set MONITOR_ROOT to a new monitor root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PACKAGE_DIR}/config/formal_500.json"
RUN_ID="${RUN_ID:-GRPO1-REC-BILATERAL-FULLBASE-CONSERVATIVE-R32-LR5E7-500}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONHASHSEED=20260816
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_conservative_grpo.py" \
  --config "${CONFIG}" --base-model "${FULLBASE_MODEL}" --data-path "${GRPO_DATA}" \
  --output-root "${RUN_ROOT}" --monitor-root "${MONITOR_ROOT}" --run-id "${RUN_ID}"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${SCRIPT_DIR}/run_conservative_grpo.py" \
  --config "${CONFIG}" --base-model "${FULLBASE_MODEL}" --data-path "${GRPO_DATA}" \
  --output-root "${RUN_ROOT}" --monitor-root "${MONITOR_ROOT}" --run-id "${RUN_ID}" --execute
