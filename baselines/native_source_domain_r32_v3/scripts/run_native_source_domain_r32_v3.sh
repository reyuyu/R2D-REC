#!/usr/bin/env bash
set -euo pipefail

REFERENCE_ROOT="${REFERENCE_ROOT:-/data/reference/llamafactory-01398eb}"
BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
VENV_ROOT="${VENV_ROOT:-/data/venvs/llamafactory-01398eb-liger081}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CONFIG_PATH="${CONFIG_PATH:-${BASELINE_ROOT}/config/train_native_source_domain_r32_v3_2epoch.yaml}"

export PYTHONPATH="${REFERENCE_ROOT}/src"
export MATERIAL_DOMAIN_MANIFEST="${BASELINE_ROOT}/dataset/manifest.json"
export GLOBAL_ITEM_WEIGHT="${GLOBAL_ITEM_WEIGHT:-8}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${VENV_ROOT}/bin/python" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  "${BASELINE_ROOT}/scripts/train_native_source_domain_r32_v3.py" \
  "${CONFIG_PATH}"
