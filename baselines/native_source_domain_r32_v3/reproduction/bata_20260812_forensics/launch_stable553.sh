#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT STABLE553-A|STABLE553-B" >&2
  exit 2
fi

RUN_ROOT="$1"
LABEL="$2"
SUPPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_SOURCE="${RUN_ROOT}/source_runtime"
RUN_DIR="${RUN_ROOT}/runs/${LABEL}"
CONFIG_PATH="${RUN_DIR}/training_config.yaml"

case "${LABEL}" in
  STABLE553-A|STABLE553-B) ;;
  *) echo "invalid stable553 label: ${LABEL}" >&2; exit 2 ;;
esac

"/data/venvs/llamafactory-01398eb-liger081/bin/python" \
  "${SUPPORT_DIR}/prepare_stable553.py" verify --root "${RUN_ROOT}" --label "${LABEL}" >/dev/null

if [[ -e "${RUN_DIR}/output" ]] || [[ -e "${RUN_DIR}/completed.json" ]]; then
  echo "refusing to reuse stable553 output: ${RUN_DIR}" >&2
  exit 3
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "all four GPUs must be free before stable553 launch" >&2
  exit 4
fi
GPU_ROWS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
if [[ "$(printf '%s\n' "${GPU_ROWS}" | wc -l)" -ne 4 ]] || \
   ! printf '%s\n' "${GPU_ROWS}" | awk -F, '{gsub(/ /,"",$2); if ($2 >= 1024) exit 1}'; then
  echo "all four GPUs must report memory.used < 1024 MiB" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE="4"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_IB_DISABLE="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NATIVE_GC_FRACTION="0.4"
export GLOBAL_ITEM_WEIGHT="8"
export MATERIAL_DOMAIN_MANIFEST="/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json"
export BASELINE_ROOT="${RUNTIME_SOURCE}"
export CONFIG_PATH

export BATA_STABLE_V0="1"
export BATA_STABLE553_V0="1"
export BATA_STABLE_START_MODE="base"
export BATA_STABLE_LABEL="${LABEL}"
export BATA_STABLE_STOP_STEP="553"
export BATA_STABLE_EVIDENCE_DIR="${RUN_DIR}/evidence"
export BATA_STABLE_BASE_CONTRACT_SHA256
export BATA_STABLE_CONFIG_SHA256
export BATA_STABLE_SOURCE_CONTRACT_SHA256
BATA_STABLE_BASE_CONTRACT_SHA256="$(awk '$2 == "base_contract_sha256" {print $1}' "${RUN_ROOT}/contract.env")"
BATA_STABLE_CONFIG_SHA256="$(awk -v key="${LABEL}_config_sha256" '$2 == key {print $1}' "${RUN_ROOT}/contract.env")"
BATA_STABLE_SOURCE_CONTRACT_SHA256="$(awk '$2 == "source_contract_sha256" {print $1}' "${RUN_ROOT}/contract.env")"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export FLASH_ATTENTION_DETERMINISTIC="1"

# Fail closed if a caller carries forensic or resume controls into this run.
unset BATA_STABLE_CHECKPOINT BATA_STABLE_EXPECTED_CHECKPOINT_SHA
unset BATA_REPLAY_EVIDENCE_DIR BATA_REPLAY_HEAVY_STEPS BATA_REPLAY_CONFIRM_FA_DETERMINISTIC
unset BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR
unset BATA_REPLAY_BATCH_FINGERPRINT_MODE BATA_REPLAY_BATCH_CONTRACT_JSON

exec bash "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"
