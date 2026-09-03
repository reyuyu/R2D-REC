#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 STABLE_ROOT SEED-RUN-ID initial|resume" >&2
  exit 2
fi

STABLE_ROOT="$1"
RUN_ID="$2"
PHASE="$3"
SUPPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${STABLE_ROOT}/seed_runs/${RUN_ID}"
PREPARE="${SUPPORT_DIR}/prepare_stable_seed.py"
PYTHON="/data/venvs/llamafactory-01398eb-liger081/bin/python"

case "${PHASE}" in
  initial)
    "${PYTHON}" "${PREPARE}" verify-initial --root "${STABLE_ROOT}" --run-id "${RUN_ID}" >/dev/null
    CONFIG_KEY="fresh_config"
    START_STEP="0"
    STOP_KEY="initial_stop_step"
    EVIDENCE_DIR="${RUN_DIR}/evidence_initial"
    ;;
  resume)
    "${PYTHON}" "${PREPARE}" prepare-resume --root "${STABLE_ROOT}" --run-id "${RUN_ID}" >/dev/null
    CONFIG_KEY="resume_config"
    START_STEP="553"
    STOP_KEY="final_stop_step"
    EVIDENCE_DIR="${RUN_DIR}/evidence_resume"
    ;;
  *) echo "invalid seed phase: ${PHASE}" >&2; exit 2 ;;
esac

RUNTIME_SOURCE="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"])' "${RUN_DIR}/manifest.json")"
CONFIG_PATH="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${RUN_DIR}/manifest.json" "${CONFIG_KEY}")"
STOP_STEP="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${RUN_DIR}/manifest.json" "${STOP_KEY}")"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "all four GPUs must be free before seed launch" >&2
  exit 4
fi
GPU_ROWS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
if [[ "$(printf '%s\n' "${GPU_ROWS}" | wc -l)" -ne 4 ]] || \
   ! printf '%s\n' "${GPU_ROWS}" | awk -F, '{gsub(/ /,"",$2); if (($2 + 0) >= 1024) exit 1}'; then
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
export BATA_STABLE_SEED="1"
export BATA_STABLE_LABEL="${RUN_ID}"
export BATA_STABLE_START_STEP="${START_STEP}"
export BATA_STABLE_STOP_STEP="${STOP_STEP}"
export BATA_STABLE_EVIDENCE_DIR="${EVIDENCE_DIR}"
export BATA_STABLE_CONFIG_SHA256
BATA_STABLE_CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
export BATA_STABLE_BASE_CONTRACT_SHA256
export BATA_STABLE_SOURCE_CONTRACT_SHA256
BATA_STABLE_BASE_CONTRACT_SHA256="$(awk '$2 == "base_contract_sha256" {print $1}' "${STABLE_ROOT}/contract.env")"
BATA_STABLE_SOURCE_CONTRACT_SHA256="$(awk '$2 == "source_contract_sha256" {print $1}' "${STABLE_ROOT}/contract.env")"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export FLASH_ATTENTION_DETERMINISTIC="1"

unset BATA_STABLE553_V0 BATA_STABLE_EPOCH2
unset BATA_REPLAY_EVIDENCE_DIR BATA_REPLAY_HEAVY_STEPS BATA_REPLAY_CONFIRM_FA_DETERMINISTIC
unset BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR
unset BATA_REPLAY_BATCH_FINGERPRINT_MODE BATA_REPLAY_BATCH_CONTRACT_JSON

if [[ "${PHASE}" == "resume" ]]; then
  export BATA_STABLE_SEED_RESUME="1"
  export BATA_STABLE_START_MODE="checkpoint-553"
  export BATA_STABLE_CHECKPOINT="${RUN_DIR}/output/checkpoint-553"
  export BATA_STABLE_EXPECTED_CHECKPOINT_SHA="${RUN_DIR}/checkpoint_553_sha256.json"
else
  export BATA_STABLE_START_MODE="base"
  unset BATA_STABLE_SEED_RESUME BATA_STABLE_CHECKPOINT BATA_STABLE_EXPECTED_CHECKPOINT_SHA
fi

exec bash "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"
