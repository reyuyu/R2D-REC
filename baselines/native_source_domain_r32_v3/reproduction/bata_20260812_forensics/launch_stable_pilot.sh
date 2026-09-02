#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT STABLE560-A|STABLE560-B|STABLE560-C" >&2
  exit 2
fi

RUN_ROOT="$1"
LABEL="$2"
SUPPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_SOURCE="${RUN_ROOT}/source_runtime"
RUN_DIR="${RUN_ROOT}/runs/${LABEL}"
CONFIG_PATH="${RUN_DIR}/training_config.yaml"
CHECKPOINT_PATH="${RUN_DIR}/output/checkpoint-553"
EXPECTED_SHA="${RUN_ROOT}/expected_checkpoint_sha256.json"

case "${LABEL}" in
  STABLE560-A|STABLE560-B|STABLE560-C) ;;
  *) echo "invalid stable pilot label: ${LABEL}" >&2; exit 2 ;;
esac

"/data/venvs/llamafactory-01398eb-liger081/bin/python" \
  "${SUPPORT_DIR}/prepare_stable_pilot.py" verify --root "${RUN_ROOT}" --label "${LABEL}" >/dev/null

if [[ -e "${RUN_DIR}/output/checkpoint-560" ]] || [[ -e "${RUN_DIR}/completed.json" ]]; then
  echo "refusing to reuse completed stable pilot: ${RUN_DIR}" >&2
  exit 3
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "all four GPUs must be free before stable pilot launch" >&2
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
export BATA_STABLE_LABEL="${LABEL}"
export BATA_STABLE_STOP_STEP="560"
export BATA_STABLE_CHECKPOINT="${CHECKPOINT_PATH}"
export BATA_STABLE_EXPECTED_CHECKPOINT_SHA="${EXPECTED_SHA}"
export BATA_STABLE_EVIDENCE_DIR="${RUN_DIR}/evidence"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export FLASH_ATTENTION_DETERMINISTIC="1"

# Fail closed if a caller accidentally carries heavy forensic controls forward.
unset BATA_REPLAY_EVIDENCE_DIR BATA_REPLAY_HEAVY_STEPS BATA_REPLAY_CONFIRM_FA_DETERMINISTIC
unset BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR
unset BATA_REPLAY_BATCH_FINGERPRINT_MODE BATA_REPLAY_BATCH_CONTRACT_JSON

exec bash "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"
