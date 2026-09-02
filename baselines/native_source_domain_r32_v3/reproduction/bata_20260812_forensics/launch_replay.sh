#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 RUN_ROOT LABEL [--deterministic]" >&2
  exit 2
}

[[ $# -eq 2 || $# -eq 3 ]] || usage
RUN_ROOT="$1"
LABEL="$2"
MODE="${3:-}"
[[ -z "${MODE}" || "${MODE}" == "--deterministic" ]] || usage
FORENSICS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNTIME_SOURCE="${BATA_REPLAY_RUNTIME_SOURCE:-${RUN_ROOT}/source_runtime}"
RUN_DIR="${RUN_ROOT}/runs/${LABEL}"
CONFIG_PATH="${RUN_DIR}/replay_config.yaml"
CHECKPOINT_PATH="${RUN_DIR}/output/checkpoint-553"
EVIDENCE_DIR="${RUN_DIR}/evidence"
EXPECTED_SHA="${RUN_DIR}/expected_checkpoint_sha256.json"
MATERIAL_MANIFEST="${BATA_REPLAY_MATERIAL_MANIFEST:-/data/lf_data_versions/alltrain/bata_baseline_v1/manifest.json}"

for path in "${CONFIG_PATH}" "${EXPECTED_SHA}" "${MATERIAL_MANIFEST}" \
  "${CHECKPOINT_PATH}/adapter_model.safetensors" \
  "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"; do
  [[ -e "${path}" ]] || { echo "missing required replay input: ${path}" >&2; exit 3; }
done

if find "${EVIDENCE_DIR}" -maxdepth 1 -name 'rank*.jsonl' -print -quit | grep -q .; then
  echo "refusing to append to existing replay evidence: ${EVIDENCE_DIR}" >&2
  exit 4
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "all four GPUs must be free before replay launch" >&2
  exit 5
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
export MATERIAL_DOMAIN_MANIFEST="${MATERIAL_MANIFEST}"
export BASELINE_ROOT="${RUNTIME_SOURCE}"
export CONFIG_PATH
export BATA_REPLAY_LABEL="${LABEL}"
export BATA_REPLAY_CHECKPOINT="${CHECKPOINT_PATH}"
export BATA_REPLAY_EVIDENCE_DIR="${EVIDENCE_DIR}"
export BATA_REPLAY_EXPECTED_SHA_JSON="${EXPECTED_SHA}"
export BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD="1"
export BATA_REPLAY_STOP_AFTER_STEP="${BATA_REPLAY_STOP_AFTER_STEP:-554}"
export BATA_REPLAY_HEAVY_STEPS="${BATA_REPLAY_HEAVY_STEPS:-554}"
export BATA_REPLAY_BATCH_FINGERPRINT_MODE="${BATA_REPLAY_BATCH_FINGERPRINT_MODE:-frozen_step554_contract}"
export BATA_REPLAY_BATCH_CONTRACT_JSON="${BATA_REPLAY_BATCH_CONTRACT_JSON:-${FORENSICS_DIR}/frozen_step554_batch_contract.json}"

[[ -f "${BATA_REPLAY_BATCH_CONTRACT_JSON}" ]] || {
  echo "missing frozen step554 batch contract: ${BATA_REPLAY_BATCH_CONTRACT_JSON}" >&2
  exit 6
}

if [[ "${MODE}" == "--deterministic" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  export BATA_REPLAY_DETERMINISTIC="1"
else
  unset BATA_REPLAY_DETERMINISTIC
fi

exec bash "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"
