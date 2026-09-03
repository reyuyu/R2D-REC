#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STABLE_ROOT EPOCH2-RUN-ID" >&2
  exit 2
fi

STABLE_ROOT="$1"
RUN_ID="$2"
SUPPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${STABLE_ROOT}/epoch2_runs/${RUN_ID}"

"/data/venvs/llamafactory-01398eb-liger081/bin/python" \
  "${SUPPORT_DIR}/prepare_stable_epoch2.py" verify \
  --root "${STABLE_ROOT}" --run-id "${RUN_ID}" >/dev/null

RUNTIME_SOURCE="$(/data/venvs/llamafactory-01398eb-liger081/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"])' "${RUN_DIR}/manifest.json")"
CONFIG_PATH="${RUN_DIR}/training_config.yaml"
CHECKPOINT_PATH="$(/data/venvs/llamafactory-01398eb-liger081/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_checkpoint"])' "${RUN_DIR}/manifest.json")"
EXPECTED_SHA="${RUN_DIR}/expected_checkpoint_sha256.json"

if [[ -e "${RUN_DIR}/output/checkpoint-1106" ]]; then
  echo "refusing to reuse completed epoch-2 run: ${RUN_ID}" >&2
  exit 3
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "all four GPUs must be free before epoch-2 launch" >&2
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
export BATA_STABLE_EPOCH2="1"
export BATA_STABLE_LABEL="${RUN_ID}"
export BATA_STABLE_STOP_STEP="1106"
export BATA_STABLE_CHECKPOINT="${CHECKPOINT_PATH}"
export BATA_STABLE_EXPECTED_CHECKPOINT_SHA="${EXPECTED_SHA}"
export BATA_STABLE_EVIDENCE_DIR="${RUN_DIR}/evidence"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export FLASH_ATTENTION_DETERMINISTIC="1"

unset BATA_STABLE553_V0
unset BATA_REPLAY_EVIDENCE_DIR BATA_REPLAY_HEAVY_STEPS BATA_REPLAY_CONFIRM_FA_DETERMINISTIC
unset BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR
unset BATA_REPLAY_BATCH_FINGERPRINT_MODE BATA_REPLAY_BATCH_CONTRACT_JSON

exec bash "${RUNTIME_SOURCE}/scripts/run_native_source_domain_r32_v3.sh"
