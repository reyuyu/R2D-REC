#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?set the existing canonical pipeline root}"
: "${GRPO2_DATA:?set the frozen Positive-A0 Think-only JSONL}"
: "${GRPO2_PROBE_DATA:?set the frozen recommendation retention JSONL}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="$(git -C "${PACKAGE_DIR}" rev-parse --show-toplevel)"
SFT_MODEL=/root/rec_fdr_v43_runs/REC-FDR-V43-STRICTDET-20260904-173024/work/output
GRPO1_CHECKPOINT=/root/grpo1_formal_500_20260905/outputs/GRPO1-REC-BILATERAL-FULLBASE-CONSERVATIVE-R32-LR5E7-500/checkpoint-500
PARENT="${ROOT}/grpo1_full_model_step500"
RUN_ID=grpo2_think_step500_parent_lr2e7_300step
RUN_ROOT="${ROOT}/${RUN_ID}"
MONITOR_ROOT="${ROOT}/monitor-formal"
CONFIG="${PACKAGE_DIR}/config/formal_300.json"
OUTSIDE_LOG="${ROOT}/logs/grpo2-formal-300.log"

printf 'RESOLVED_ROOT = %s\n' "${ROOT}"
test -d "${ROOT}"
test ! -e "${PARENT}"
test ! -e "${RUN_ROOT}"
test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"
test "$(sha256sum "${SFT_MODEL}/model.safetensors" | cut -d' ' -f1)" = "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"
test "$(sha256sum "${GRPO1_CHECKPOINT}/adapter_model.safetensors" | cut -d' ' -f1)" = "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"
test "$(sha256sum "${GRPO2_DATA}" | cut -d' ' -f1)" = "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"

FREE_DISK_GB=$(df -BG --output=avail "${ROOT}" | tail -1 | tr -dc '0-9')
EXPECTED_TOTAL_CHECKPOINT_GB=8
printf 'FREE_DISK_GB = %s\nEXPECTED_TOTAL_CHECKPOINT_GB = %s\n' "${FREE_DISK_GB}" "${EXPECTED_TOTAL_CHECKPOINT_GB}"
test "${FREE_DISK_GB}" -ge 100
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=20260816
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FLASH_ATTENTION_DETERMINISTIC=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export GRPO_MONITOR_DIR="${MONITOR_ROOT}"
export GRPO_GIT_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_test_parent.py" \
  --sft-model "${SFT_MODEL}" \
  --grpo1-adapter "${GRPO1_CHECKPOINT}" \
  --source-grpo1-step 500 \
  --canonical-for-this-run \
  --output "${PARENT}" \
  --data-path "${GRPO2_DATA}" \
  --device cuda:0

PARENT_HASH_BEFORE=$("${PYTHON_BIN}" - "${PARENT}/full_model_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["canonical_model_identity"])
PY
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/step0_gate.py" \
  --parent-manifest "${PARENT}/full_model_manifest.json" \
  --config "${CONFIG}" \
  --data-path "${GRPO2_DATA}" \
  --probe-data-path "${GRPO2_PROBE_DATA}" \
  --output "${ROOT}/step0_gate_step500.json"

COMMON=(
  --config "${CONFIG}"
  --parent-manifest "${PARENT}/full_model_manifest.json"
  --data-path "${GRPO2_DATA}"
  --output-dir "${ROOT}"
  --run-id "${RUN_ID}"
  --seed 20260816
  --lr 2e-7
  --max-steps 300
  --save-steps 10000
  --save-total-limit 5
  --probe-every-steps 0
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_think.py" "${COMMON[@]}" --preflight-only
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${SCRIPT_DIR}/run_grpo2_think.py" "${COMMON[@]}"

PARENT_HASH_AFTER=$("${PYTHON_BIN}" - "${PARENT}/full_model_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["canonical_model_identity"])
PY
)
test "${PARENT_HASH_BEFORE}" = "${PARENT_HASH_AFTER}"
mkdir -p "${RUN_ROOT}/logs"
cp "${OUTSIDE_LOG}" "${RUN_ROOT}/logs/formal-300.log"
printf '%s\n' READY_FOR_GRPO2_CHECKPOINT_EVALUATION
