#!/usr/bin/env bash
set -euo pipefail

: "${GRPO2_BASE_MODEL:?set the immutable Rec FDR V4.3 full SFT directory}"
: "${GRPO2_ADAPTER_PARENT:?set the immutable GRPO-1 checkpoint-500 directory}"
: "${GRPO2_DATA:?set the frozen Positive-A0 Think-only JSONL}"
: "${GRPO2_PROBE_DATA:?set the frozen recommendation GRPO JSONL for retention only}"
: "${GRPO2_OUTPUT_ROOT:?set a fresh output root}"
: "${GRPO2_MONITOR_ROOT:?set a fresh monitor root}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NATIVE_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_A="GRPO2-CONTINUED-ADAPTER-SMOKE-A-${STAMP}"
SMOKE_B="GRPO2-CONTINUED-ADAPTER-SMOKE-B-${STAMP}"
PILOT="GRPO2-CONTINUED-ADAPTER-PILOT20-${STAMP}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONHASHSEED=20260816
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FLASH_ATTENTION_DETERMINISTIC=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export GRPO_MONITOR_DIR="${GRPO2_MONITOR_ROOT}"
export GRPO_GIT_COMMIT="${GRPO_GIT_COMMIT:-$(git -C "${PACKAGE_DIR}" rev-parse HEAD)}"

run_one() {
  local run_id="$1"
  local config="$2"
  local steps="$3"
  local probe_every="$4"
  local save_steps="$5"
  local save_limit="$6"
  local common=(
    --config "${config}"
    --base-model "${GRPO2_BASE_MODEL}"
    --adapter-parent "${GRPO2_ADAPTER_PARENT}"
    --data-path "${GRPO2_DATA}"
    --probe-data-path "${GRPO2_PROBE_DATA}"
    --output-dir "${GRPO2_OUTPUT_ROOT}"
    --run-id "${run_id}"
    --seed 20260816
    --lr 2e-7
    --max-steps "${steps}"
    --save-steps "${save_steps}"
    --save-total-limit "${save_limit}"
    --probe-every-steps "${probe_every}"
  )
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_continued.py" "${common[@]}" --preflight-only
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/run_grpo2_continued.py" "${common[@]}"
}

SMOKE_CONFIG="${PACKAGE_DIR}/config/smoke_5.json"
PILOT_CONFIG="${PACKAGE_DIR}/config/pilot_20.json"

STEP0="GRPO2-CONTINUED-ADAPTER-STEP0-${STAMP}"
STEP0_COMMON=(
  --config "${SMOKE_CONFIG}"
  --base-model "${GRPO2_BASE_MODEL}"
  --adapter-parent "${GRPO2_ADAPTER_PARENT}"
  --data-path "${GRPO2_DATA}"
  --probe-data-path "${GRPO2_PROBE_DATA}"
  --output-dir "${GRPO2_OUTPUT_ROOT}"
  --run-id "${STEP0}"
  --seed 20260816
  --lr 2e-7
  --max-steps 5
  --save-steps 5
  --save-total-limit 1
  --probe-every-steps 5
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_continued.py" "${STEP0_COMMON[@]}" --preflight-only
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${SCRIPT_DIR}/run_grpo2_continued.py" "${STEP0_COMMON[@]}" --step0-only
"${PYTHON_BIN}" -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["status"] == "PASS" and value["optimizer_steps"] == 0, value' \
  "${GRPO2_OUTPUT_ROOT}/${STEP0}/STEP0_INHERITED_ADAPTER_PARITY.json"

run_one "${SMOKE_A}" "${SMOKE_CONFIG}" 5 5 5 1
run_one "${SMOKE_B}" "${SMOKE_CONFIG}" 5 5 5 1

COMPARE="${GRPO2_OUTPUT_ROOT}/smoke-comparison-${STAMP}.json"
"${PYTHON_BIN}" "${NATIVE_DIR}/grpo2_think_fullbase_conservative_v1/scripts/compare_smokes.py" \
  --run-a "${GRPO2_OUTPUT_ROOT}/${SMOKE_A}" \
  --run-b "${GRPO2_OUTPUT_ROOT}/${SMOKE_B}" \
  --monitor-a "${GRPO2_MONITOR_ROOT}/${SMOKE_A}" \
  --monitor-b "${GRPO2_MONITOR_ROOT}/${SMOKE_B}" \
  --output "${COMPARE}"
"${PYTHON_BIN}" -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["status"] == "BYTE_EXACT", value' "${COMPARE}"

run_one "${PILOT}" "${PILOT_CONFIG}" 20 20 10 2
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_report.py" \
  --comparison "${COMPARE}" \
  --pilot "${GRPO2_OUTPUT_ROOT}/${PILOT}/summary.json" \
  --output "${GRPO2_OUTPUT_ROOT}/GRPO2_CONTINUED_ADAPTER_REPORT.json"

printf '%s\n' "READY_FOR_GRPO2_CONTINUED_ADAPTER_FORMAL_TRAINING"
