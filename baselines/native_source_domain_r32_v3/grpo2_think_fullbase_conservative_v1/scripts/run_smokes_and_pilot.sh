#!/usr/bin/env bash
set -euo pipefail

: "${GRPO2_PARENT_MANIFEST:?set the isolated GRPO-2 parent manifest}"
: "${GRPO2_DATA:?set the frozen Positive-A0 Think-only JSONL}"
: "${GRPO2_PROBE_DATA:?set the frozen recommendation GRPO JSONL for retention only}"
: "${GRPO2_OUTPUT_ROOT:?set a fresh output root}"
: "${GRPO2_MONITOR_ROOT:?set a fresh monitor root}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_A="GRPO2-THINK-SAMPLE8-SMOKE-A-${STAMP}"
SMOKE_B="GRPO2-THINK-SAMPLE8-SMOKE-B-${STAMP}"
PILOT="GRPO2-THINK-SAMPLE8-PILOT20-${STAMP}"

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

run_one() {
  local run_id="$1"
  local config="$2"
  local steps="$3"
  local probe_every="$4"
  local common=(
    --config "${config}"
    --parent-manifest "${GRPO2_PARENT_MANIFEST}"
    --allow-test-parent
    --data-path "${GRPO2_DATA}"
    --probe-data-path "${GRPO2_PROBE_DATA}"
    --output-dir "${GRPO2_OUTPUT_ROOT}"
    --run-id "${run_id}"
    --seed 20260816
    --lr 2e-7
    --max-steps "${steps}"
    --save-steps "$([ "${steps}" = 5 ] && echo 5 || echo 10)"
    --save-total-limit "$([ "${steps}" = 5 ] && echo 1 || echo 2)"
    --probe-every-steps "${probe_every}"
  )
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_think.py" "${common[@]}" --preflight-only
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/run_grpo2_think.py" "${common[@]}"
}

SMOKE_CONFIG="${PACKAGE_DIR}/config/smoke_5.json"
PILOT_CONFIG="${PACKAGE_DIR}/config/pilot_20.json"
run_one "${SMOKE_A}" "${SMOKE_CONFIG}" 5 5
run_one "${SMOKE_B}" "${SMOKE_CONFIG}" 5 5

COMPARE="${GRPO2_OUTPUT_ROOT}/smoke-comparison-${STAMP}.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_smokes.py" \
  --run-a "${GRPO2_OUTPUT_ROOT}/${SMOKE_A}" \
  --run-b "${GRPO2_OUTPUT_ROOT}/${SMOKE_B}" \
  --monitor-a "${GRPO2_MONITOR_ROOT}/${SMOKE_A}" \
  --monitor-b "${GRPO2_MONITOR_ROOT}/${SMOKE_B}" \
  --output "${COMPARE}"

"${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "BYTE_EXACT", p' "${COMPARE}"
run_one "${PILOT}" "${PILOT_CONFIG}" 20 20

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_report.py" \
  --comparison "${COMPARE}" \
  --pilot "${GRPO2_OUTPUT_ROOT}/${PILOT}/summary.json" \
  --output "${GRPO2_OUTPUT_ROOT}/GRPO2_PILOT_REPORT.json"

printf '%s\n' "READY_FOR_GRPO2_PARENT_FINALIZATION"
