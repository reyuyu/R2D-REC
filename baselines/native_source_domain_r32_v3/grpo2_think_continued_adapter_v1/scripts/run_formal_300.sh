#!/usr/bin/env bash
set -euo pipefail

: "${GRPO2_BASE_MODEL:?set immutable Rec FDR V4.3 full SFT directory}"
: "${GRPO2_ADAPTER_PARENT:?set original GRPO-1 checkpoint-500 directory}"
: "${GRPO2_DATA:?set frozen 611-row Think-only JSONL}"
: "${GRPO2_PROBE_DATA:?set frozen recommendation probe JSONL}"
: "${GRPO2_FORMAL_ROOT:?set fresh canonical server root}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="$(git -C "${PACKAGE_DIR}" rev-parse --show-toplevel)"
CONFIG="${PACKAGE_DIR}/config/formal_300.json"
RUN_ID="GRPO2-REC-THINK-CONTINUED-ADAPTER-FROM-GRPO1-STEP500-LR2E7-300"
RUN_ROOT="${GRPO2_FORMAL_ROOT}/${RUN_ID}"
MONITOR_ROOT="${GRPO2_FORMAL_ROOT}/monitor"
DRIVER_LOG="${GRPO2_FORMAL_ROOT}/logs/formal-300-driver.log"

printf 'RESOLVED_ROOT = %s\n' "${RUN_ROOT}"
test ! -e "${RUN_ROOT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
mkdir -p "${GRPO2_FORMAL_ROOT}/logs" "${MONITOR_ROOT}"

CHECKPOINT_BYTES=$(du -sb "${GRPO2_ADAPTER_PARENT}" | cut -f1)
EXPECTED_TOTAL_BYTES=$((CHECKPOINT_BYTES * 5))
SAFETY_BYTES=$((20 * 1024 * 1024 * 1024))
FREE_BYTES=$(df -B1 --output=avail "${GRPO2_FORMAL_ROOT}" | tail -1 | tr -d ' ')
printf 'FREE_DISK_GB = %s\nEXPECTED_CHECKPOINT_DISK_GB = %s\n' \
  "$((FREE_BYTES / 1024 / 1024 / 1024))" "$((EXPECTED_TOTAL_BYTES / 1024 / 1024 / 1024 + 1))"
test "${FREE_BYTES}" -gt "$((EXPECTED_TOTAL_BYTES + SAFETY_BYTES))"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
while IFS=, read -r used; do
  test "${used// /}" -lt 1024
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

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
export GRPO_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"

COMMON=(
  --config "${CONFIG}"
  --base-model "${GRPO2_BASE_MODEL}"
  --adapter-parent "${GRPO2_ADAPTER_PARENT}"
  --data-path "${GRPO2_DATA}"
  --probe-data-path "${GRPO2_PROBE_DATA}"
  --output-dir "${GRPO2_FORMAL_ROOT}"
  --run-id "${RUN_ID}"
  --seed 20260816
  --lr 2e-7
  --max-steps 300
  --save-steps 10000
  --save-total-limit 5
  --probe-every-steps 0
)

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_continued.py" "${COMMON[@]}" --preflight-only
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${SCRIPT_DIR}/run_grpo2_continued.py" "${COMMON[@]}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/evidence/offline-monitor" "${RUN_ROOT}/evidence/offline-runs"
cp "${DRIVER_LOG}" "${RUN_ROOT}/logs/formal-300-driver.log"

LABELS=(GRPO1-step500 GRPO2-step100 GRPO2-step150 GRPO2-step200 GRPO2-step250 GRPO2-step300)
STEPS=(0 100 150 200 250 300)
ADAPTERS=(
  "${GRPO2_ADAPTER_PARENT}"
  "${RUN_ROOT}/checkpoint-100"
  "${RUN_ROOT}/checkpoint-150"
  "${RUN_ROOT}/checkpoint-200"
  "${RUN_ROOT}/checkpoint-250"
  "${RUN_ROOT}/checkpoint-300"
)
ENTRIES=()
for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  step="${STEPS[$index]}"
  adapter="${ADAPTERS[$index]}"
  probe_run="OFFLINE-${label}"
  export GRPO_MONITOR_DIR="${RUN_ROOT}/evidence/offline-monitor"
  PROBE_COMMON=(
    --config "${CONFIG}"
    --base-model "${GRPO2_BASE_MODEL}"
    --adapter-parent "${GRPO2_ADAPTER_PARENT}"
    --probe-only-adapter "${adapter}"
    --probe-only-step "${step}"
    --data-path "${GRPO2_DATA}"
    --probe-data-path "${GRPO2_PROBE_DATA}"
    --output-dir "${RUN_ROOT}/evidence/offline-runs"
    --run-id "${probe_run}"
    --seed 20260816 --lr 2e-7 --max-steps 300
    --save-steps 10000 --save-total-limit 5 --probe-every-steps 0
  )
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_grpo2_continued.py" "${PROBE_COMMON[@]}" --preflight-only
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/run_grpo2_continued.py" "${PROBE_COMMON[@]}"
  ENTRIES+=(--entry "${label}=${RUN_ROOT}/evidence/offline-monitor/${probe_run}/probes.jsonl")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_offline_comparison.py" "${ENTRIES[@]}" \
  --output-json "${RUN_ROOT}/offline_checkpoint_comparison.json" \
  --output-md "${RUN_ROOT}/offline_checkpoint_comparison.md"

GPU_RELEASED=YES
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
  GPU_RELEASED=NO
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_formal_report.py" \
  --run-root "${RUN_ROOT}" \
  --comparison "${RUN_ROOT}/offline_checkpoint_comparison.json" \
  --code-root "${CODE_ROOT}" \
  --gpu-released "${GPU_RELEASED}"
printf '%s\n' READY_FOR_GRPO2_CHECKPOINT_EVALUATION
