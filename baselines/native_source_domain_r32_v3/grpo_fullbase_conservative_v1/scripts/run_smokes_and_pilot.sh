#!/usr/bin/env bash
set -euo pipefail

: "${FULLBASE_MODEL:?set FULLBASE_MODEL to the immutable full-SFT directory}"
: "${GRPO_DATA:?set GRPO_DATA to the frozen recommendation GRPO JSONL}"
: "${RUN_ROOT:?set RUN_ROOT to a new output root}"
: "${MONITOR_ROOT:?set MONITOR_ROOT to a new monitor root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SMOKE_CONFIG="${PACKAGE_DIR}/config/conservative_smoke_5.json"
PILOT_CONFIG="${PACKAGE_DIR}/config/conservative_pilot_20.json"
STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_A="GR-REC-FULLBASE-CONSERVATIVE-SMOKE-A-${STAMP}"
SMOKE_B="GR-REC-FULLBASE-CONSERVATIVE-SMOKE-B-${STAMP}"
PILOT="GR-REC-FULLBASE-CONSERVATIVE-PILOT20-${STAMP}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONHASHSEED=20260816
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false

run_one() {
  local run_id="$1"
  local config="$2"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_conservative_grpo.py" \
    --config "${config}" --base-model "${FULLBASE_MODEL}" --data-path "${GRPO_DATA}" \
    --output-root "${RUN_ROOT}" --monitor-root "${MONITOR_ROOT}" --run-id "${run_id}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${SCRIPT_DIR}/run_conservative_grpo.py" \
    --config "${config}" --base-model "${FULLBASE_MODEL}" --data-path "${GRPO_DATA}" \
    --output-root "${RUN_ROOT}" --monitor-root "${MONITOR_ROOT}" --run-id "${run_id}" --execute
}

run_one "${SMOKE_A}" "${SMOKE_CONFIG}"
run_one "${SMOKE_B}" "${SMOKE_CONFIG}"

COMPARE_PATH="${RUN_ROOT}/smoke-repeatability-${STAMP}.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_smokes.py" \
  --run-a "${RUN_ROOT}/${SMOKE_A}" --run-b "${RUN_ROOT}/${SMOKE_B}" \
  --output "${COMPARE_PATH}"

# compare_smokes exits nonzero on any exact evidence divergence, so the pilot
# can only begin after both independent five-step runs satisfy the contract.
run_one "${PILOT}" "${PILOT_CONFIG}"

"${PYTHON_BIN}" - "${RUN_ROOT}" "${STAMP}" "${SMOKE_A}" "${SMOKE_B}" "${PILOT}" <<'PY'
import json
import sys
from pathlib import Path

root, stamp, smoke_a, smoke_b, pilot = Path(sys.argv[1]), *sys.argv[2:]
report = {
    "status": "PASS",
    "smoke_a": smoke_a,
    "smoke_b": smoke_b,
    "pilot": pilot,
    "repeatability": json.loads((root / f"smoke-repeatability-{stamp}.json").read_text()),
    "pilot_summary": json.loads((root / pilot / "summary.json").read_text()),
}
(root / f"final-report-{stamp}.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, sort_keys=True))
PY
