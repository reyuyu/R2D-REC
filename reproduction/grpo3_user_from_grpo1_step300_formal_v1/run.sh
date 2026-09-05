#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${GRPO3_FORMAL_ROOT:-/root/grpo3_user_from_grpo1_step300_formal_200_20260906}"
SOURCE="$ROOT/source"
RUNS="$ROOT/runs"
CHECKPOINTS="$ROOT/checkpoints"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
EVIDENCE="$ROOT/evidence"
RUN_ID="GRPO3-USER-CONTINUED-GRPO1-STEP300-LR3E7-200"
RUN_DIR="$RUNS/$RUN_ID"
USER_DIR="$SOURCE/baselines/native_source_domain_r32_v3/grpo/user"
RUNNER="$USER_DIR/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py"
PROBE_RUNNER="$USER_DIR/scripts/run_mc_user_formal_probe_sidecar_v1.py"
CONFIG="$USER_DIR/configs/grpo3_user_from_grpo1_step300_formal_v1.json"
PROBE_DATA="/data/GRPO_USER/data/gr_user_v1/probe_v1.jsonl"
SMOKE_COMPARISON="/root/grpo3_user_determinism_20260906/comparison.json"
PYTHON_BIN="/usr/local/bin/python3"

mkdir -p "$RUNS" "$CHECKPOINTS" "$STATE" "$LOGS" "$EVIDENCE"

write_state() {
  "$PYTHON_BIN" - "$STATE/status.json" "$1" "${2:-}" <<'PY'
import json
import sys
from pathlib import Path

path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path.write_text(json.dumps({"state": state, "detail": detail}, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  write_state STOPPED "command failed at line $1"
}
trap 'on_error "$LINENO"' ERR

test ! -e "$RUN_DIR"
test -f "$SMOKE_COMPARISON"
test -f "$PROBE_DATA"
test -z "$(git -C "$SOURCE" status --porcelain)"
"$PYTHON_BIN" - "$SMOKE_COMPARISON" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "status": "PASS",
    "FIRST_DIVERGENCE": "NONE",
    "DETERMINISM_LEVEL": "BYTE_EXACT",
    "REGISTERED_DATASET_USED_BY_TRAINER": "YES",
    "TRAINING_SEMANTICS_CHANGED": "NO",
    "READY_FOR_GRPO3_FORMAL_TRAINING": "YES",
}
actual = {key: value.get(key) for key in required}
if actual != required:
    raise SystemExit(f"determinism gate failed: {actual}")
PY

FREE_BYTES=$(df -B1 --output=avail "$ROOT" | tail -1 | tr -d ' ')
test "$FREE_BYTES" -gt $((20 * 1024 * 1024 * 1024))

export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=20260823

write_state PREFLIGHT "validating GRPO1 checkpoint-300 parent, registered data, source and four GPUs"
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYTHON_BIN" "$RUNNER" \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --output-root "$RUNS" \
  --checkpoint-root "$CHECKPOINTS" \
  >"$LOGS/preflight.log" 2>&1

write_state STEP0_USER_PROBE "frozen 3+3 User trend proxy; not an official aggregate score"
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" "$PROBE_RUNNER" \
  --formal-run-dir "$RUN_DIR" \
  --gpu-id 1 \
  --base-model "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model"])' "$CONFIG")" \
  --parent-adapter "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"])' "$CONFIG")" \
  --probe "$PROBE_DATA" \
  --stop-after-step 0 \
  --execute \
  >"$LOGS/step0-user-probe.log" 2>&1
cp "$RUN_DIR/evaluations/user_light_probe/results.json" "$EVIDENCE/step0_user_probe.json"

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
write_state TRAINING "four-GPU candidate-parallel DDP, K=4, 200 prompts; GRPO1 checkpoint-300 parent"
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$RUNNER" \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --execute \
  --output-root "$RUNS" \
  --checkpoint-root "$CHECKPOINTS" \
  >"$LOGS/formal-200.log" 2>&1

write_state USER_CHECKPOINT_PROBES "paired frozen User probe for parent and six checkpoints"
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" "$PROBE_RUNNER" \
  --formal-run-dir "$RUN_DIR" \
  --gpu-id 1 \
  --base-model "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model"])' "$CONFIG")" \
  --parent-adapter "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"])' "$CONFIG")" \
  --probe "$PROBE_DATA" \
  --execute \
  >"$LOGS/user-checkpoint-probes.log" 2>&1

write_state TRAINING_AND_USER_PROBES_COMPLETE "official aggregate evaluation remains pending"
trap - ERR
