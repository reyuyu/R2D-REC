#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/GRPO_USER
PYTHON=/data/venvs/llamafactory-01398eb-liger081/bin/python
SCRIPT="$ROOT/scripts/diagnose_user_chain.py"
RUN_ID="${RUN_ID:-GR-USER-CHAIN-DIAGNOSIS-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

BASE=/data/models/onereason-8b-pretrain-competition
C0=/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106
C20=$ROOT/runs/GR-USER-PILOT150-20260820-004130/pilot150-final
C40=$ROOT/runs/GR-USER-PILOT300-CONT-20260820-031716/pilot300-final
PROBE=$ROOT/data/gr_user_v1/probe_chain_v2.jsonl

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" evaluate --label C0 --base-model "$BASE" --adapter "$C0" --probe "$PROBE" --physical-gpu-id 0 --output "$RUN_DIR/C0.jsonl" --integrity "$RUN_DIR/C0_integrity.json" >"$RUN_DIR/C0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT" evaluate --label C20 --base-model "$BASE" --adapter "$C20" --probe "$PROBE" --physical-gpu-id 1 --output "$RUN_DIR/C20.jsonl" --integrity "$RUN_DIR/C20_integrity.json" >"$RUN_DIR/C20.log" 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$SCRIPT" evaluate --label C40 --base-model "$BASE" --adapter "$C40" --probe "$PROBE" --physical-gpu-id 2 --output "$RUN_DIR/C40.jsonl" --integrity "$RUN_DIR/C40_integrity.json" >"$RUN_DIR/C40.log" 2>&1 &
PID2=$!

STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
wait "$PID2" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "One or more inference-only diagnosis workers failed; inspect $RUN_DIR/*.log" >&2
  exit "$STATUS"
fi

echo "$RUN_ID"
