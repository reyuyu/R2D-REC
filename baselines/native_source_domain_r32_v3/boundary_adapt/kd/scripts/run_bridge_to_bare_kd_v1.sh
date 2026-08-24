#!/usr/bin/env bash
set -euo pipefail
cd /data/GRPO
export PYTHONPATH=/data/GRPO:/data/GRPO/scripts
ROWS=/data/GRPO/boundary_adapt/results/bridge_to_bare_kd_rows.jsonl
STATS=/data/GRPO/boundary_adapt/results/dataset_stats.json
SCRIPT=boundary_adapt/kd/train_bridge_to_bare_kd.py
[[ -f "$SCRIPT" ]] || SCRIPT=baselines/native_source_domain_r32_v3/boundary_adapt/kd/train_bridge_to_bare_kd.py
MODE="${1:?usage: $0 preflight|formal}"
if [[ "$MODE" == preflight ]]; then
  torchrun --standalone --nproc_per_node=4 "$SCRIPT" --mode preflight --rows "$ROWS" --stats "$STATS" \
    --result /data/GRPO/boundary_adapt/results/bridge_to_bare_kd_preflight_20260824.json
elif [[ "$MODE" == formal ]]; then
  export BRIDGE_KD_CONFIRM_FORMAL=1
  torchrun --standalone --nproc_per_node=4 "$SCRIPT" --mode formal --rows "$ROWS" --stats "$STATS" \
    --output-dir /data/outputs/boundary_adapt/step900_bridge_to_bare_kd_v1 \
    --result /data/GRPO/boundary_adapt/results/bridge_to_bare_kd_training_20260824.json
else
  echo "invalid mode: $MODE" >&2; exit 2
fi
