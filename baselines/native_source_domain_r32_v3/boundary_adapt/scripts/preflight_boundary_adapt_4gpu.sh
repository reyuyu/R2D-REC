#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/GRPO/boundary_adapt
OUT=$ROOT/results/boundary_adapt_preflight_20260824.json
DATA=$ROOT/results/boundary_adapt_rows.jsonl
PY=/data/venvs/llamafactory-01398eb-liger081/bin/python
mkdir -p "$ROOT/results"
"$PY" "$ROOT/build_boundary_adapt_dataset.py" --source /data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl --output "$DATA" --stats "$ROOT/results/dataset_stats.json"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 "$ROOT/train_boundary_adapt.py" \
  --mode preflight --rows "$DATA" --stats "$ROOT/results/dataset_stats.json" --result "$OUT"
