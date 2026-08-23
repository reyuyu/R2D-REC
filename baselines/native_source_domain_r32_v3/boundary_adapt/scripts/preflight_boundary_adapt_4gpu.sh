#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/GRPO/boundary_adapt
OUT=$ROOT/results/boundary_adapt_preflight_20260824.json
DATA=$ROOT/results/boundary_adapt_rows.jsonl
mkdir -p "$ROOT/results"
python "$ROOT/build_boundary_adapt_dataset.py" --source /data/GRPO/data/rec_mp_grpo_v2/train.jsonl --output "$DATA" --stats "$ROOT/results/dataset_stats.json"
torchrun --standalone --nproc_per_node=4 "$ROOT/train_boundary_adapt.py" --preflight-rows "$DATA" --result "$OUT"
