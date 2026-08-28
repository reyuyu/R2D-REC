#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

RUN_ID="${RUN_ID:-GR-REC-THINK-SUFFIX-SID-G8-REROLL-V1-FORMAL-E1-20260828}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GRPO-checkpoints}"
export GRPO_PARENT_ADAPTER=/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500
export GRPO_MONITOR=1
export GRPO_DETAILED_MONITOR=1
export GRPO_GENERATION_PROFILE=0
export GRPO_TRACE_EVERY="${GRPO_TRACE_EVERY:-10}"
export GRPO_GIT_COMMIT="$(git rev-parse HEAD)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29215}" \
  -m ablations.gr_rec_think_suffix_sid_v1.run_think_suffix_sid_train \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_ROOT}" \
  --lr 1e-6 \
  --seed 20260816 \
  --n-groups all \
  --probe-groups 0 \
  --save-steps 250 \
  --save-total-limit 8 \
  "$@"
