#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"
RUN_ID="${RUN_ID:-GR-REC-THINK-EXACT-SHARPEN-V4-FORMAL-E1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GRPO-checkpoints}"
export GRPO_PARENT_ADAPTER=/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-250
export GRPO_MONITOR=1 GRPO_DETAILED_MONITOR=1
export GRPO_GIT_COMMIT="$(git rev-parse HEAD)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
exec torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29245}" \
  -m ablations.gr_rec_think_exact_sharpen_v4.run_exact_sharpen_train \
  --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}" --lr 1e-6 --seed 20260816 \
  --n-groups all --probe-groups 4 --probe-every-steps 50 --save-steps 50 \
  --save-total-limit 64 "$@"
