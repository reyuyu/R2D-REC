#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

RUN_ID="${RUN_ID:-GR-REC-THINK-SAMPLE8-FULLSID-POSITIVE-A0-V3-STAGE2-E1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GRPO-checkpoints}"
export GRPO_PARENT_ADAPTER=/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-250
export GRPO_MONITOR=1
export GRPO_MONITOR_DIR="${GRPO_MONITOR_DIR:-/data/GRPO/runs}"
export GRPO_DETAILED_MONITOR=1
export GRPO_GENERATION_PROFILE=0
export GRPO_GIT_COMMIT="$(git rev-parse HEAD)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29255}" \
  -m ablations.gr_rec_think_sample8_fullsid_positive_a0_v3.run_positive_a0_train \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_ROOT}" \
  --lr 1e-6 \
  --seed 20260816 \
  --n-groups all \
  --max-steps 1222 \
  --probe-groups 4 \
  --probe-every-steps 50 \
  --save-steps 50 \
  --save-total-limit 64 \
  "$@"
