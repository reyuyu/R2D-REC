#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-GR-REC-VIDEO-OFFICIAL-ANTICOPY-V5-FORMAL-E1-20260830}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GRPO-checkpoints}"
MASTER_PORT="${MASTER_PORT:-29641}"
MAX_STEPS="${MAX_STEPS:-1074}"
export GRPO_RUN_ID="${RUN_ID}"
export GRPO_MONITOR_ROOT="${GRPO_MONITOR_ROOT:-/data/GRPO/runs}"

torchrun --standalone --nproc_per_node=4 --master_port="${MASTER_PORT}" \
  -m ablations.gr_rec_video_official_anticopy_v5.run_video_official_anticopy_train \
  --run-id "${RUN_ID}" \
  --n-groups all \
  --max-steps "${MAX_STEPS}" \
  --lr 1e-6 \
  --seed 20260816 \
  --output-dir "${OUTPUT_ROOT}" \
  --save-steps 50 \
  --save-total-limit 64 \
  --probe-groups 4 \
  --probe-group-id fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e \
  --probe-group-id 6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f \
  --probe-group-id 281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700 \
  --probe-group-id 2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8 \
  --probe-every-steps 50 \
  --probe-seed 20260818
