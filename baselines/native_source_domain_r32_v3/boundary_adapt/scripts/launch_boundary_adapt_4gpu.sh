#!/usr/bin/env bash
set -euo pipefail
if [[ "${BOUNDARY_ADAPT_CONFIRM_FORMAL:-}" != "1" ]]; then
  echo "Set BOUNDARY_ADAPT_CONFIRM_FORMAL=1 to launch formal Boundary Adaptation." >&2
  exit 2
fi
cd /data/GRPO/boundary_adapt
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
/data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  train_boundary_adapt.py --mode formal \
  --rows results/boundary_adapt_rows.jsonl --stats results/dataset_stats.json \
  --result results/formal_300step.json --output-dir /data/outputs/boundary_adapt/formal_300step --max-steps 300
