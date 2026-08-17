#!/usr/bin/env bash
set -euo pipefail
cd /data/GRPO/scripts
export PYTHONIOENCODING=utf-8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29517 \
  run_grpo_smoke.py --rollouts 5 --policy-epochs 2 --lr 1e-6 --seed 20260816 \
  > /data/GRPO/logs/grpo_smoke10.log 2>&1
