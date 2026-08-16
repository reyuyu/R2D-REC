#!/usr/bin/env bash
set -euo pipefail
cd /data/GRPO/scripts
for i in 0 1 2 3; do
  off=$((i * 16))
  CUDA_VISIBLE_DEVICES=$i PYTHONIOENCODING=utf-8 \
    nohup /data/venvs/llamafactory-01398eb-liger081/bin/python run_smoke.py \
      --groups 64 --offset $off --limit 16 --device cuda:0 \
      --out /data/GRPO/logs/smoke_v1_gpu${i}.json \
      > /data/GRPO/logs/smoke_v1_gpu${i}.log 2>&1 &
  echo "started gpu $i pid $!"
done
echo "ALL_LAUNCHED"
