#!/usr/bin/env bash
cd /data/GRPO/scripts
for i in 0 1 2 3; do
  off=$((i * 8))
  CUDA_VISIBLE_DEVICES=$i PYTHONIOENCODING=utf-8 \
    nohup /data/venvs/llamafactory-01398eb-liger081/bin/python run_smoke.py \
      --groups 32 --offset $off --limit 8 --device cuda:0 \
      --out /data/GRPO/logs/smoke_v2_gpu${i}.json \
      > /data/GRPO/logs/smoke_v2_gpu${i}.log 2>&1 &
  echo "gpu $i pid $!"
done
echo ALL_LAUNCHED
