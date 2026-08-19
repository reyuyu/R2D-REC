#!/usr/bin/env bash
set -euo pipefail

cd /data/GRPO_USER

if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 4 ]]; then
  echo "runtime benchmark requires exactly four GPUs" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU compute process detected; refusing to start" >&2
  exit 1
fi
while IFS=, read -r index used; do
  if [[ "${used// /}" -gt 1024 ]]; then
    echo "GPU ${index// /} is not idle" >&2
    exit 1
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/GRPO_USER/scripts
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

exec /data/venvs/llamafactory-01398eb-liger081/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/benchmark_user_grpo_runtime.py \
  --learning-rate 1e-6 \
  "$@"
