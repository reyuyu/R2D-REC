#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --base-model PATH --work-root PATH [--data-root PATH] [--prepare-only]" >&2
}

base_model=""
work_root=""
prepare_only=0
data_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) base_model=${2:-}; shift 2 ;;
    --work-root) work_root=${2:-}; shift 2 ;;
    --data-root) data_root=${2:-}; shift 2 ;;
    --prepare-only) prepare_only=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
if [[ -z "$base_model" || -z "$work_root" ]]; then
  usage
  exit 2
fi

package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base_model=$(realpath "$base_model")
mkdir -p "$work_root"
work_root=$(realpath "$work_root")
config="$work_root/training_config.yaml"
entry="$package_root/training/rec_fdr_v43_hcr_fullft_sft.py"
fsdp_config="$package_root/training/configs/fsdp_4gpu_full_shard.yaml"

export PYTHONPATH="$package_root/vendor/LLaMA-Factory/src:$package_root/training:$package_root/scripts/data_generation${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONHASHSEED=19260817
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REC_DISABLE_INTERMEDIATE_CHECKPOINTS=1

if [[ -z "$data_root" ]]; then
  data_root="$package_root/data"
else
  data_root=$(realpath "$data_root")
fi
if [[ ! -d "$data_root/base" || ! -d "$data_root/recommendation" ]]; then
  echo "Training data is not present. Pass --data-root with base/ and recommendation/." >&2
  exit 2
fi
python3 "$package_root/scripts/verify_rec_fdr_v43_package.py" "$package_root" --data-root "$data_root"
python3 "$package_root/scripts/verify_base_model.py" "$package_root/BASE_MODEL_MANIFEST.json" "$base_model"
python3 -m unittest -v "$package_root/training/test_rec_fdr_v43_hcr.py"
python3 "$package_root/scripts/materialize_rec_fdr_v43_config.py" \
  --package-root "$package_root" --base-model "$base_model" \
  --data-root "$data_root" \
  --work-root "$work_root" --output "$config"
python3 "$entry" "$config" --validate-only
python3 "$entry" "$config" --prepare-data-only

if [[ "$prepare_only" == 1 ]]; then
  echo "Tokenized pools prepared under $work_root/cache"
  exit 0
fi

python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, f"exactly four visible GPUs are required, got {torch.cuda.device_count()}"
print({"cuda_devices": torch.cuda.device_count()})
PY

accelerate launch \
  --config_file "$fsdp_config" \
  --main_process_port "${MASTER_PORT:-29749}" \
  "$entry" "$config" --reuse-cache 2>&1 | tee -a "$work_root/train.log"
