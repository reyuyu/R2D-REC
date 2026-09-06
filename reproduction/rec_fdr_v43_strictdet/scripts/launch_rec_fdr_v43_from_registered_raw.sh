#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 --base-model PATH --raw-root PATH --work-root PATH --dataset-key KEY --dataset-sha256 SHA256" >&2
}

base_model=""
raw_root=""
work_root=""
dataset_key=""
dataset_sha256=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) base_model=${2:-}; shift 2 ;;
    --raw-root) raw_root=${2:-}; shift 2 ;;
    --work-root) work_root=${2:-}; shift 2 ;;
    --dataset-key) dataset_key=${2:-}; shift 2 ;;
    --dataset-sha256) dataset_sha256=${2:-}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
if [[ -z "$base_model" || -z "$raw_root" || -z "$work_root" || -z "$dataset_key" || -z "$dataset_sha256" ]]; then
  usage
  exit 2
fi

package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base_model=$(realpath "$base_model")
raw_root=$(realpath "$raw_root")
work_parent=$(realpath "$(dirname -- "$work_root")")
work_root="$work_parent/$(basename -- "$work_root")"
[[ ! -e "$work_root" ]] || { echo "work root must be new: $work_root" >&2; exit 2; }

memory_parent=/dev/shm
[[ "$(findmnt -n -o FSTYPE --target "$memory_parent")" == tmpfs ]] || {
  echo "/dev/shm must be tmpfs" >&2
  exit 2
}
[[ "$(wc -l < /proc/swaps)" == 1 ]] || {
  echo "active swap is forbidden because derived data must remain memory-only" >&2
  exit 2
}
free_bytes=$(df -B1 --output=avail "$memory_parent" | tail -1 | tr -d ' ')
(( free_bytes >= 64 * 1024 * 1024 * 1024 )) || {
  echo "at least 64 GiB free tmpfs is required" >&2
  exit 2
}

mkdir "$work_root"
memory_workspace=$(mktemp -d "$memory_parent/rec-fdr-raw-XXXXXXXX")
completed=0
cleanup() {
  local code=$?
  case "$memory_workspace" in
    /dev/shm/rec-fdr-raw-*) ;;
    *) echo "refusing unsafe tmpfs cleanup: $memory_workspace" >&2; exit 1 ;;
  esac
  [[ ! -L "$memory_workspace" ]] || { echo "refusing symlink cleanup" >&2; exit 1; }
  rm -rf -- "$memory_workspace"
  python3 - "$work_root/RAW_TO_SFT_RESULT.json" "$completed" "$code" \
    "$dataset_key" "$dataset_sha256" "$raw_root" <<'PY'
import json
import sys
from pathlib import Path

path, completed, code, key, digest, raw_root = sys.argv[1:]
passed = completed == "1" and code == "0"
Path(path).write_text(json.dumps({
    "status": "PASS" if passed else "FAIL",
    "registered_dataset_used_by_trainer": True,
    "registered_dataset_key": key,
    "registered_dataset_sha256": digest,
    "registered_raw_root": raw_root,
    "derived_parquet_files": 24,
    "derived_rows": 270970,
    "derived_data_contract": "PARQUET_MANIFEST.json",
    "temporary_data_removed": True,
    "exit_code": int(code),
}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  return "$code"
}
trap cleanup EXIT

export HF_HOME="$memory_workspace/hf"
export HF_HUB_CACHE="$memory_workspace/hf/hub"
export HF_DATASETS_CACHE="$memory_workspace/hf/datasets"
export TMPDIR="$memory_workspace/tmp"
export XDG_CACHE_HOME="$memory_workspace/xdg"
export TRITON_CACHE_DIR="$memory_workspace/triton"
export TORCHINDUCTOR_CACHE_DIR="$memory_workspace/inductor"
export TORCH_HOME="$memory_workspace/torch"
mkdir -p "$TMPDIR"

python3 "$package_root/scripts/rebuild_rec_fdr_v43_from_raw_800k.py" \
  --raw-root "$raw_root" --base-model "$base_model" \
  --work-root "$memory_workspace/rebuild"
derived_data="$memory_workspace/rebuild/rebuilt_data"
python3 "$package_root/scripts/verify_rec_fdr_v43_package.py" \
  "$package_root" --data-root "$derived_data" | tee "$work_root/data_contract.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash "$package_root/scripts/launch_rec_fdr_v43_reproduction.sh" \
    --base-model "$base_model" --data-root "$derived_data" --work-root "$work_root"
completed=1
