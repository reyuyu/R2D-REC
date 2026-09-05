#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run.sh --run-root PATH --sft-data-root PATH --grpo-data FILE [options]

Required:
  --run-root PATH       New, non-existing output root outside this Git checkout.
  --sft-data-root PATH  Frozen Rec FDR V4.3 data root containing base/ and recommendation/.
  --grpo-data FILE      Frozen rec_mp_grpo_v2 train.jsonl.

Options:
  --base-model PATH     OneReason-8B pretrain model (default: /data/models/onereason-8b-pretrain-competition).
  --sft-port PORT       Accelerate rendezvous port (default: 29749).
  -h, --help            Show this help.
EOF
}

BASE_MODEL="${BASE_MODEL:-/data/models/onereason-8b-pretrain-competition}"
RUN_ROOT=""
SFT_DATA_ROOT=""
GRPO_DATA=""
SFT_PORT="${SFT_PORT:-29749}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL=${2:-}; shift 2 ;;
    --run-root) RUN_ROOT=${2:-}; shift 2 ;;
    --sft-data-root) SFT_DATA_ROOT=${2:-}; shift 2 ;;
    --grpo-data) GRPO_DATA=${2:-}; shift 2 ;;
    --sft-port) SFT_PORT=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" || -z "$SFT_DATA_ROOT" || -z "$GRPO_DATA" ]]; then
  usage >&2
  exit 2
fi
if ! [[ "$SFT_PORT" =~ ^[0-9]+$ ]]; then
  echo "The SFT port must be a decimal integer." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
SFT_PACKAGE="$REPO_ROOT/reproduction/rec_fdr_v43_strictdet"
GRPO_PACKAGE="$REPO_ROOT/baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1"
VERIFY="$SCRIPT_DIR/verify_chain.py"

BASE_MODEL=$(realpath -e "$BASE_MODEL")
SFT_DATA_ROOT=$(realpath -e "$SFT_DATA_ROOT")
GRPO_DATA=$(realpath -e "$GRPO_DATA")
RUN_PARENT=$(realpath -e "$(dirname -- "$RUN_ROOT")")
RUN_ROOT="$RUN_PARENT/$(basename -- "$RUN_ROOT")"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Run root already exists; refusing to overwrite or resume: $RUN_ROOT" >&2
  exit 2
fi
case "$RUN_ROOT/" in
  "$REPO_ROOT/"*) echo "Run root must be outside the Git checkout." >&2; exit 2 ;;
esac
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "Git working tree must be clean before starting the deterministic chain." >&2
  exit 2
fi

for command in python3 nvidia-smi sha256sum git realpath; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done
if [[ $(nvidia-smi -L | wc -l) -lt 4 ]]; then
  echo "At least four physical GPUs are required." >&2
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU compute processes are present; refusing to compete for the four training GPUs." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"/{checkpoints,logs,monitor,outputs,state}
if ! mkdir "$RUN_ROOT/.run.lock"; then
  echo "Could not acquire run lock." >&2
  exit 2
fi

CHAIN_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$CHAIN_COMMIT" > "$RUN_ROOT/state/source_commit.txt"
printf '%s\n' "$STARTED_AT" > "$RUN_ROOT/state/started_at.txt"

mark_failed() {
  local exit_code=$?
  trap - ERR
  printf '{"status":"FAILED","exit_code":%d,"failed_at":"%s"}\n' \
    "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/state/FAILED.json"
  exit "$exit_code"
}
trap mark_failed ERR

assert_source_unchanged() {
  if [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$CHAIN_COMMIT" ]]; then
    echo "Git commit changed during the chain." >&2
    return 1
  fi
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Git working tree changed during the chain." >&2
    return 1
  fi
}

wait_for_gpu_release() {
  local deadline=$((SECONDS + 180))
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    if (( SECONDS >= deadline )); then
      echo "GPU processes did not release within 180 seconds." >&2
      return 1
    fi
    sleep 5
  done
}

echo "[1/2] Starting deterministic Rec FDR V4.3 full-parameter SFT."
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT="$SFT_PORT" \
  bash "$SFT_PACKAGE/scripts/launch_rec_fdr_v43_reproduction.sh" \
    --base-model "$BASE_MODEL" \
    --data-root "$SFT_DATA_ROOT" \
    --work-root "$RUN_ROOT/sft" \
  2>&1 | tee "$RUN_ROOT/logs/01_sft.log"

python3 "$VERIFY" verify-sft --output "$RUN_ROOT/sft/output" \
  --report "$RUN_ROOT/state/01_sft.PASS.json"
ln -s "$RUN_ROOT/sft/output" "$RUN_ROOT/checkpoints/01_sft_final"
wait_for_gpu_release
assert_source_unchanged

echo "[2/2] Starting deterministic GRPO-1 0-to-500 run."
FULLBASE_MODEL="$RUN_ROOT/sft/output" \
GRPO_DATA="$GRPO_DATA" \
RUN_ROOT="$RUN_ROOT/outputs" \
MONITOR_ROOT="$RUN_ROOT/monitor" \
RUN_ID="GRPO1-REC-BILATERAL-FULLBASE-CONSERVATIVE-R32-LR5E7-500-FINAL-ONLY" \
PYTHON_BIN=python3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash "$GRPO_PACKAGE/scripts/run_formal_500_final_only.sh" \
  2>&1 | tee "$RUN_ROOT/logs/02_grpo1.log"

GRPO_RUN_DIR="$RUN_ROOT/outputs/GRPO1-REC-BILATERAL-FULLBASE-CONSERVATIVE-R32-LR5E7-500-FINAL-ONLY"
python3 "$VERIFY" verify-grpo --output "$GRPO_RUN_DIR" \
  --report "$RUN_ROOT/state/02_grpo1.PASS.json"
ln -s "$GRPO_RUN_DIR/checkpoint-500" "$RUN_ROOT/checkpoints/02_grpo1_step500"
wait_for_gpu_release
assert_source_unchanged

python3 "$VERIFY" final-report \
  --run-root "$RUN_ROOT" --source-commit "$CHAIN_COMMIT" \
  --report "$RUN_ROOT/FINAL_REPORT.json"
cp "$RUN_ROOT/FINAL_REPORT.json" "$RUN_ROOT/state/PASS.json"
rmdir "$RUN_ROOT/.run.lock"
trap - ERR
echo "SFT_GRPO1_CHAIN_PASS"
