#!/usr/bin/env bash
set -Eeuo pipefail

SFT_EPOCHS="${SFT_EPOCHS:-1}"
GRPO1_STEPS="${GRPO1_STEPS:-300}"
GRPO2_STEPS="${GRPO2_STEPS:-250}"
GRPO3_STEPS="${GRPO3_STEPS:-200}"

RUN_SFT="${RUN_SFT:-1}"
RUN_GRPO1="${RUN_GRPO1:-1}"
RUN_GRPO2="${RUN_GRPO2:-0}"
RUN_GRPO3="${RUN_GRPO3:-1}"
DRY_RUN="${DRY_RUN:-0}"

BASE_MODEL="${BASE_MODEL:-/data/models/onereason-8b-pretrain-competition}"
SFT_DATASET_KEY="${SFT_DATASET_KEY:-rec_fdr_v43_full_sft_raw_800k}"
REPRO_DATA_ROOT="${REPRO_DATA_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-/root/onereason_deterministic_pipeline_$(date -u +%Y%m%d-%H%M%S)}"
SFT_PORT="${SFT_PORT:-29759}"
GRPO2_PROBE_DATA="${GRPO2_PROBE_DATA:-}"
EXPECTED_SFT_MODEL_SHA256="${EXPECTED_SFT_MODEL_SHA256:-8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59}"
EXPECTED_GRPO1_STEP300_ADAPTER_SHA256="${EXPECTED_GRPO1_STEP300_ADAPTER_SHA256:-fbae37f3892c414a7c86f2285a4568f2a5bb526c8d249616f0445bd9b90f6c19}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
CONTROL="$SCRIPT_DIR/scripts/pipeline_control.py"
RESOLVER="$SCRIPT_DIR/scripts/resolve_repro_dataset.py"
SFT_PACKAGE="$REPO_ROOT/reproduction/rec_fdr_v43_strictdet"
SFT_RAW_MANIFEST="$SFT_PACKAGE/RAW_800K_MANIFEST.json"
if [[ "$DRY_RUN" == 1 && -n "${REPRO_TEST_SFT_MANIFEST:-}" ]]; then
  SFT_RAW_MANIFEST="$REPRO_TEST_SFT_MANIFEST"
fi
GRPO1_PACKAGE="$REPO_ROOT/baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1"
GRPO2_PACKAGE="$REPO_ROOT/baselines/native_source_domain_r32_v3/grpo2_think_continued_adapter_v1"
GRPO3_PACKAGE="$REPO_ROOT/baselines/native_source_domain_r32_v3/grpo/user"

for value in "$RUN_SFT" "$RUN_GRPO1" "$RUN_GRPO2" "$RUN_GRPO3" "$DRY_RUN"; do
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "stage flags must be 0 or 1" >&2; exit 2; }
done
for value in "$GRPO1_STEPS" "$GRPO2_STEPS" "$GRPO3_STEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "step counts must be positive integers" >&2; exit 2; }
done
[[ "$SFT_EPOCHS" == 1 ]] || { echo "only the validated one-epoch SFT is supported" >&2; exit 2; }
[[ "$GRPO3_STEPS" == 200 ]] || { echo "only the validated 200-step User-GRPO is supported" >&2; exit 2; }

resolve_dataset() {
  local key=$1
  local args=(--key "$key" --format lines)
  [[ -z "$REPRO_DATA_ROOT" ]] || args+=(--root "$REPRO_DATA_ROOT")
  python3 "$RESOLVER" "${args[@]}"
}

resolve_sft_dataset() {
  local args=(--key "$SFT_DATASET_KEY" --manifest "$SFT_RAW_MANIFEST" --format lines)
  [[ -z "$REPRO_DATA_ROOT" ]] || args+=(--root "$REPRO_DATA_ROOT")
  python3 "$RESOLVER" "${args[@]}"
}

if [[ "$RUN_SFT" == 1 ]]; then
  mapfile -t SFT_DATASET < <(resolve_sft_dataset)
  [[ "${SFT_DATASET[6]}" == raw_parquet_directory ]] || {
    echo "SFT registry entry must be kind=raw_parquet_directory" >&2
    exit 2
  }
fi
if [[ "$RUN_GRPO1" == 1 ]]; then
  mapfile -t GRPO1_DATASET < <(resolve_dataset recommendation_grpo_bilateral)
fi
if [[ "$RUN_GRPO2" == 1 ]]; then
  mapfile -t GRPO2_DATASET < <(resolve_dataset recommendation_grpo_think_only)
fi
if [[ "$RUN_GRPO3" == 1 ]]; then
  mapfile -t GRPO3_DATASET < <(resolve_dataset user_grpo)
fi

GRPO3_PARENT_STAGE=GRPO1
[[ "$RUN_GRPO2" == 0 ]] || GRPO3_PARENT_STAGE=GRPO2
PIPELINE="SFT -> GRPO1 -> GRPO3"
[[ "$RUN_GRPO2" == 0 ]] || PIPELINE="SFT -> GRPO1 -> GRPO2 -> GRPO3"

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY_RUN=PASS\n'
  printf 'PIPELINE=%s\n' "$PIPELINE"
  printf 'SFT_EPOCHS=%s\nGRPO1_STEPS=%s\n' "$SFT_EPOCHS" "$GRPO1_STEPS"
  printf 'SFT_REGISTERED_DATASET=%s\n' "${SFT_DATASET[0]}"
  printf 'SFT_REGISTERED_RAW_PATH=%s\n' "${SFT_DATASET[1]}"
  printf 'SFT_REGISTERED_RAW_SHA256=%s\n' "${SFT_DATASET[2]}"
  printf 'SFT_RAW_TO_TRAINING_DATA=AUTOMATIC_TMPFS\n'
  if [[ "$RUN_GRPO2" == 1 ]]; then
    printf 'GRPO2_STATUS=ENABLED\nGRPO2_STEPS=%s\n' "$GRPO2_STEPS"
  else
    printf 'GRPO2_STATUS=SKIPPED\nGRPO2_DATASET_RESOLUTION=SKIPPED\n'
  fi
  printf 'GRPO3_STEPS=%s\nGRPO3_PARENT=%s_FINAL_ADAPTER\n' "$GRPO3_STEPS" "$GRPO3_PARENT_STAGE"
  exit 0
fi

[[ "$RUN_SFT" == 1 && "$RUN_GRPO1" == 1 && "$RUN_GRPO3" == 1 ]] || {
  echo "execution currently requires SFT, GRPO1, and GRPO3" >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists" >&2; exit 2; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Git working tree must be clean" >&2
  exit 2
}
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" ]] || {
  echo "GPU compute processes are already present" >&2
  exit 2
}
if [[ "$RUN_GRPO2" == 1 ]]; then
  [[ -f "$GRPO2_PROBE_DATA" ]] || {
    echo "RUN_GRPO2=1 requires GRPO2_PROBE_DATA to name the frozen probe JSONL" >&2
    exit 2
  }
fi

mkdir -p "$RUN_ROOT"/{configs,logs,monitor,reports,state,work}
mkdir "$RUN_ROOT/.run.lock"
SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)

failed() {
  local code=$?
  trap - ERR
  python3 - "$RUN_ROOT/state/FAILED.json" "$code" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"status": "FAILED", "exit_code": int(sys.argv[2])}, indent=2) + "\n",
    encoding="utf-8",
)
PY
  exit "$code"
}
trap failed ERR

assert_source_unchanged() {
  [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
}

wait_for_gpu_release() {
  local deadline=$((SECONDS + 180))
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" ]]; do
    (( SECONDS < deadline )) || { echo "GPU release timeout" >&2; return 1; }
    sleep 5
  done
}

export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "[1] Full-parameter SFT: 1 epoch"
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT="$SFT_PORT" \
  bash "$SFT_PACKAGE/scripts/launch_rec_fdr_v43_from_registered_raw.sh" \
    --base-model "$BASE_MODEL" --raw-root "${SFT_DATASET[1]}" \
    --dataset-key "${SFT_DATASET[0]}" --dataset-sha256 "${SFT_DATASET[2]}" \
    --work-root "$RUN_ROOT/work/sft" 2>&1 | tee "$RUN_ROOT/logs/01_sft.log"
SFT_FINAL="$RUN_ROOT/work/sft/output"
python3 "$CONTROL" verify-sft --output "$SFT_FINAL" --epochs "$SFT_EPOCHS" \
  --expected-model-sha256 "$EXPECTED_SFT_MODEL_SHA256" \
  --dataset-key "${SFT_DATASET[0]}" --dataset-sha256 "${SFT_DATASET[2]}" \
  --dataset-rows "${SFT_DATASET[3]}" --dataset-split "${SFT_DATASET[4]}" \
  --derivation-report "$RUN_ROOT/work/sft/RAW_TO_SFT_RESULT.json" \
  --report "$RUN_ROOT/reports/sft.json"
ln -s "$SFT_FINAL" "$RUN_ROOT/01_sft_final"
wait_for_gpu_release
assert_source_unchanged

echo "[2] GRPO1 Recommendation Bilateral: $GRPO1_STEPS steps"
GRPO1_CONFIG="$RUN_ROOT/configs/grpo1.json"
python3 "$CONTROL" materialize-grpo1 \
  --template "$GRPO1_PACKAGE/config/formal_500_final_only.json" \
  --output "$GRPO1_CONFIG" --base-model "$SFT_FINAL" --steps "$GRPO1_STEPS" \
  --dataset-key "${GRPO1_DATASET[0]}" \
  --dataset-sha256 "${GRPO1_DATASET[2]}" --dataset-rows "${GRPO1_DATASET[3]}" \
  --dataset-split "${GRPO1_DATASET[4]}"
GRPO1_RUN_ID="GRPO1-PIPELINE-${GRPO1_STEPS}"
GRPO1_OUTPUT_ROOT="$RUN_ROOT/work/grpo1/outputs"
GRPO1_MONITOR_ROOT="$RUN_ROOT/monitor/grpo1"
GRPO1_RUN="$GRPO1_OUTPUT_ROOT/$GRPO1_RUN_ID"
GRPO1_RUNNER="$GRPO1_PACKAGE/scripts/run_conservative_grpo.py"
COMMON_GRPO1=(--config "$GRPO1_CONFIG" --base-model "$SFT_FINAL" \
  --data-path "${GRPO1_DATASET[1]}" --output-root "$GRPO1_OUTPUT_ROOT" \
  --monitor-root "$GRPO1_MONITOR_ROOT" --run-id "$GRPO1_RUN_ID")
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "$GRPO1_RUNNER" "${COMMON_GRPO1[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$GRPO1_RUNNER" "${COMMON_GRPO1[@]}" --execute 2>&1 | tee "$RUN_ROOT/logs/02_grpo1.log"
GRPO1_FINAL="$GRPO1_RUN/checkpoint-$GRPO1_STEPS"
GRPO1_EXPECTED_ARGS=()
if [[ "$GRPO1_STEPS" == 300 ]]; then
  GRPO1_EXPECTED_ARGS=(--expected-adapter-sha256 "$EXPECTED_GRPO1_STEP300_ADAPTER_SHA256")
fi
python3 "$CONTROL" verify-adapter --stage GRPO1 --run-dir "$GRPO1_RUN" \
  --checkpoint "$GRPO1_FINAL" --checkpoint-container "$GRPO1_RUN" \
  --config "$GRPO1_CONFIG" --step "$GRPO1_STEPS" \
  --parent-sha256 "$EXPECTED_SFT_MODEL_SHA256" \
  --dataset-key "${GRPO1_DATASET[0]}" --dataset-sha256 "${GRPO1_DATASET[2]}" \
  --report "$RUN_ROOT/reports/grpo1.json" "${GRPO1_EXPECTED_ARGS[@]}"
ln -s "$GRPO1_FINAL" "$RUN_ROOT/02_grpo1_final"
wait_for_gpu_release
assert_source_unchanged

if [[ "$RUN_GRPO2" == 1 ]]; then
  echo "[3] Optional GRPO2 Think-only: $GRPO2_STEPS steps"
  GRPO2_CONFIG="$RUN_ROOT/configs/grpo2.json"
  python3 "$CONTROL" materialize-grpo2 \
    --template "$GRPO2_PACKAGE/config/formal_300.json" --output "$GRPO2_CONFIG" \
    --base-model "$SFT_FINAL" --adapter-parent "$GRPO1_FINAL" \
    --parent-step "$GRPO1_STEPS" --parent-dataset-sha256 "${GRPO1_DATASET[2]}" \
    --steps "$GRPO2_STEPS" --dataset-key "${GRPO2_DATASET[0]}" \
    --dataset-sha256 "${GRPO2_DATASET[2]}" --dataset-rows "${GRPO2_DATASET[3]}" \
    --dataset-split "${GRPO2_DATASET[4]}"
  GRPO2_RUN_ID="GRPO2-PIPELINE-${GRPO2_STEPS}"
  GRPO2_RUN="$RUN_ROOT/work/grpo2/$GRPO2_RUN_ID"
  GRPO2_RUNNER="$GRPO2_PACKAGE/scripts/run_grpo2_continued.py"
  export GRPO_MONITOR_DIR="$RUN_ROOT/monitor/grpo2"
  COMMON_GRPO2=(--config "$GRPO2_CONFIG" --base-model "$SFT_FINAL" \
    --adapter-parent "$GRPO1_FINAL" --data-path "${GRPO2_DATASET[1]}" \
    --probe-data-path "$GRPO2_PROBE_DATA" --output-dir "$RUN_ROOT/work/grpo2" \
    --run-id "$GRPO2_RUN_ID" --seed 20260816 --lr 2e-7 \
    --max-steps "$GRPO2_STEPS" --save-steps 10000 --save-total-limit 1 \
    --probe-every-steps 0)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "$GRPO2_RUNNER" "${COMMON_GRPO2[@]}" --preflight-only
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$GRPO2_RUNNER" "${COMMON_GRPO2[@]}" 2>&1 | tee "$RUN_ROOT/logs/03_grpo2.log"
  GRPO2_FINAL="$GRPO2_RUN/checkpoint-$GRPO2_STEPS"
  GRPO1_SHA=$(sha256sum "$GRPO1_FINAL/adapter_model.safetensors" | awk '{print $1}')
  python3 "$CONTROL" verify-adapter --stage GRPO2 --run-dir "$GRPO2_RUN" \
    --checkpoint "$GRPO2_FINAL" --checkpoint-container "$GRPO2_RUN" \
    --config "$GRPO2_CONFIG" --step "$GRPO2_STEPS" --parent-sha256 "$GRPO1_SHA" \
    --dataset-key "${GRPO2_DATASET[0]}" --dataset-sha256 "${GRPO2_DATASET[2]}" \
    --report "$RUN_ROOT/reports/grpo2.json"
  ln -s "$GRPO2_FINAL" "$RUN_ROOT/03_grpo2_final"
  GRPO3_PARENT="$GRPO2_FINAL"
  GRPO3_PARENT_STEP="$GRPO2_STEPS"
  GRPO3_LOG_INDEX=04
  GRPO3_LINK="$RUN_ROOT/04_grpo3_final"
  wait_for_gpu_release
  assert_source_unchanged
else
  GRPO3_PARENT="$GRPO1_FINAL"
  GRPO3_PARENT_STEP="$GRPO1_STEPS"
  GRPO3_LOG_INDEX=03
  GRPO3_LINK="$RUN_ROOT/03_grpo3_final"
fi

echo "[$GRPO3_LOG_INDEX] GRPO3 User: $GRPO3_STEPS steps from $GRPO3_PARENT_STAGE"
GRPO3_CONFIG="$RUN_ROOT/configs/grpo3.json"
python3 "$CONTROL" materialize-grpo3 \
  --template "$GRPO3_PACKAGE/configs/grpo3_user_from_grpo1_step300_formal_v1.json" \
  --output "$GRPO3_CONFIG" --base-model "$SFT_FINAL" --adapter-parent "$GRPO3_PARENT" \
  --parent-step "$GRPO3_PARENT_STEP" --parent-stage "$GRPO3_PARENT_STAGE" \
  --steps "$GRPO3_STEPS" --dataset-key "${GRPO3_DATASET[0]}" \
  --dataset-path "${GRPO3_DATASET[1]}" --dataset-sha256 "${GRPO3_DATASET[2]}" \
  --dataset-rows "${GRPO3_DATASET[3]}" --dataset-split "${GRPO3_DATASET[4]}"
GRPO3_RUN_ID="GRPO3-USER-PIPELINE-${GRPO3_STEPS}"
GRPO3_RUNS="$RUN_ROOT/work/grpo3/runs"
GRPO3_CHECKPOINTS="$RUN_ROOT/work/grpo3/checkpoints"
GRPO3_RUN="$GRPO3_RUNS/$GRPO3_RUN_ID"
GRPO3_RUNNER="$GRPO3_PACKAGE/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py"
COMMON_GRPO3=(--config "$GRPO3_CONFIG" --run-id "$GRPO3_RUN_ID" \
  --output-root "$GRPO3_RUNS" --checkpoint-root "$GRPO3_CHECKPOINTS")
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "$GRPO3_RUNNER" "${COMMON_GRPO3[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$GRPO3_RUNNER" "${COMMON_GRPO3[@]}" --execute 2>&1 | tee "$RUN_ROOT/logs/${GRPO3_LOG_INDEX}_grpo3.log"
GRPO3_CONTAINER="$GRPO3_CHECKPOINTS/$GRPO3_RUN_ID/checkpoints"
GRPO3_FINAL="$GRPO3_CONTAINER/prompt-step-$(printf '%04d' "$GRPO3_STEPS")"
GRPO3_PARENT_SHA=$(sha256sum "$GRPO3_PARENT/adapter_model.safetensors" | awk '{print $1}')
python3 "$CONTROL" verify-adapter --stage GRPO3 --run-dir "$GRPO3_RUN" \
  --checkpoint "$GRPO3_FINAL" --checkpoint-container "$GRPO3_CONTAINER" \
  --config "$GRPO3_CONFIG" --step "$GRPO3_STEPS" \
  --parent-sha256 "$GRPO3_PARENT_SHA" --parent-stage "$GRPO3_PARENT_STAGE" \
  --dataset-key "${GRPO3_DATASET[0]}" --dataset-sha256 "${GRPO3_DATASET[2]}" \
  --report "$RUN_ROOT/reports/grpo3.json"
ln -s "$GRPO3_FINAL" "$GRPO3_LINK"
wait_for_gpu_release
assert_source_unchanged

REPORT_ARGS=(--output "$RUN_ROOT/PIPELINE_REPORT.json" --source-commit "$SOURCE_COMMIT" \
  --repro-data-root "${SFT_DATASET[5]}" --sft-epochs "$SFT_EPOCHS" \
  --sft-dataset-key "$SFT_DATASET_KEY" \
  --grpo1-steps "$GRPO1_STEPS" --grpo2-steps "$GRPO2_STEPS" \
  --grpo3-steps "$GRPO3_STEPS" --run-sft "$RUN_SFT" --run-grpo1 "$RUN_GRPO1" \
  --run-grpo2 "$RUN_GRPO2" --run-grpo3 "$RUN_GRPO3" \
  --stage-report "sft=$RUN_ROOT/reports/sft.json" \
  --stage-report "grpo1=$RUN_ROOT/reports/grpo1.json" \
  --stage-report "grpo3=$RUN_ROOT/reports/grpo3.json")
[[ "$RUN_GRPO2" == 0 ]] || REPORT_ARGS+=(--stage-report "grpo2=$RUN_ROOT/reports/grpo2.json")
python3 "$CONTROL" final-report "${REPORT_ARGS[@]}"
cp "$RUN_ROOT/PIPELINE_REPORT.json" "$RUN_ROOT/state/PASS.json"
rmdir "$RUN_ROOT/.run.lock"
trap - ERR
echo "PIPELINE_PASS"
