#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${ROOT}/code"
CONTROL="${ROOT}/repro_control.py"
DATA_ROOT="${REPRO_DATA_ROOT:-/root/reproduce_datasets/onereason_final_chain_20260901}"
TEACHER_ROOT="${TEACHER_ROOT:-/root/teacher_expert_ckpt}"
BASE_MODEL="${BASE_MODEL:-/data/models/onereason-8b-pretrain-competition}"
OUTPUTS="${ROOT}/outputs"
LOGS="${ROOT}/logs"
STATE="${ROOT}/state"
RUNTIME_CONFIGS="${ROOT}/runtime_configs"
MONITOR_ROOT="${ROOT}/monitor"
CHECKPOINT_ROOT="${ROOT}/checkpoints"
REPRO_MONITOR="${ROOT}/repro_monitor_server.py"
REPRO_MONITOR_PORT="${REPRO_MONITOR_PORT:-8891}"

SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/local/bin/python3}"
SFT_VENV="${SFT_VENV:-/data/venvs/llamafactory-01398eb-liger081}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${ROOT}/dependencies/llamafactory}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

BASELINE_ROOT="${CODE_ROOT}/baselines/native_source_domain_r32_v3"
GRPO_ROOT="${BASELINE_ROOT}/grpo"
USER_ROOT="${GRPO_ROOT}/user"

SFT_RUN_DIR="${OUTPUTS}/01_sft_beta"
SFT_CHECKPOINT="${SFT_RUN_DIR}/checkpoint-1106"
GRREC_RUN_ID="GR-REC-V1-REPRO-TO1500"
GRREC_OUTPUT_ROOT="${OUTPUTS}/02_gr_rec_v1"
GRREC_RUN_DIR="${GRREC_OUTPUT_ROOT}/${GRREC_RUN_ID}"
GRREC_CHECKPOINT="${GRREC_RUN_DIR}/checkpoint-1500"
TK_RUN_ID="GRPO-TK-REPRO-TO250"
TK_OUTPUT_ROOT="${OUTPUTS}/03_grpo_tk"
TK_RUN_DIR="${TK_OUTPUT_ROOT}/${TK_RUN_ID}"
TK_CHECKPOINT="${TK_RUN_DIR}/checkpoint-250"
MC_RUN_ID="MC-USER-HYBRID-K4-REPRO-TO100"
MC_OUTPUT_ROOT="${OUTPUTS}/04_mc_user/runs"
MC_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/04_mc_user"
MC_RUN_DIR="${MC_OUTPUT_ROOT}/${MC_RUN_ID}"
MC_CHECKPOINT="${MC_CHECKPOINT_ROOT}/${MC_RUN_ID}/checkpoints/prompt-step-0100"

MODE="run"
FROM_STAGE=1
START_REPRO_MONITOR=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight) MODE="preflight"; shift ;;
    --monitor-only) MODE="monitor-only"; shift ;;
    --no-monitor) START_REPRO_MONITOR=0; shift ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash run.sh [--preflight|--monitor-only] [--from-stage 1|2|3|4] [--no-monitor]

Default mode reproduces the selected final checkpoint chain:
  BETA SFT 1106 -> GR_REC_v1 1500 -> GRPO-TK 250 -> MC_USER Hybrid 100

Completed stages are skipped only after their PASS marker and checkpoint
contract are both verified. Incomplete existing output is never overwritten.
The dedicated read-only reproduction dashboard starts on port 8891 by default.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "${FROM_STAGE}" =~ ^[1-4]$ ]]; then
  echo "--from-stage must be 1, 2, 3, or 4" >&2
  exit 2
fi
if [[ "${CUDA_DEVICES}" != "0,1,2,3" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3" >&2
  exit 2
fi

mkdir -p "${LOGS}" "${STATE}" "${RUNTIME_CONFIGS}" "${MONITOR_ROOT}" "${CHECKPOINT_ROOT}"
exec 9>"${ROOT}/.run.lock"
if ! flock -n 9; then
  echo "Another reproduction run holds ${ROOT}/.run.lock" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_CACHE="${ROOT}/cache/huggingface_datasets"
export HF_HOME="${ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT}/cache/transformers"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

"${SYSTEM_PYTHON}" "${CONTROL}" render-configs \
  --root "${ROOT}" --data-root "${DATA_ROOT}" --base-model "${BASE_MODEL}"
PREFLIGHT_JSON="$("${SYSTEM_PYTHON}" "${CONTROL}" preflight \
  --root "${ROOT}" --data-root "${DATA_ROOT}" --teacher-root "${TEACHER_ROOT}" \
  --base-model "${BASE_MODEL}" --base-manifest "${ROOT}/BASE_MODEL_SHA256SUMS" \
  --system-python "${SYSTEM_PYTHON}" --sft-python "${SFT_VENV}/bin/python" \
  --llamafactory-root "${LLAMAFACTORY_ROOT}" --environment-lock "${ROOT}/ENVIRONMENT_LOCK.json" \
  --code-root "${CODE_ROOT}")"
printf '%s\n' "${PREFLIGHT_JSON}" | tee "${STATE}/preflight_report.json"

start_repro_monitor() {
  local pid_file="${STATE}/repro_monitor.pid"
  if [[ -f "${pid_file}" ]]; then
    local existing_pid
    existing_pid="$(cat "${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "REPRO_MONITOR_RUNNING http://127.0.0.1:${REPRO_MONITOR_PORT} pid=${existing_pid}"
      return
    fi
  fi
  nohup "${SYSTEM_PYTHON}" "${REPRO_MONITOR}" \
    --root "${ROOT}" --host 127.0.0.1 --port "${REPRO_MONITOR_PORT}" \
    >"${LOGS}/repro_monitor.log" 2>&1 9>&- &
  local monitor_pid=$!
  printf '%s\n' "${monitor_pid}" >"${pid_file}"
  for _ in {1..30}; do
    if "${SYSTEM_PYTHON}" -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:${REPRO_MONITOR_PORT}/healthz',timeout=1)); raise SystemExit(0 if data.get('status') == 'ok' else 1)" 2>/dev/null; then
      echo "REPRO_MONITOR_READY http://127.0.0.1:${REPRO_MONITOR_PORT} pid=${monitor_pid}"
      return
    fi
    sleep 0.2
  done
  echo "Reproduction monitor failed to become ready; see ${LOGS}/repro_monitor.log" >&2
  return 1
}

if [[ "${MODE}" == "preflight" ]]; then
  echo "READY_TO_REPRODUCE_FINAL_CHAIN"
  exit 0
fi

if (( START_REPRO_MONITOR == 1 )); then
  start_repro_monitor
fi
if [[ "${MODE}" == "monitor-only" ]]; then
  exit 0
fi

CURRENT_STAGE="none"
on_error() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    "${SYSTEM_PYTHON}" "${CONTROL}" mark-failure \
      --state-dir "${STATE}" --stage "${CURRENT_STAGE}" --exit-code "${rc}" || true
    echo "STOPPED stage=${CURRENT_STAGE} exit_code=${rc}" >&2
  fi
  exit "${rc}"
}
trap on_error ERR INT TERM

stage_complete() {
  local stage="$1" checkpoint="$2" kind="$3" step="$4"
  local marker="${STATE}/${stage}.PASS.json"
  [[ -f "${marker}" ]] || return 1
  "${SYSTEM_PYTHON}" "${CONTROL}" verify-checkpoint \
    --path "${checkpoint}" --kind "${kind}" --step "${step}" >/dev/null
}

guard_new_stage() {
  local stage="$1" output_path="$2" checkpoint="$3" kind="$4" step="$5"
  if stage_complete "${stage}" "${checkpoint}" "${kind}" "${step}"; then
    echo "SKIP ${stage}: PASS marker and checkpoint verified"
    return 1
  fi
  if [[ -e "${output_path}" ]]; then
    echo "BLOCKED ${stage}: incomplete output exists at ${output_path}" >&2
    echo "Preserve it for audit; do not overwrite or auto-resume." >&2
    exit 4
  fi
  return 0
}

finish_stage() {
  local stage="$1" checkpoint="$2" kind="$3" step="$4"
  "${SYSTEM_PYTHON}" "${CONTROL}" verify-checkpoint \
    --path "${checkpoint}" --kind "${kind}" --step "${step}"
  "${SYSTEM_PYTHON}" "${CONTROL}" mark-pass \
    --state-dir "${STATE}" --stage "${stage}" --checkpoint "${checkpoint}" \
    --kind "${kind}" --step "${step}"
}

gpu_gate() {
  "${SYSTEM_PYTHON}" "${CONTROL}" gpu-check --expected-gpus 4 --memory-threshold-mib 1024
}

run_sft() {
  CURRENT_STAGE="01_sft_beta"
  if ! guard_new_stage "${CURRENT_STAGE}" "${SFT_RUN_DIR}" "${SFT_CHECKPOINT}" trainer 1106; then return; fi
  gpu_gate
  export PYTHONHASHSEED=20260806
  echo "START ${CURRENT_STAGE} $(date -Is)" | tee "${LOGS}/${CURRENT_STAGE}.log"
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    NPROC_PER_NODE=4 \
    NATIVE_GC_FRACTION=0.4 \
    GLOBAL_ITEM_WEIGHT=8 \
    MATERIAL_DOMAIN_MANIFEST="${DATA_ROOT}/01_sft_beta/manifest.json" \
    BASELINE_ROOT="${BASELINE_ROOT}" \
    REFERENCE_ROOT="${LLAMAFACTORY_ROOT}" \
    VENV_ROOT="${SFT_VENV}" \
    CONFIG_PATH="${RUNTIME_CONFIGS}/01_sft_beta.yaml" \
    bash "${BASELINE_ROOT}/scripts/run_native_source_domain_r32_v3.sh" \
    2>&1 | tee -a "${LOGS}/${CURRENT_STAGE}.log"
  finish_stage "${CURRENT_STAGE}" "${SFT_CHECKPOINT}" trainer 1106
}

run_grrec() {
  CURRENT_STAGE="02_gr_rec_v1"
  if ! guard_new_stage "${CURRENT_STAGE}" "${GRREC_OUTPUT_ROOT}" "${GRREC_CHECKPOINT}" trainer 1500; then return; fi
  stage_complete 01_sft_beta "${SFT_CHECKPOINT}" trainer 1106 || { echo "BETA parent is not verified" >&2; exit 5; }
  gpu_gate
  export PYTHONHASHSEED=20260816
  echo "START ${CURRENT_STAGE} $(date -Is)" | tee "${LOGS}/${CURRENT_STAGE}.log"
  (
    cd "${GRPO_ROOT}"
    env \
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      GRPO_BASE_MODEL="${BASE_MODEL}" \
      GRPO_DATA_PATH="${DATA_ROOT}/02_recommendation_grpo/train.jsonl" \
      GRPO_PARENT_ADAPTER="${SFT_CHECKPOINT}" \
      GRPO_MONITOR=1 \
      GRPO_MONITOR_DIR="${MONITOR_ROOT}" \
      GRPO_DETAILED_MONITOR=0 \
      GRPO_GENERATION_PROFILE=0 \
      GRPO_TRACE_EVERY=20 \
      GRPO_BEAM_RANK_BALANCE=1 \
      GRPO_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)" \
      PYTHONPATH="${LLAMAFACTORY_ROOT}/src:${GRPO_ROOT}/scripts:${PYTHONPATH:-}" \
      "${SYSTEM_PYTHON}" -m torch.distributed.run \
        --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29411 \
        scripts/run_grpo_trl_train.py \
        --run-id "${GRREC_RUN_ID}" --output-dir "${GRREC_OUTPUT_ROOT}" \
        --max-steps 1500 --lr 1e-6 --seed 20260816 --n-groups all \
        --save-steps 500 --save-total-limit 4 \
        --probe-groups 4 --probe-every-steps 200 --probe-seed 20260818
  ) 2>&1 | tee -a "${LOGS}/${CURRENT_STAGE}.log"
  finish_stage "${CURRENT_STAGE}" "${GRREC_CHECKPOINT}" trainer 1500
}

run_tk() {
  CURRENT_STAGE="03_grpo_tk"
  if ! guard_new_stage "${CURRENT_STAGE}" "${TK_OUTPUT_ROOT}" "${TK_CHECKPOINT}" trainer 250; then return; fi
  stage_complete 02_gr_rec_v1 "${GRREC_CHECKPOINT}" trainer 1500 || { echo "GR_REC parent is not verified" >&2; exit 5; }
  gpu_gate
  export PYTHONHASHSEED=20260816
  local parent_sha
  parent_sha="$(sha256sum "${GRREC_CHECKPOINT}/adapter_model.safetensors" | awk '{print $1}')"
  echo "START ${CURRENT_STAGE} $(date -Is)" | tee "${LOGS}/${CURRENT_STAGE}.log"
  (
    cd "${GRPO_ROOT}"
    env \
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      GRPO_BASE_MODEL="${BASE_MODEL}" \
      GRPO_DATA_PATH="${DATA_ROOT}/02_recommendation_grpo/train.jsonl" \
      GRPO_TK_PARENT_ADAPTER="${GRREC_CHECKPOINT}" \
      GRPO_TK_PARENT_SHA256="${parent_sha}" \
      GRPO_TK_EXPECTED_STEPS=250 \
      GRPO_TK_OUTPUT_ROOT="${TK_OUTPUT_ROOT}" \
      GRPO_PARENT_ADAPTER="${GRREC_CHECKPOINT}" \
      GRPO_MONITOR=1 \
      GRPO_MONITOR_DIR="${MONITOR_ROOT}" \
      GRPO_DETAILED_MONITOR=1 \
      GRPO_GENERATION_PROFILE=0 \
      GRPO_TRACE_EVERY=10 \
      GRPO_BEAM_RANK_BALANCE=1 \
      GRPO_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)" \
      PYTHONPATH="${LLAMAFACTORY_ROOT}/src:${GRPO_ROOT}:${GRPO_ROOT}/scripts:${PYTHONPATH:-}" \
      "${SYSTEM_PYTHON}" -m torch.distributed.run \
        --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29412 \
        -m ablations.gr_rec_think_sample8_fullsid_v3.run_sample8_fullsid_train \
        --run-id "${TK_RUN_ID}" --output-dir "${TK_OUTPUT_ROOT}" \
        --max-steps 250 --lr 1e-6 --seed 20260816 --n-groups all \
        --save-steps 50 --save-total-limit 64 \
        --probe-groups 4 --probe-every-steps 50 --probe-seed 20260818
  ) 2>&1 | tee -a "${LOGS}/${CURRENT_STAGE}.log"
  finish_stage "${CURRENT_STAGE}" "${TK_CHECKPOINT}" trainer 250
}

run_mc() {
  CURRENT_STAGE="04_mc_user"
  local stage_root="${OUTPUTS}/04_mc_user"
  if ! guard_new_stage "${CURRENT_STAGE}" "${stage_root}" "${MC_CHECKPOINT}" mc 100; then return; fi
  stage_complete 03_grpo_tk "${TK_CHECKPOINT}" trainer 250 || { echo "GRPO-TK parent is not verified" >&2; exit 5; }
  gpu_gate
  export PYTHONHASHSEED=20260823
  mkdir -p "${MC_OUTPUT_ROOT}" "${MC_CHECKPOINT_ROOT}"
  local runner="${USER_ROOT}/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py"
  local config="${RUNTIME_CONFIGS}/04_mc_user.json"
  local -a common_env=(
    "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
    "MC_USER_REPRO_ROOT=${ROOT}"
    "MC_USER_BASE_MODEL=${BASE_MODEL}"
    "MC_USER_PARENT_ADAPTER=${TK_CHECKPOINT}"
    "MC_USER_TRAIN_DATA=${DATA_ROOT}/03_user_grpo/train_3000.jsonl"
    "PYTHONPATH=${USER_ROOT}/scripts:${LLAMAFACTORY_ROOT}/src:${PYTHONPATH:-}"
  )
  echo "START ${CURRENT_STAGE} $(date -Is)" | tee "${LOGS}/${CURRENT_STAGE}.log"
  env "${common_env[@]}" "${SYSTEM_PYTHON}" "${runner}" \
    --config "${config}" --run-id "${MC_RUN_ID}" \
    --output-root "${MC_OUTPUT_ROOT}" --checkpoint-root "${MC_CHECKPOINT_ROOT}" \
    2>&1 | tee -a "${LOGS}/${CURRENT_STAGE}.log"
  "${SYSTEM_PYTHON}" "${CONTROL}" verify-mc-selection \
    --manifest "${MC_RUN_DIR}/manifest.json" \
    --expected "${DATA_ROOT}/03_user_grpo/selected_200_order.jsonl" --count 100
  env "${common_env[@]}" "${SYSTEM_PYTHON}" -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29413 \
    "${runner}" --config "${config}" --run-id "${MC_RUN_ID}" \
    --output-root "${MC_OUTPUT_ROOT}" --checkpoint-root "${MC_CHECKPOINT_ROOT}" --execute \
    2>&1 | tee -a "${LOGS}/${CURRENT_STAGE}.log"
  "${SYSTEM_PYTHON}" "${CONTROL}" verify-mc-summary \
    --summary "${MC_RUN_DIR}/summary.json" --prompt-step 100 --optimizer-step-max 100
  finish_stage "${CURRENT_STAGE}" "${MC_CHECKPOINT}" mc 100
}

if (( FROM_STAGE <= 1 )); then run_sft; fi
if (( FROM_STAGE <= 2 )); then run_grrec; fi
if (( FROM_STAGE <= 3 )); then run_tk; fi
if (( FROM_STAGE <= 4 )); then run_mc; fi

CURRENT_STAGE="complete"
"${SYSTEM_PYTHON}" "${CONTROL}" final-report \
  --root "${ROOT}" --sft "${SFT_CHECKPOINT}" --grrec "${GRREC_CHECKPOINT}" \
  --tk "${TK_CHECKPOINT}" --mc "${MC_CHECKPOINT}"
echo "FINAL_CHAIN_REPRODUCTION_PASS"
