#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
STATE_DIR="${STATE_DIR:-/data/schedulers/bata_baseline_4gpu}"
LAUNCHER="${BASELINE_ROOT}/scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh"
POLL_SECONDS="${POLL_SECONDS:-60}"
REQUIRED_STABLE_CHECKS="${REQUIRED_STABLE_CHECKS:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-1024}"
GPU_LIST="0,1,2,3"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/watcher.lock"
if ! flock -n 9; then
  echo "Another bata_baseline watcher already owns ${STATE_DIR}/watcher.lock" >&2
  exit 2
fi

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

all_four_gpus_free() {
  local rows compute_apps expected_index used
  mapfile -t rows < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sed 's/[[:space:]]//g'
  )
  if [[ "${#rows[@]}" -ne 4 ]]; then
    return 1
  fi
  for expected_index in 0 1 2 3; do
    IFS=',' read -r index used <<< "${rows[${expected_index}]}"
    if [[ "${index}" != "${expected_index}" || ! "${used}" =~ ^[0-9]+$ || "${used}" -gt "${MAX_USED_MIB}" ]]; then
      return 1
    fi
  done
  compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -z "${compute_apps}" ]]
}

if [[ -e "${STATE_DIR}/launched.env" ]]; then
  log "Refusing duplicate launch: ${STATE_DIR}/launched.env already exists."
  exit 3
fi
if pgrep -af 'train_bata_baseline_4gpu_gc04_2epoch|BATA-BASELINE-R32-2E-GC04-4GPU' >/dev/null 2>&1; then
  log "Refusing duplicate launch: a bata_baseline process already exists."
  exit 4
fi

stable_checks=0
log "Watcher armed for GPUs ${GPU_LIST}; threshold=${MAX_USED_MIB}MiB; stable_checks=${REQUIRED_STABLE_CHECKS}."
while true; do
  if all_four_gpus_free; then
    stable_checks=$((stable_checks + 1))
    log "GPU-free check ${stable_checks}/${REQUIRED_STABLE_CHECKS} passed."
    if [[ "${stable_checks}" -ge "${REQUIRED_STABLE_CHECKS}" ]]; then
      break
    fi
  else
    if [[ "${stable_checks}" -ne 0 ]]; then
      log "GPU state became busy again; resetting stable check counter."
    fi
    stable_checks=0
  fi
  sleep "${POLL_SECONDS}"
done

if ! all_four_gpus_free; then
  log "Final GPU check failed; returning to watch loop is required."
  exit 5
fi

run_id="BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-$(date +%Y%m%d-%H%M%S)"
{
  printf 'RUN_ID=%q\n' "${run_id}"
  printf 'LAUNCHED_AT=%q\n' "$(date --iso-8601=seconds)"
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "${GPU_LIST}"
  printf 'LAUNCHER=%q\n' "${LAUNCHER}"
} > "${STATE_DIR}/launched.env"
log "Launching ${run_id}."

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" RUN_ID="${run_id}" bash "${LAUNCHER}"
exit_code=$?
set -e
printf '%s\n' "${exit_code}" > "${STATE_DIR}/exit_code"
log "Run ${run_id} exited with code ${exit_code}."
exit "${exit_code}"
