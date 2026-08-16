#!/usr/bin/env bash
# Queue Mini-Fix-E3 only after the named U3K run has completed successfully.
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/data/baselines/native_source_domain_r32_v3}"
VENV_ROOT="${VENV_ROOT:-/data/venvs/llamafactory-01398eb-liger081}"
UPSTREAM_RUN_ID="${UPSTREAM_RUN_ID:?UPSTREAM_RUN_ID is required}"
UPSTREAM_PID="${UPSTREAM_PID:?UPSTREAM_PID is required}"
UPSTREAM_OUTPUT="/data/outputs/baselines/native_source_domain_r32_v3/${UPSTREAM_RUN_ID}"
UPSTREAM_LOG="/data/logs/baselines/native_source_domain_r32_v3/${UPSTREAM_RUN_ID}/train.log"
LOG_BASE="/data/logs/baselines/native_source_domain_r32_v3"
QUEUE_DIR="${LOG_BASE}/queued-${UPSTREAM_RUN_ID}-to-mini-fix-e3"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOG_BASE}"
log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

upstream_succeeded() {
  test -f "${UPSTREAM_OUTPUT}/train_results.json"
  test -f "${UPSTREAM_OUTPUT}/trainer_state.json"
  grep -q '^\*\*\*\*\* train metrics \*\*\*\*\*$' "${UPSTREAM_LOG}"
  "${VENV_ROOT}/bin/python" - "${UPSTREAM_OUTPUT}/trainer_state.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
if float(state.get("epoch", 0.0)) < 1.999:
    raise SystemExit("U3K final trainer state is below two epochs")
PY
}

while kill -0 "${UPSTREAM_PID}" 2>/dev/null; do
  log "waiting for U3K pid=${UPSTREAM_PID} run=${UPSTREAM_RUN_ID}"
  sleep "${POLL_SECONDS}"
done

if ! upstream_succeeded; then
  log "U3K exited without the required successful-completion markers; Mini-Fix-E3 will not start."
  exit 1
fi

if ! mkdir "${QUEUE_DIR}" 2>/dev/null; then
  log "queue lock already exists at ${QUEUE_DIR}; refusing duplicate Mini-Fix-E3 launch."
  exit 0
fi

E3_RUN_ID="MINI-FIX-E3-R32-3E-GC04-4GPU-$(date +%Y%m%d-%H%M%S)"
E3_LAUNCHER_LOG="${LOG_BASE}/${E3_RUN_ID}.launcher.log"
log "U3K success verified; launching ${E3_RUN_ID}."
nohup env RUN_ID="${E3_RUN_ID}" bash "${BASELINE_ROOT}/scripts/launch_mini_fix_e3_4gpu_gc04.sh" \
  > "${E3_LAUNCHER_LOG}" 2>&1 < /dev/null &
E3_PID=$!
printf 'upstream_run_id=%s\nupstream_pid=%s\ne3_run_id=%s\ne3_pid=%s\nstarted_at=%s\n' \
  "${UPSTREAM_RUN_ID}" "${UPSTREAM_PID}" "${E3_RUN_ID}" "${E3_PID}" "$(date -Is)" \
  > "${QUEUE_DIR}/launch_status.env"
log "Mini-Fix-E3 launched pid=${E3_PID}; launcher_log=${E3_LAUNCHER_LOG}"
