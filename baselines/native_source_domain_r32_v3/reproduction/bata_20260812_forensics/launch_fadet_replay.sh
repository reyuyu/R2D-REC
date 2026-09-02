#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT FADET-A|FADET-B" >&2
  exit 2
fi

RUN_ROOT="$1"
LABEL="$2"
FORENSICS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_DIR="${RUN_ROOT}/fadet_local_gradient_reference"

case "${LABEL}" in
  FADET-A)
    [[ ! -e "${REFERENCE_DIR}" ]] || {
      echo "refusing to overwrite existing FADET gradient reference" >&2
      exit 3
    }
    export BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR="${REFERENCE_DIR}"
    unset BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR
    ;;
  FADET-B)
    [[ -d "${REFERENCE_DIR}" ]] || {
      echo "missing FADET-A gradient reference" >&2
      exit 4
    }
    export BATA_REPLAY_COMPARE_LOCAL_GRAD_REFERENCE_DIR="${REFERENCE_DIR}"
    unset BATA_REPLAY_SAVE_LOCAL_GRAD_REFERENCE_DIR
    ;;
  *)
    echo "label must be FADET-A or FADET-B" >&2
    exit 2
    ;;
esac

# These values are exported before launch_replay starts the Python process,
# imports torch/CUDA, or constructs the model.
export FLASH_ATTENTION_DETERMINISTIC="1"
export BATA_REPLAY_CONFIRM_FA_DETERMINISTIC="1"

exec bash "${FORENSICS_DIR}/launch_replay.sh" "${RUN_ROOT}" "${LABEL}" --deterministic
