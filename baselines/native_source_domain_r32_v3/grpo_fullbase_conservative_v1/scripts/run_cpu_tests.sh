#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEGACY_DIR="$(cd "${PACKAGE_DIR}/../grpo/scripts" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" -m pytest -q "${PACKAGE_DIR}/tests/test_fullbase_contract.py"

for test_file in \
  test_grpo_reward.py \
  test_grpo_trainer.py \
  test_grpo_trl.py \
  test_compute_loss_parity.py \
  test_fixed_probe.py \
  test_formal_runner.py
do
  "${PYTHON_BIN}" "${LEGACY_DIR}/${test_file}"
done
