#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29184}" \
  -m ablations.gr_rec_think_composite_interest_v1.run_gr_rec_think_composite_interest_v1 \
  --run-id GR-REC-THINK-COMPOSITE-INTEREST-V1-BETAGAMMA-E2-BEAMFIRST-300-20260826 \
  --max-steps 300 \
  --checkpoint-steps 200 250 300 \
  --output-dir /root/GRPO-checkpoints \
  --parent-adapter /data/outputs/baselines/native_source_domain_r32_v3/BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127 \
  --parent-adapter-sha256 582e3b2bf1b6c47ce0659f27d0f026ff4c63325c14bec6cf1b22304788406e6a \
  --parent-label beta-gamma-epoch2-step1102
