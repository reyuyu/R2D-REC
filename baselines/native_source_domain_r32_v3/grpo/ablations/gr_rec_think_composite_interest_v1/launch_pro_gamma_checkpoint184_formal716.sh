#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29684}" \
  -m ablations.gr_rec_think_composite_interest_v1.run_gr_rec_think_composite_interest_v1 \
  --run-id GR-REC-THINK-COMPOSITE-INTEREST-V1-PROGAMMA184-BEAMFIRST-FORMAL716-20260826 \
  --max-steps 716 \
  --checkpoint-steps 200 300 400 600 716 \
  --output-dir /root/GRPO-checkpoints \
  --parent-adapter /data/outputs/baselines/native_source_domain_r32_v3/PRO-GAMMA-R32-2E-GC04-4GPU-RETRY1-20260825-220900/checkpoint-184 \
  --parent-adapter-sha256 f79018111254eda26d7739fd329afc17a0c62b05e103c29afad5fe1b2b5618df \
  --parent-label pro-gamma-r32-e2-checkpoint184
