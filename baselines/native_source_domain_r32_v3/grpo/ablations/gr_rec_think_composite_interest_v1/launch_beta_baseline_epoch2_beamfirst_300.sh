#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29185}" \
  -m ablations.gr_rec_think_composite_interest_v1.run_gr_rec_think_composite_interest_v1 \
  --run-id GR-REC-THINK-COMPOSITE-INTEREST-V1-BETA-BASELINE-E2-BEAMFIRST-300-20260827 \
  --max-steps 300 \
  --checkpoint-steps 200 250 300 \
  --output-dir /root/GRPO-checkpoints \
  --parent-adapter /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  --parent-adapter-sha256 4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3 \
  --parent-label beta-baseline-epoch2-step1106
