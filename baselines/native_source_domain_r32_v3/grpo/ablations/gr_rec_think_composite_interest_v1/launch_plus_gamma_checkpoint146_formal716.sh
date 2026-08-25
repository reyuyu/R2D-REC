#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GRPO_ROOT}"

exec torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29146}" \
  -m ablations.gr_rec_think_composite_interest_v1.run_gr_rec_think_composite_interest_v1 \
  --run-id GR-REC-THINK-COMPOSITE-INTEREST-V1-PLUSGAMMA146-BEAMFIRST-FORMAL716-20260826 \
  --max-steps 716 \
  --checkpoint-steps 200 300 400 600 716 \
  --output-dir /root/GRPO-checkpoints \
  --parent-adapter /root/outputs/baselines/native_source_domain_r32_v3/PLUS-GAMMA-R32-2LE-GC04-4GPU-20260826-025731/checkpoint-146 \
  --parent-adapter-sha256 728e602621b4a549d69be118a7adfa4ef3fdb262c4f1376bbb226819a9087a72 \
  --parent-label plus-gamma-r32-2le-checkpoint146
