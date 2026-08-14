#!/usr/bin/env bash
set -euo pipefail
export RUN_ID=ALPHA-MINI-V3-R32-2E-GC04-4GPU-20260815-RETRY1
exec bash /data/baselines/native_source_domain_r32_v3/scripts/launch_alpha_mini_v3_4gpu_gc04_2epoch.sh
