#!/usr/bin/env python3
"""Independent 4-GPU DSR-Simple smoke runner, capped at 12 steps."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

import run_grpo_trl_smoke as baseline

from gr_rec_dsr_simple_v1.simple_monitor import decorate_simple_monitor
from gr_rec_dsr_simple_v1.simple_runtime import (
    make_dsr_nothink_reward_func,
    make_simple_beam32_fn,
    make_simple_think_reward_func,
    reset_global_capture,
)
from gr_rec_dsr_simple_v1.simple_trainer import SimpleDsrGRPOTrainer


BASELINE_BEAM_FACTORY = baseline.make_beam32_fn
BASELINE_MONITOR_FACTORY = baseline.monitor_from_env


def simple_beam_factory(model, tokenizer, monitor_writer=None):
    return make_simple_beam32_fn(BASELINE_BEAM_FACTORY, model, tokenizer, monitor_writer)


def simple_monitor_factory(tag, rank):
    return decorate_simple_monitor(
        BASELINE_MONITOR_FACTORY(tag, rank),
        "ablations/gr_rec_dsr_simple_v1/run_simple_smoke.py",
    )


def main():
    max_steps = 12
    if "--max-steps" in sys.argv:
        max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1])
    if max_steps < 1 or max_steps > 12:
        raise ValueError("DSR-Simple smoke permits 1..12 optimizer steps only")
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "GR-REC-DSR-SIMPLE-V1-SMOKE"])
    os.environ.setdefault("GRPO_MONITOR", "1")
    os.environ.setdefault(
        "GRPO_RUN_ID",
        "GR-REC-DSR-SIMPLE-V1-SMOKE-" + time.strftime("%Y%m%d-%H%M%S"),
    )
    os.environ.setdefault("GRPO_DETAILED_MONITOR", "1")
    os.environ.setdefault("GRPO_TRACE_EVERY", "1")
    reset_global_capture()
    baseline.RecGRPOTrainer = SimpleDsrGRPOTrainer
    baseline.make_nothink_reward_func = make_dsr_nothink_reward_func
    baseline.make_think_reward_func = make_simple_think_reward_func
    baseline.make_beam32_fn = simple_beam_factory
    baseline.monitor_from_env = simple_monitor_factory
    baseline.main()


if __name__ == "__main__":
    main()
