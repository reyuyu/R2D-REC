#!/usr/bin/env python3
"""Independent 4-GPU DSR smoke runner, reusing the frozen baseline main."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

import run_grpo_trl_smoke as baseline

from gr_rec_dsr_v1.dsr_runtime import (
    make_dsr_beam32_fn,
    make_dsr_nothink_reward_func,
    make_dsr_think_reward_func,
    reset_global_capture,
)
from gr_rec_dsr_v1.dsr_trainer import DsrGRPOTrainer


BASELINE_BEAM_FACTORY = baseline.make_beam32_fn
BASELINE_MONITOR_FACTORY = baseline.monitor_from_env


def dsr_beam_factory(model, tokenizer, monitor_writer=None):
    return make_dsr_beam32_fn(BASELINE_BEAM_FACTORY, model, tokenizer, monitor_writer)


def dsr_monitor_factory(tag, rank):
    writer = BASELINE_MONITOR_FACTORY(tag, rank)
    original = writer.write_manifest

    def write_manifest(manifest):
        return original({
            **manifest,
            "runner": "ablations/gr_rec_dsr_v1/run_dsr_smoke.py",
            "experiment": "GR_REC_DSR_Ablation_v1",
            "parent": "BATA baseline",
            "baseline": "GR_REC_v1",
            "dsr": {
                "think_lambda": float(os.environ.get("DSR_THINK_LAMBDA", "0.10")),
                "nothink_scale": float(os.environ.get("DSR_NOTHINK_SCALE", "1.0")),
                "extra_forward": False,
                "sampling_changed": False,
            },
        })

    writer.write_manifest = write_manifest
    return writer


def main():
    max_steps = 12
    if "--max-steps" in sys.argv:
        max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1])
    if max_steps < 1 or max_steps > 12:
        raise ValueError("DSR smoke permits 1..12 optimizer steps only")
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "GR-REC-DSR-V1-SMOKE"])
    os.environ.setdefault("GRPO_MONITOR", "1")
    os.environ.setdefault("GRPO_RUN_ID", "GR-REC-DSR-V1-SMOKE-" + time.strftime("%Y%m%d-%H%M%S"))
    os.environ.setdefault("GRPO_DETAILED_MONITOR", "1")
    os.environ.setdefault("GRPO_TRACE_EVERY", "1")
    reset_global_capture()
    baseline.RecGRPOTrainer = DsrGRPOTrainer
    baseline.make_nothink_reward_func = make_dsr_nothink_reward_func
    baseline.make_think_reward_func = make_dsr_think_reward_func
    baseline.make_beam32_fn = dsr_beam_factory
    baseline.monitor_from_env = dsr_monitor_factory
    baseline.main()


if __name__ == "__main__":
    main()
