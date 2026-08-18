#!/usr/bin/env python3
"""Future DSR pilot runner. It always starts from the BATA adapter."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

import run_grpo_trl_smoke as baseline_smoke
import run_grpo_trl_train as baseline_train

from gr_rec_dsr_v1.dsr_runtime import (
    make_dsr_beam32_fn,
    make_dsr_nothink_reward_func,
    make_dsr_think_reward_func,
    reset_global_capture,
)
from gr_rec_dsr_v1.dsr_contract import enforce_formal_contract
from gr_rec_dsr_v1.dsr_monitor import decorate_dsr_monitor
from gr_rec_dsr_v1.dsr_probe import DsrFixedProbeEvaluator
from gr_rec_dsr_v1.dsr_trainer import DsrGRPOTrainer


BASELINE_BEAM_FACTORY = baseline_smoke.make_beam32_fn
BASELINE_MONITOR_FACTORY = baseline_train.monitor_from_env


def dsr_beam_factory(model, tokenizer, monitor_writer=None):
    return make_dsr_beam32_fn(BASELINE_BEAM_FACTORY, model, tokenizer, monitor_writer)


def dsr_monitor_factory(run_id, rank):
    return decorate_dsr_monitor(
        BASELINE_MONITOR_FACTORY(run_id, rank),
        "ablations/gr_rec_dsr_v1/run_dsr_train.py",
    )


def main(argv=None):
    argv = enforce_formal_contract(sys.argv[1:] if argv is None else argv)
    run_id = argv[argv.index("--run-id") + 1] if "--run-id" in argv else ""
    if not run_id.startswith("GR-REC-DSR-V1-"):
        raise ValueError("DSR --run-id must start with GR-REC-DSR-V1-")
    reset_global_capture()
    baseline_train.RecGRPOTrainer = DsrGRPOTrainer
    baseline_train.make_nothink_reward_func = make_dsr_nothink_reward_func
    baseline_train.make_think_reward_func = make_dsr_think_reward_func
    baseline_train.make_beam32_fn = dsr_beam_factory
    baseline_train.monitor_from_env = dsr_monitor_factory
    baseline_train.FixedProbeEvaluator = DsrFixedProbeEvaluator
    baseline_train.main(argv)


if __name__ == "__main__":
    main()
