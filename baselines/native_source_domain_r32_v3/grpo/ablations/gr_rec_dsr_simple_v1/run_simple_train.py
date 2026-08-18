#!/usr/bin/env python3
"""Formal DSR-Simple runner; always starts from the original BATA adapter."""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

import run_grpo_trl_smoke as baseline_smoke
import run_grpo_trl_train as baseline_train

from gr_rec_dsr_simple_v1.simple_contract import enforce_simple_contract
from gr_rec_dsr_simple_v1.simple_monitor import decorate_simple_monitor
from gr_rec_dsr_simple_v1.simple_probe import SimpleFixedProbeEvaluator
from gr_rec_dsr_simple_v1.simple_runtime import (
    make_dsr_nothink_reward_func,
    make_simple_beam32_fn,
    make_simple_think_reward_func,
    reset_global_capture,
)
from gr_rec_dsr_simple_v1.simple_trainer import SimpleDsrGRPOTrainer


BASELINE_BEAM_FACTORY = baseline_smoke.make_beam32_fn
BASELINE_MONITOR_FACTORY = baseline_train.monitor_from_env


def simple_beam_factory(model, tokenizer, monitor_writer=None):
    return make_simple_beam32_fn(BASELINE_BEAM_FACTORY, model, tokenizer, monitor_writer)


def simple_monitor_factory(run_id, rank):
    return decorate_simple_monitor(
        BASELINE_MONITOR_FACTORY(run_id, rank),
        "ablations/gr_rec_dsr_simple_v1/run_simple_train.py",
    )


def main(argv=None):
    argv = enforce_simple_contract(sys.argv[1:] if argv is None else argv)
    run_id = argv[argv.index("--run-id") + 1] if "--run-id" in argv else ""
    if not run_id.startswith("GR-REC-DSR-SIMPLE-V1-"):
        raise ValueError("DSR-Simple --run-id must start with GR-REC-DSR-SIMPLE-V1-")
    reset_global_capture()
    baseline_train.RecGRPOTrainer = SimpleDsrGRPOTrainer
    baseline_train.make_nothink_reward_func = make_dsr_nothink_reward_func
    baseline_train.make_think_reward_func = make_simple_think_reward_func
    baseline_train.make_beam32_fn = simple_beam_factory
    baseline_train.monitor_from_env = simple_monitor_factory
    baseline_train.FixedProbeEvaluator = SimpleFixedProbeEvaluator
    baseline_train.main(argv)


if __name__ == "__main__":
    main()
