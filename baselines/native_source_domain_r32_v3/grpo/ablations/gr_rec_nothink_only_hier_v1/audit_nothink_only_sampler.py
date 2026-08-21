"""CPU-only sampler audit for the NoThink-only full-epoch attribution run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_nothink_only_hier_train import (
    FORMAL_RUN_ID,
    formal_checkpoint_steps,
    prepare_nothink_only_run_plan,
    validate_experiment_args,
    baseline_runner,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--probe-seed", type=int, default=20260818)
    parser.add_argument("--probe-groups", type=int, default=4, choices=(4,))
    args = parser.parse_args(argv)
    runner_args = validate_experiment_args([
        "--run-id", FORMAL_RUN_ID,
        "--n-groups", "all",
        "--seed", str(args.seed),
        "--probe-seed", str(args.probe_seed),
        "--probe-groups", str(args.probe_groups),
        "--save-steps", "100000",
        "--save-total-limit", "7",
    ])
    plan = prepare_nothink_only_run_plan(runner_args)
    payload = {
        "experiment": "GR_REC_NoThinkOnly_Hier_v1",
        "run_id": FORMAL_RUN_ID,
        "gpu_used": False,
        "dataset_path": baseline_runner.DATA,
        "raw_group_count": plan["raw_groups"],
        "probe_group_ids": plan["probe_group_ids"],
        "probe_group_count": len(plan["probe_group_ids"]),
        "sampler_audit": plan["audit"],
        "derived_full_epoch_step": plan["max_steps"],
        "formal_checkpoint_steps": list(formal_checkpoint_steps(plan["max_steps"])),
        "training_routes": ["no_think"],
        "think_optimizer_updates": 0,
        "route_multiplier": 0.5,
        "num_iterations": 2,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
