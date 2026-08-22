#!/usr/bin/env python3
"""Dedicated 12-step entry that reuses the frozen Composite formal runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .provenance import DEFAULT_GRPO, DEFAULT_SOURCE
from .run_gr_rec_think_composite_interest_v1 import (
    EXPERIMENT,
    launch_training,
    prepare_plan,
)
from .smoke12_contract import smoke12_plan


DEFAULT_RUN_ID = "GR-REC-THINK-COMPOSITE-INTEREST-V1-SMOKE12"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{EXPERIMENT} Smoke12")
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument("--output-dir", default="/data/GRPO/outputs/smoke")
    ap.add_argument("--grpo-data", type=Path, default=DEFAULT_GRPO)
    ap.add_argument("--gold-data", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--cpu-plan-only", action="store_true")
    return ap


def smoke_args(namespace):
    namespace.max_steps = 12
    return namespace


def future_launch_command(master_port: int = 29631) -> list[str]:
    return [
        "torchrun", "--nproc_per_node=4", "--master_addr=127.0.0.1",
        f"--master_port={int(master_port)}", "-m",
        "ablations.gr_rec_think_composite_interest_v1."
        "run_gr_rec_think_composite_interest_v1_smoke12",
    ]


def main(argv=None) -> int:
    args = smoke_args(parser().parse_args(argv))
    plan = prepare_plan(args)
    contract = smoke12_plan()
    if plan["train_probe_overlap"] != 0:
        raise RuntimeError("Smoke training cohort overlaps fixed probes")
    if args.cpu_plan_only:
        print(json.dumps({
            "status": "CPU_PLAN_READY",
            "smoke": contract,
            "training_groups": plan["topology"]["training_groups"],
            "train_probe_overlap": plan["train_probe_overlap"],
            "gpu_used": False,
        }))
        return 0
    launch_training(
        args,
        plan,
        enable_probes=False,
        enable_checkpoints=False,
        smoke_mode=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
