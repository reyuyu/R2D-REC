#!/usr/bin/env python3
"""Think-only Composite Interest runner; dry-run never loads a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = GRPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from .data_adapter import build_think_composite_dataset
from .provenance import DEFAULT_GRPO, DEFAULT_SOURCE, load_gold, load_think_groups
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units

EXPERIMENT = "GR_REC_Think_CompositeInterest_v1"
BASE_MAIN_SHA = "7b8005aebe8610c0824624be344cececebceaaba"
SEED = 20260818
PROBE_IDS = (
    "662e21149595c33240d7ec0baed2282d687f71e1a0983252f28edc9c18855cbc",
    "9d9852a42bddfd801f85ee92a72da13a2145f3e3eb9e80cab93bd248ac428559",
    "994c3186e677aec015938e71f08257afcf9f235a5a35df24af7fa67849c752d5",
    "6aa980e4a7c35366d33d2c2f61a28def8e24b906b9fa90fe1c673b03cc8cbdc2",
)
CHECKPOINT_STEPS = (100, 200, 250, 300, 350, 400, 450, 500, 600, 720)
RESULT_PATH = GRPO_ROOT / "results/gr_rec_think_composite_interest_v1_runner_dry_run_20260822.json"
AUTO_SAVE_STEPS = 10000
SAVE_TOTAL_LIMIT = len(CHECKPOINT_STEPS)


def frozen_contract():
    return {
        "route": "think_only", "g": 4, "temperature": 0.9, "top_p": 0.95,
        "learning_rate": 1e-6, "beta": 0.0, "epsilon": 0.2,
        "loss_type": "grpo", "num_iterations": 2,
        "steps_per_generation": 1, "route_multiplier": 1.0,
        "stop_token": "</think>", "beam_num_beams": 32,
        "reward": "0.60*U_beam+0.40*U_cot",
        "advantage": "(R-mean)/(population_std+1e-4); correction=0",
    }


def checkpoint_save_config():
    return {"save_strategy": "steps", "save_steps": AUTO_SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT}


def should_save_checkpoint(step):
    return int(step) in CHECKPOINT_STEPS


def parser():
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--output-dir", default="/data/GRPO/outputs/formal")
    ap.add_argument("--run-id", default="GR-REC-THINK-COMPOSITE-INTEREST-V1")
    ap.add_argument("--grpo-data", type=Path, default=DEFAULT_GRPO)
    ap.add_argument("--gold-data", type=Path, default=DEFAULT_SOURCE)
    return ap


def validate_args(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", args.run_id):
        raise ValueError("invalid --run-id")
    if not 1 <= args.max_steps <= 720:
        raise ValueError("--max-steps must be within 1..720")


def prepare_plan(args):
    validate_args(args)
    groups = load_think_groups(args.grpo_data)
    gold, _, ambiguous = load_gold(args.gold_data, set(groups))
    if ambiguous:
        raise RuntimeError("ambiguous Gold CoT provenance")
    eligible = {
        group_id for group_id, cot in gold.items()
        if (parsed := extract_interest_units(cot, groups[group_id]["prompt"])).parser_success
        and parsed.units
    }
    if (len(groups), len(gold), len(eligible)) != (1549, 1458, 1446):
        raise RuntimeError("audited cohort drift")
    if not set(PROBE_IDS).issubset(eligible):
        raise RuntimeError("fixed probe is absent from eligible cohort")
    ordered = sorted(groups)
    random.Random(SEED).shuffle(ordered)
    training_ids = [group_id for group_id in ordered
                    if group_id in eligible and group_id not in PROBE_IDS]
    records = build_think_composite_dataset(
        [groups[group_id] for group_id in training_ids], gold,
        eligible_group_ids=training_ids,
    )
    dropped_ids = training_ids[-(len(training_ids) % 4):]
    topology = {
        "original_think_groups": len(groups),
        "exact_gold_join_groups": len(gold),
        "parser_valid_gold_groups": len(eligible),
        "fixed_probe_groups": len(PROBE_IDS),
        "post_probe_groups": len(records),
        "sampler_drop_groups": len(dropped_ids),
        "training_groups": len(records) - len(dropped_ids),
        "fresh_g4_rollouts": (len(records) - len(dropped_ids)) // 4,
        "optimizer_steps_num_iterations_2": ((len(records) - len(dropped_ids)) // 4) * 2,
    }
    expected = {
        "post_probe_groups": 1442, "sampler_drop_groups": 2,
        "training_groups": 1440, "fresh_g4_rollouts": 360,
        "optimizer_steps_num_iterations_2": 720,
    }
    if any(topology[key] != value for key, value in expected.items()):
        raise RuntimeError("G4 topology drift")
    return {
        "records": records,
        "probe_records": {group_id: {"think": groups[group_id], "gold_cot": gold[group_id]}
                          for group_id in PROBE_IDS},
        "probe_ids": list(PROBE_IDS),
        "dropped_group_ids": dropped_ids,
        "topology": topology,
        "output_dir": str(Path(args.output_dir) / args.run_id),
    }


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=GRPO_ROOT.parents[2], text=True
    ).strip()


def dry_run_report(args, plan):
    from grpo_model import ADAPTER, BASE
    report = {
        "experiment": EXPERIMENT,
        "status": "CPU_DRY_RUN_PASS",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_main_sha": BASE_MAIN_SHA,
        "inspected_head": git_head(),
        "gpu_used": False,
        "model_loaded": False,
        "generation_called": False,
        "optimizer_step_called": False,
        "training_started": False,
        "paths": {"base_model": BASE, "fresh_original_bata_adapter": ADAPTER,
                  "grpo_data": str(args.grpo_data), "gold_data": str(args.gold_data),
                  "future_output_dir": plan["output_dir"]},
        "topology": plan["topology"],
        "seed": SEED,
        "fixed_probe_group_ids": plan["probe_ids"],
        "fixed_probes_excluded_from_training": True,
        "sampler_dropped_group_ids": plan["dropped_group_ids"],
        "effective_max_steps": args.max_steps,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "frozen_contract": frozen_contract(),
        "gold_cot_contract": "reward_only; absent from prompt/input_ids/generation/beam",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def launch_training(args, plan):
    """Future GPU entry point. It is unreachable from --dry-run."""
    import torch
    from datasets import Dataset
    from grpo_model import ADAPTER, BASE, load_model
    from grpo_probe import FixedProbeCallback
    from grpo_trl_trainer import make_think_reward_func
    from monitor.writer import monitor_from_env
    from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config
    from transformers import TrainerCallback
    from .composite_probe import CompositeThinkProbeEvaluator
    from .composite_trainer import ThinkCompositeInterestRecGRPOTrainer

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["GRPO_RUN_ID"] = args.run_id
    os.environ.setdefault("GRPO_MONITOR", "1")
    if not Path(BASE).exists() or not Path(ADAPTER).exists():
        raise FileNotFoundError("fresh original base/BATA path missing")
    torch.cuda.set_device(rank)
    torch.manual_seed(SEED + rank)
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name.lower()
    cfg = make_grpo_config(
        plan["output_dir"], args.max_steps, 1e-6, SEED, **checkpoint_save_config(),
    )
    monitor = monitor_from_env(args.run_id, rank)
    if monitor.enabled:
        monitor.write_manifest({
            "run_id": args.run_id, "experiment": EXPERIMENT,
            "git_commit": git_head(), "runner": Path(__file__).name,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "model_path": BASE, "adapter_path": ADAPTER,
            "fresh_original_bata": True, "dataset_path": str(args.grpo_data),
            "gold_source_path": str(args.gold_data), "gold_cot_reward_only": True,
            "sampler_audit": plan["topology"], "fixed_probe_ids": plan["probe_ids"],
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "frozen_contract": frozen_contract(),
        })
    beam32 = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    trainer = ThinkCompositeInterestRecGRPOTrainer(
        model=model, args=cfg, processing_class=tokenizer,
        train_dataset=Dataset.from_list(plan["records"]),
        reward_funcs=[make_think_reward_func(beam32_fn=beam32)],
        monitor_writer=monitor,
    )

    class MilestoneSaveCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if should_save_checkpoint(state.global_step):
                control.should_save = True
            return control

    trainer.add_callback(MilestoneSaveCallback())
    probe_beam32 = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    probe_evaluator = CompositeThinkProbeEvaluator(
        trainer=trainer,
        records=plan["probe_records"],
        group_ids=plan["probe_ids"],
        beam32_fn=probe_beam32,
        monitor=monitor,
        seed=SEED,
        every_steps=100,
    )
    trainer.add_callback(FixedProbeCallback(probe_evaluator))
    trainer.train()


def main(argv=None):
    args = parser().parse_args(argv)
    plan = prepare_plan(args)
    if args.dry_run:
        report = dry_run_report(args, plan)
        print(json.dumps({"status": report["status"], "topology": report["topology"]}, ensure_ascii=False))
        return 0
    launch_training(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
