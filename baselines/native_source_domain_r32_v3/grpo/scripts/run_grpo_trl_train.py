# -*- coding: utf-8 -*-
"""Formal full-data GRPO runner. It intentionally owns no GRPO math."""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

CURRENT_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(CURRENT_SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(CURRENT_SCRIPTS_DIR))
sys.path.insert(0, str(CURRENT_SCRIPTS_DIR))
import trl_import_fix  # noqa: F401
import grpo_model
from grpo_model import ADAPTER, BASE, load_model
from grpo_probe import (
    FixedProbeCallback,
    FixedProbeEvaluator,
    load_probe_records,
    select_probe_group_ids,
    validate_probe_schedule,
)
from grpo_run_support import (
    audit_sampler,
    count_raw_groups,
    resolve_n_groups,
    validate_resume_checkpoint,
    validate_save_steps,
)
from grpo_trl_trainer import (
    M_NO,
    M_THINK,
    RecGRPOTrainer,
    RouteAwareRepeatSampler,
    build_route_dataset,
    make_nothink_reward_func,
    make_think_reward_func,
)
from monitor.writer import monitor_from_env
from run_grpo_trl_smoke import (
    DATA,
    WORLD_CHUNK,
    current_git_commit,
    make_beam32_fn,
    make_grpo_config,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Formal REC-MP GRPO training")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--n-groups", default="all")
    parser.add_argument("--output-dir", default="/data/GRPO/outputs/formal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--probe-groups", type=int, default=0, choices=(0, 4))
    parser.add_argument("--probe-group-id", action="append", default=[])
    parser.add_argument("--probe-every-steps", type=int, default=200)
    parser.add_argument("--probe-seed", type=int, default=20260818)
    parser.add_argument("--secondary-probe-group-id", action="append", default=[])
    parser.add_argument("--secondary-probe-suite", default=None)
    parser.add_argument("--secondary-probe-every-steps", type=int, default=None)
    parser.add_argument(
        "--probe-only-step",
        type=int,
        default=None,
        help="Load the resume checkpoint as the inference adapter and append only fixed-probe rows.",
    )
    parser.add_argument(
        "--probe-only-adapter",
        default=None,
        help="Optional adapter path for probe-only evaluation (used for the step-0 parent).",
    )
    return parser


def apply_monitor_defaults(run_id):
    defaults = {
        "GRPO_MONITOR": "1",
        "GRPO_DETAILED_MONITOR": "0",
        "GRPO_GENERATION_PROFILE": "0",
        "GRPO_BEAM_RANK_BALANCE": "1",
        "GRPO_TRACE_EVERY": "20",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ["GRPO_RUN_ID"] = run_id


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", run_id):
        raise ValueError("--run-id must contain only letters, digits, '.', '_' or '-'")
    return run_id


def validate_probe_only(args, plan):
    if args.probe_only_step is None:
        return False
    probe_only_adapter = getattr(args, "probe_only_adapter", None)
    adapter_path = probe_only_adapter or args.resume_from_checkpoint
    if adapter_path is None:
        raise ValueError(
            "--probe-only-step requires --resume-from-checkpoint or --probe-only-adapter"
        )
    if probe_only_adapter is None and args.probe_only_step != plan["resume_step"]:
        raise ValueError(
            "--probe-only-step must equal the resume checkpoint global step: "
            f"probe={args.probe_only_step} resume={plan['resume_step']}"
        )
    if not plan["probe_group_ids"]:
        raise ValueError("--probe-only-step requires fixed probe groups")
    return True


def prepare_run_plan(args):
    validate_run_id(args.run_id)
    validate_save_steps(args.save_steps)
    resume_step = validate_resume_checkpoint(args.resume_from_checkpoint)
    if args.save_total_limit < 1:
        raise ValueError("--save-total-limit must be positive")
    raw_groups = count_raw_groups(DATA)
    selected_groups = resolve_n_groups(args.n_groups, raw_groups)
    probe_group_ids = select_probe_group_ids(
        DATA, selected_groups, args.seed, args.probe_groups, args.probe_group_id
    )
    validate_probe_schedule(probe_group_ids, args.probe_every_steps)
    probe_records = load_probe_records(DATA, probe_group_ids) if probe_group_ids else {}
    secondary_probe_group_ids = list(args.secondary_probe_group_id)
    if secondary_probe_group_ids:
        if len(secondary_probe_group_ids) % 4:
            raise ValueError("secondary fixed probe groups must be a multiple of four")
        if not args.secondary_probe_suite:
            raise ValueError("secondary fixed probe groups require --secondary-probe-suite")
    secondary_probe_records = (
        load_probe_records(DATA, secondary_probe_group_ids)
        if secondary_probe_group_ids else {}
    )
    dataset = build_route_dataset(
        DATA, n_groups=selected_groups, seed=args.seed, chunk=8,
        exclude_group_ids=probe_group_ids,
    )
    sampler = RouteAwareRepeatSampler(
        dataset, generation_batch_size=16, repeat_count=2, shuffle=False
    )
    audit = audit_sampler(dataset, sampler)
    max_steps = args.max_steps if args.max_steps is not None else audit["optimizer_steps"]
    if max_steps < 1:
        raise ValueError("--max-steps must be positive")
    output_dir = Path(args.output_dir) / args.run_id
    return {
        "raw_groups": raw_groups,
        "selected_groups_arg": selected_groups,
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "resume_step": resume_step,
        "probe_group_ids": probe_group_ids,
        "probe_records": probe_records,
        "secondary_probe_group_ids": secondary_probe_group_ids,
        "secondary_probe_records": secondary_probe_records,
    }


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    apply_monitor_defaults(args.run_id)
    plan = prepare_run_plan(args)
    probe_only = validate_probe_only(args, plan)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    output_dir = plan["output_dir"]
    monitor_dir = Path(os.environ.get("GRPO_MONITOR_DIR", "/data/GRPO/runs")) / args.run_id
    if args.resume_from_checkpoint is None and not probe_only:
        occupied = (
            (output_dir.exists() and any(output_dir.iterdir()))
            or (monitor_dir.exists() and any(monitor_dir.iterdir()))
        )
        if occupied:
            raise FileExistsError(
                f"run-id {args.run_id!r} already has output or monitor data; use a new run-id"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        print("route schedule summary:", json.dumps(plan["audit"], ensure_ascii=False), flush=True)

    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    if probe_only:
        # PeftModel.from_pretrained reads this module global. Loading the target
        # checkpoint directly avoids constructing/restoring optimizer state and
        # makes backfill strictly inference-only.
        adapter_path = args.probe_only_adapter or args.resume_from_checkpoint
        grpo_model.ADAPTER = str(Path(adapter_path).resolve())
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name.lower()

    def lora_norm():
        squared = 0.0
        for name, parameter in model.named_parameters():
            if "lora" in name.lower():
                squared += (parameter.detach().float() ** 2).sum().item()
        return squared ** 0.5

    def base_checksum():
        for name, parameter in model.named_parameters():
            if "lora" not in name.lower():
                return float(parameter.detach().float().norm())
        return 0.0

    pre_lora = lora_norm()
    pre_base = base_checksum()

    cfg = make_grpo_config(
        str(output_dir), plan["max_steps"], args.lr, args.seed,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    monitor = monitor_from_env(args.run_id, rank)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if monitor.enabled and not probe_only:
        monitor.write_manifest({
            "run_id": args.run_id,
            "start_time": started,
            "git_commit": current_git_commit(),
            "runner": "run_grpo_trl_train.py",
            "runner_args": vars(args),
            "world_size": world,
            "model_path": BASE,
            "adapter_path": ADAPTER,
            "dataset_path": DATA,
            "raw_group_count": plan["raw_groups"],
            "sampled_group_count": plan["audit"]["selected_groups"],
            "sampler_audit": plan["audit"],
            "expected_rollouts": {
                "think": plan["audit"]["think_rollouts"],
                "no_think": plan["audit"]["nothink_rollouts"],
            },
            "expected_optimizer_steps": plan["audit"]["optimizer_steps"],
            "effective_max_steps": plan["max_steps"],
            "fixed_probe": {
                "enabled": bool(plan["probe_group_ids"]),
                "group_ids": plan["probe_group_ids"],
                "every_steps": args.probe_every_steps,
                "seed": args.probe_seed,
                "routes": ["think", "no_think"],
                "excluded_from_training": True,
                "batch_shape": {
                    "think": "4 groups x G=4",
                    "no_think": "2 groups x G=8 (two batches)",
                },
            },
            "secondary_fixed_probe": {
                "enabled": bool(plan["secondary_probe_group_ids"]),
                "group_ids": plan["secondary_probe_group_ids"],
                "suite": args.secondary_probe_suite,
                "every_steps": args.secondary_probe_every_steps or args.probe_every_steps,
                "seed": args.probe_seed,
                "excluded_from_training": False,
            },
            "checkpoint": {
                "save_strategy": "steps",
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "resume_step": plan["resume_step"],
                "boundary": "even global steps (num_iterations=2)",
            },
            "frozen_contract": {
                "think_g": M_THINK,
                "nothink_g": M_NO,
                "temperature": {"think": 0.9, "no_think": 1.0},
                "top_p": {"think": 0.95, "no_think": 1.0},
                "beta": cfg.beta,
                "epsilon": cfg.epsilon,
                "loss_type": cfg.loss_type,
                "max_prompt_length": cfg.max_prompt_length,
                "max_completion_length": cfg.max_completion_length,
                "beam32": {"context_batch": 1, "num_beams": 32,
                           "num_return_sequences": 32, "max_new_tokens": 128},
            },
        })

    beam32_fn = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    trainer = RecGRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=plan["dataset"],
        reward_funcs=[
            make_nothink_reward_func(tokenizer=tokenizer),
            make_think_reward_func(beam32_fn=beam32_fn),
        ],
        monitor_writer=monitor,
    )
    probe_evaluator = None
    if plan["probe_group_ids"]:
        probe_beam32_fn = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
        probe_evaluator = FixedProbeEvaluator(
            trainer=trainer,
            records=plan["probe_records"],
            group_ids=plan["probe_group_ids"],
            beam32_fn=probe_beam32_fn,
            monitor=monitor,
            seed=args.probe_seed,
            every_steps=args.probe_every_steps,
        )
        trainer.add_callback(FixedProbeCallback(probe_evaluator))
    secondary_probe_evaluator = None
    if plan["secondary_probe_group_ids"]:
        secondary_probe_evaluator = FixedProbeEvaluator(
            trainer=trainer,
            records=plan["secondary_probe_records"],
            group_ids=plan["secondary_probe_group_ids"],
            beam32_fn=probe_beam32_fn,
            monitor=monitor,
            seed=args.probe_seed,
            every_steps=args.secondary_probe_every_steps or args.probe_every_steps,
            probe_suite=args.secondary_probe_suite,
        )
        trainer.add_callback(FixedProbeCallback(secondary_probe_evaluator))
    if probe_only:
        probe_evaluator.evaluate(args.probe_only_step, "backfill")
        if secondary_probe_evaluator is not None:
            secondary_probe_evaluator.evaluate(args.probe_only_step, "backfill")
        trainer.accelerator.wait_for_everyone()
        return None
    started_wall = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    post_lora = lora_norm()
    post_base = base_checksum()
    summary = {
        "run_id": args.run_id,
        "rank": rank,
        "world_size": world,
        "global_step": trainer.state.global_step,
        "train_runtime_sec": time.time() - started_wall,
        "train_loss": result.training_loss,
        "lora_norm_pre": pre_lora,
        "lora_norm_post": post_lora,
        "lora_delta": abs(post_lora - pre_lora),
        "base_norm_pre": pre_base,
        "base_norm_post": post_base,
        "base_delta": abs(post_base - pre_base),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(rank) // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(rank) // (1024 * 1024),
        "rollouts": trainer._smoke_log,
        "log_history": trainer.state.log_history,
    }
    with (output_dir / f"run-summary-rank{rank}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if is_main:
        print(json.dumps({
            "run_id": args.run_id,
            "train_runtime_sec": summary["train_runtime_sec"],
            "global_step": trainer.state.global_step,
            "train_loss": result.training_loss,
            "lora_delta": summary["lora_delta"],
            "base_delta": summary["base_delta"],
            "output_dir": str(output_dir),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
