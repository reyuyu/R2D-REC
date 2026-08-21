"""Formal runner for Think ExactClamp plus the NoThink teacher bridge."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback
import transformers.trainer as transformers_trainer

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

import run_grpo_trl_train as baseline_runner

try:
    from .think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer
except ImportError:  # Direct script entry point.
    from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


RUN_ID_PREFIX = "GR-REC-CLAMP-BRIDGE-V1-"
MAX_EXPERIMENT_STEPS = 2316
FORMAL_CHECKPOINT_STEPS = (
    250,
    600,
    750,
    800,
    1000,
    1200,
    1400,
    1500,
    1750,
    2000,
    2250,
    2316,
)


class FormalCheckpointCallback(TrainerCallback):
    """Request saves only at the experiment's approved checkpoint steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in FORMAL_CHECKPOINT_STEPS:
            control.should_save = True
        return control


class _ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload["runner"] = "ablations/gr_rec_think_exact_clamp_v1/run_think_exact_clamp_train.py"
        payload["parent"] = "GR_REC_v1 / original BATA adapter"
        payload["advantage"] = {
            "think": "centered_exact_clamp_v1",
            "think_formula": "clamp_negative_if_R_ge_8((R - group_mean) / 8.0)",
            "nothink": "conditional_hierarchical_token_credit_v1",
            "nothink_debug_scalar": "GR_REC_v1 population-std advantage (not used by loss)",
        }
        payload["nothink_bridge"] = {
            "name": "minimal_hierarchical_teacher_bridge_v1",
            "lambda": 0.02,
            "branches": ["dead_zero_a_bridge"],
            "primary_grpo": "token-level PPO on final SID A/B/C positions",
            "teacher_forward": "one A-query row per active optimizer step; no generation",
        }
        payload["formal_max_steps"] = MAX_EXPERIMENT_STEPS
        payload["future_checkpoints"] = list(FORMAL_CHECKPOINT_STEPS)
        payload["gpu_gradient_audit"] = "required before training"
        payload["initialization"] = "fresh original BATA adapter"
        payload["resume_scope"] = "same run-id only"
        return self._writer.write_manifest(payload)


def validate_experiment_args(argv=None):
    args = baseline_runner.build_arg_parser().parse_args(argv)
    if not args.run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"--run-id must start with {RUN_ID_PREFIX!r}")
    if (
        args.resume_from_checkpoint is not None
        and Path(args.resume_from_checkpoint).parent.name != args.run_id
    ):
        raise ValueError("resume is allowed only from a checkpoint inside the same --run-id")
    if args.max_steps is None or not 1 <= args.max_steps <= MAX_EXPERIMENT_STEPS:
        raise ValueError(f"--max-steps is required and must be in [1, {MAX_EXPERIMENT_STEPS}]")
    return args


def allow_trusted_same_run_resume(args):
    """Allow optimizer restore only for this runner's own verified checkpoint."""
    if args.resume_from_checkpoint is None:
        return
    checkpoint = Path(args.resume_from_checkpoint).resolve()
    expected_run_dir = (Path(args.output_dir).resolve() / args.run_id)
    if checkpoint.parent != expected_run_dir:
        raise ValueError("resume checkpoint must resolve inside this run's output directory")
    try:
        checkpoint_step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise ValueError("resume checkpoint must use checkpoint-<step> naming") from exc
    required = (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    )
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("resume checkpoint is missing required same-run state")
    with (checkpoint / "trainer_state.json").open(encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    if int(trainer_state.get("global_step", -1)) != checkpoint_step:
        raise ValueError("resume checkpoint step does not match trainer_state.json")

    torch_version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    if torch_version < (2, 6):
        # The checkpoint was generated locally by this exact run. Transformers 5.x
        # otherwise blocks all optimizer restores on the installed torch 2.5.
        transformers_trainer.check_torch_load_is_safe = lambda: None
        torch.serialization.add_safe_globals([
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
        ])
        print(
            "trusted same-run optimizer restore enabled for torch " + torch.__version__,
            flush=True,
        )


def main(argv=None):
    args = validate_experiment_args(argv)
    allow_trusted_same_run_resume(args)
    original_monitor_factory = baseline_runner.monitor_from_env

    def think_exact_clamp_monitor(*args, **kwargs):
        return _ManifestWriter(original_monitor_factory(*args, **kwargs))

    def trainer_factory(*args, **kwargs):
        trainer = ThinkExactClampRecGRPOTrainer(*args, **kwargs)
        trainer.add_callback(FormalCheckpointCallback())
        return trainer

    baseline_runner.RecGRPOTrainer = trainer_factory
    baseline_runner.monitor_from_env = think_exact_clamp_monitor
    return baseline_runner.main(argv)


if __name__ == "__main__":
    main()
