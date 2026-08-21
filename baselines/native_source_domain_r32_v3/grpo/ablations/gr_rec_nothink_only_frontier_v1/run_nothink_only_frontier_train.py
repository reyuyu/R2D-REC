"""Formal runner for the isolated NoThink-only Frontier ablation."""

from __future__ import annotations

import sys
from pathlib import Path


GRPO_ROOT = Path(__file__).resolve().parents[2]
THINK_PARENT = GRPO_ROOT / "ablations" / "gr_rec_think_exact_clamp_v1"
NOTHINK_PARENT = GRPO_ROOT / "ablations" / "gr_rec_nothink_only_hier_v1"
sys.path.insert(0, str(GRPO_ROOT / "scripts"))
sys.path.insert(0, str(THINK_PARENT))
sys.path.insert(0, str(NOTHINK_PARENT))

import run_grpo_trl_train as baseline_runner
import run_nothink_only_hier_train as parent_runner
from grpo_model import ADAPTER, BASE
from run_think_exact_clamp_train import allow_trusted_same_run_resume

from frontier_trainer import NoThinkOnlyFrontierTrainer


RUN_ID_PREFIX = "GR-REC-NOTHINK-ONLY-FRONTIER-G8BASE-E1-"
FORMAL_RUN_ID = RUN_ID_PREFIX + "20260822"
SMOKE_RUN_ID = "GR-REC-NOTHINK-ONLY-FRONTIER-V1-SMOKE24-20260822"
EXPECTED_BASE = parent_runner.EXPECTED_BASE
EXPECTED_ADAPTER = parent_runner.EXPECTED_ADAPTER
REQUESTED_CHECKPOINT_STEPS = parent_runner.REQUESTED_CHECKPOINT_STEPS
AttributionProbeEvaluator = parent_runner.AttributionProbeEvaluator
FormalCheckpointCallback = parent_runner.FormalCheckpointCallback
TimingEvidenceCallback = parent_runner.TimingEvidenceCallback
formal_checkpoint_steps = parent_runner.formal_checkpoint_steps
build_nothink_only_dataset = parent_runner.build_nothink_only_dataset
audit_nothink_only_sampler = parent_runner.audit_nothink_only_sampler

_ACTIVE_PLAN = None


def prepare_nothink_only_run_plan(args):
    global _ACTIVE_PLAN
    plan = parent_runner.prepare_nothink_only_run_plan(args)
    _ACTIVE_PLAN = plan
    return plan


class _ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_NoThinkOnly_Frontier_v1",
            "parent_experiment": "GR_REC_NoThinkOnly_Hier_v1",
            "runner": (
                "ablations/gr_rec_nothink_only_frontier_v1/"
                "run_nothink_only_frontier_train.py"
            ),
            "experiment_type": "NoThink-only strict-format plus frontier-credit ablation",
            "initialization": "fresh original BATA adapter",
            "resume_parent": None,
            "training_routes": ["no_think"],
            "think_training_rollouts": 0,
            "think_optimizer_updates": 0,
            "training_beam32": False,
            "nothink_route_multiplier": 0.5,
            "nothink_hierarchy": (
                "strict_format_then_D_A_B_C; full_G8_positive_means; "
                "absolute_first_error_frontier; prefix_gated; SUM_reduction"
            ),
            "format_violation": {
                "scalar_reward": -1.0,
                "sequence_advantage_total": -0.09375,
                "length_normalized": True,
            },
            "frontier_penalties": {
                "domain": -0.03125,
                "a": -0.0625,
                "b": -0.1875,
                "c": -0.75,
            },
            "dead_zero_bridge": {"branch": "gold_A_only", "lambda": 0.02},
            "fixed_think_probe": "inference-only; never enters backward or optimizer",
        })
        if _ACTIVE_PLAN is not None:
            payload["formal_checkpoint_steps"] = list(
                formal_checkpoint_steps(_ACTIVE_PLAN["max_steps"])
            )
        return self._writer.write_manifest(payload)


def validate_experiment_args(argv=None):
    args = baseline_runner.build_arg_parser().parse_args(argv)
    if not (
        args.run_id.startswith(RUN_ID_PREFIX)
        or args.run_id == SMOKE_RUN_ID
    ):
        raise ValueError(
            f"--run-id must start with {RUN_ID_PREFIX!r} or equal {SMOKE_RUN_ID!r}"
        )
    if args.run_id == SMOKE_RUN_ID and args.max_steps != 24:
        raise ValueError("the authorized Smoke24 run requires --max-steps 24")
    if str(Path(BASE)) != EXPECTED_BASE or str(Path(ADAPTER)) != EXPECTED_ADAPTER:
        raise RuntimeError("base/adapter constants do not match fresh original BATA")
    if args.resume_from_checkpoint is not None:
        checkpoint = Path(args.resume_from_checkpoint)
        if checkpoint.parent.name != args.run_id:
            raise ValueError("resume is allowed only inside the same Frontier run")
    return args


def main(argv=None):
    args = validate_experiment_args(argv)
    allow_trusted_same_run_resume(args)
    original_monitor_factory = baseline_runner.monitor_from_env

    def monitor_factory(*factory_args, **factory_kwargs):
        return _ManifestWriter(original_monitor_factory(*factory_args, **factory_kwargs))

    def trainer_factory(*trainer_args, **trainer_kwargs):
        reward_funcs = list(trainer_kwargs.get("reward_funcs", ()))
        if not reward_funcs:
            raise RuntimeError("NoThink reward function is missing")
        trainer_kwargs["reward_funcs"] = [reward_funcs[0]]
        trainer = NoThinkOnlyFrontierTrainer(*trainer_args, **trainer_kwargs)
        final_step = _ACTIVE_PLAN["max_steps"]
        trainer.add_callback(FormalCheckpointCallback(final_step))
        trainer.add_callback(TimingEvidenceCallback(final_step, trainer._monitor))
        return trainer

    baseline_runner.prepare_run_plan = prepare_nothink_only_run_plan
    baseline_runner.RecGRPOTrainer = trainer_factory
    baseline_runner.FixedProbeEvaluator = AttributionProbeEvaluator
    baseline_runner.monitor_from_env = monitor_factory
    return baseline_runner.main(argv)


if __name__ == "__main__":
    main()
