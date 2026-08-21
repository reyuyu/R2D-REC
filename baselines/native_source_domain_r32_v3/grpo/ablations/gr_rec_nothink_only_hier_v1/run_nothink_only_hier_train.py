"""Formal NoThink-only attribution runner using the frozen joint-run trainer math."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


GRPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_ABLATION = GRPO_ROOT / "ablations" / "gr_rec_think_exact_clamp_v1"
sys.path.insert(0, str(GRPO_ROOT / "scripts"))
sys.path.insert(0, str(PARENT_ABLATION))

import run_grpo_trl_train as baseline_runner
from grpo_model import ADAPTER, BASE
from grpo_probe import FixedProbeEvaluator as ParentFixedProbeEvaluator
from grpo_trl_trainer import ROUTE_ID, RouteAwareRepeatSampler
from run_think_exact_clamp_train import allow_trusted_same_run_resume
from think_diagnostics import interest_diagnostics
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


RUN_ID_PREFIX = "GR-REC-NOTHINK-ONLY-HIER-G8BASE-E1-"
FORMAL_RUN_ID = RUN_ID_PREFIX + "20260821"
EXPECTED_BASE = "/data/models/onereason-8b-pretrain-competition"
EXPECTED_ADAPTER = (
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333"
)
REQUESTED_CHECKPOINT_STEPS = (250, 500, 666, 750, 1000, 1250)
TIMING_STEPS = (50, 100)

_JOINT_PREPARE_RUN_PLAN = baseline_runner.prepare_run_plan
_ACTIVE_PLAN = None


def build_nothink_only_dataset(mixed_dataset, exclude_group_ids=()):
    excluded = set(exclude_group_ids)
    indices = [
        index for index, row in enumerate(mixed_dataset)
        if row["route"] == "no_think"
        and row["recommendation_group_id"] not in excluded
    ]
    dataset = mixed_dataset.select(indices)
    if not dataset or any(row["route"] != "no_think" for row in dataset):
        raise RuntimeError("NoThink-only dataset filter produced an invalid route set")
    return dataset


def audit_nothink_only_sampler(dataset, sampler):
    rows = list(dataset)
    selected = {row["recommendation_group_id"] for row in rows}
    if len(selected) != len(rows):
        raise RuntimeError("NoThink-only dataset must contain one row per training group")
    covered = set()
    rollout_group_ids = []
    for route, indices in sampler._chunks:
        if route != "no_think":
            raise RuntimeError(f"unexpected training route in sampler: {route!r}")
        group_ids = [rows[index]["recommendation_group_id"] for index in indices]
        if len(group_ids) != 2 or len(set(group_ids)) != 2:
            raise RuntimeError("each NoThink global rollout must contain two unique G8 groups")
        rollout_group_ids.append(group_ids)
        covered.update(group_ids)
    dropped = sorted(selected - covered)
    if dropped:
        raise RuntimeError(f"sampler dropped NoThink training groups: {dropped}")
    flattened = [group_id for pair in rollout_group_ids for group_id in pair]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("a NoThink group appears in more than one fresh rollout")
    optimizer_steps = len(rollout_group_ids) * sampler.repeat_count
    if optimizer_steps % sampler.repeat_count:
        raise RuntimeError("optimizer steps do not end on a rollout boundary")
    return {
        "selected_groups": len(selected),
        "trained_groups": len(covered),
        "dropped_groups": 0,
        "dropped_group_ids": [],
        "think_unique_groups": 0,
        "nothink_unique_groups": len(covered),
        "think_rollouts": 0,
        "nothink_rollouts": len(rollout_group_ids),
        "fresh_rollout_count": len(rollout_group_ids),
        "unique_groups_per_global_rollout": 2,
        "optimizer_steps": optimizer_steps,
        "nothink_optimizer_steps": optimizer_steps,
        "think_optimizer_steps": 0,
        "repeat_count": sampler.repeat_count,
        "num_iterations": sampler.repeat_count,
        "route_schedule_preview": ["no_think"] * min(24, len(rollout_group_ids)),
        "rollout_group_ids_preview": rollout_group_ids[:8],
    }


def prepare_nothink_only_run_plan(args):
    global _ACTIVE_PLAN
    plan = _JOINT_PREPARE_RUN_PLAN(args)
    # The joint parent selected 1545 post-probe groups but its fixed chunk
    # topology explicitly dropped one tail group. Reuse that audited exclusion
    # so both experiments train the exact same 1544-group cohort.
    joint_tail_exclusions = tuple(plan["audit"]["dropped_group_ids"])
    post_probe_candidate_groups = plan["audit"]["selected_groups"]
    dataset = build_nothink_only_dataset(
        plan["dataset"], exclude_group_ids=joint_tail_exclusions
    )
    sampler = RouteAwareRepeatSampler(
        dataset, generation_batch_size=16, repeat_count=2, shuffle=False
    )
    audit = audit_nothink_only_sampler(dataset, sampler)
    audit["post_probe_candidate_groups"] = post_probe_candidate_groups
    audit["joint_sampler_parity_excluded_groups"] = len(joint_tail_exclusions)
    audit["joint_sampler_parity_excluded_group_ids"] = list(joint_tail_exclusions)
    max_steps = args.max_steps if args.max_steps is not None else audit["optimizer_steps"]
    if max_steps < 1 or max_steps > audit["optimizer_steps"]:
        raise ValueError(
            f"--max-steps must be in [1, {audit['optimizer_steps']}] for this sampler"
        )
    if max_steps % audit["num_iterations"]:
        raise ValueError("--max-steps must end on a num_iterations=2 rollout boundary")
    plan["dataset"] = dataset
    plan["audit"] = audit
    plan["max_steps"] = max_steps
    _ACTIVE_PLAN = plan
    return plan


def formal_checkpoint_steps(final_step):
    steps = {step for step in REQUESTED_CHECKPOINT_STEPS if step <= final_step}
    steps.add(int(final_step))
    if any(step % 2 for step in steps):
        raise ValueError("all checkpoints must lie on num_iterations=2 boundaries")
    return tuple(sorted(steps))


class FormalCheckpointCallback(TrainerCallback):
    def __init__(self, final_step):
        self.steps = formal_checkpoint_steps(final_step)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in self.steps:
            control.should_save = True
        return control


class TimingEvidenceCallback(TrainerCallback):
    def __init__(self, final_step, monitor):
        self.final_step = int(final_step)
        self.monitor = monitor
        self.started = None
        self.start_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.start_step = int(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in TIMING_STEPS or self.started is None:
            return control
        completed = step - self.start_step
        elapsed = time.monotonic() - self.started
        sec_per_step = elapsed / max(completed, 1)
        event = {
            "type": "runtime_eta",
            "step": step,
            "elapsed_wall_sec": elapsed,
            "mean_sec_per_optimizer_step": sec_per_step,
            "eta_to_full_epoch_sec": max(self.final_step - step, 0) * sec_per_step,
            "full_epoch_step": self.final_step,
        }
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print("NOTHINK_ONLY_TIMING " + json.dumps(event, sort_keys=True), flush=True)
            if self.monitor is not None and self.monitor.enabled:
                self.monitor._append("timing.jsonl", event)
        return control


class NoThinkOnlyHierTrainer(ThinkExactClampRecGRPOTrainer):
    """Route guards only; all NoThink reward and loss math remains inherited."""

    def _calculate_rewards(self, inputs, *args, **kwargs):
        if not inputs or any(item.get("route") != "no_think" for item in inputs):
            raise RuntimeError("NoThink-only trainer received a non-NoThink reward batch")
        return super()._calculate_rewards(inputs, *args, **kwargs)

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        route_ids = inputs["route_id"]
        if not bool((route_ids == ROUTE_ID["no_think"]).all()):
            raise RuntimeError("NoThink-only trainer received a non-NoThink loss batch")
        return super()._compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class AttributionProbeEvaluator(ParentFixedProbeEvaluator):
    """Attach monitor-only interest diagnostics to immutable Think probes."""

    def _think(self):
        result = super()._think()
        for candidate in result["candidates"]:
            candidate.update(
                interest_diagnostics(candidate["completion"], result["prompt"])
            )
        return result


class _ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_NoThinkOnly_Hier_v1",
            "runner": (
                "ablations/gr_rec_nothink_only_hier_v1/"
                "run_nothink_only_hier_train.py"
            ),
            "experiment_type": "NoThink-only RL attribution ablation",
            "initialization": "fresh original BATA adapter",
            "resume_parent": None,
            "training_routes": ["no_think"],
            "think_training_rollouts": 0,
            "think_optimizer_updates": 0,
            "training_beam32": False,
            "nothink_route_multiplier": 0.5,
            "nothink_hierarchy": (
                "earliest_domain_commitment_then_A_B_C; "
                "full_G8_milestone_baselines; prefix_gated; SUM_reduction"
            ),
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
    if not args.run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"--run-id must start with {RUN_ID_PREFIX!r}")
    if str(Path(BASE)) != EXPECTED_BASE or str(Path(ADAPTER)) != EXPECTED_ADAPTER:
        raise RuntimeError("base/adapter constants do not match fresh original BATA")
    if args.resume_from_checkpoint is not None:
        checkpoint = Path(args.resume_from_checkpoint)
        if checkpoint.parent.name != args.run_id:
            raise ValueError("resume is allowed only inside the same NoThink-only run")
    return args


def main(argv=None):
    args = validate_experiment_args(argv)
    allow_trusted_same_run_resume(args)
    original_monitor_factory = baseline_runner.monitor_from_env

    def monitor_factory(*factory_args, **factory_kwargs):
        return _ManifestWriter(original_monitor_factory(*factory_args, **factory_kwargs))

    def trainer_factory(*trainer_args, **trainer_kwargs):
        reward_funcs = list(trainer_kwargs.get("reward_funcs", ()))
        if len(reward_funcs) < 1:
            raise RuntimeError("NoThink reward function is missing")
        trainer_kwargs["reward_funcs"] = [reward_funcs[0]]
        trainer = NoThinkOnlyHierTrainer(*trainer_args, **trainer_kwargs)
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
