#!/usr/bin/env python3
"""Formal runner for GR_REC_ThinkSuffixSID_Resample_v1."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = GRPO_ROOT / "scripts"
PARENT_ADAPTER = Path(
    "/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/"
    "REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500"
)
PARENT_ADAPTER_SHA256 = "a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436"
PARENT_RECORDED_SCORE = 1.3510
RUN_ID_PREFIX = "GR-REC-THINK-SUFFIX-SID-G8-REROLL-V1-"

# grpo_model resolves this at import time.
os.environ.setdefault("GRPO_PARENT_ADAPTER", str(PARENT_ADAPTER))
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .suffix_objective import GROUP_SIZE, MAX_RESAMPLE_ROUNDS
from .think_suffix_sid_trainer import (
    ThinkG8SingleGroupSampler,
    ThinkSuffixSIDTrainer,
    audit_think_g8_sampler,
    make_think_suffix_reward_func,
)


_BASE_PREPARE_RUN_PLAN = baseline_runner.prepare_run_plan
_RUNTIME_IMPORT_PROVENANCE = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_adapter() -> dict:
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (PARENT_ADAPTER / name).is_file()]
    if missing:
        raise FileNotFoundError(f"GR_REC_V1_STEP1500_PARENT_INCOMPLETE: {missing}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(
            "GR_REC_V1_STEP1500_PARENT_SHA_MISMATCH: "
            f"expected={PARENT_ADAPTER_SHA256} actual={actual}"
        )
    return {
        "path": str(PARENT_ADAPTER),
        "adapter_sha256": actual,
        "recorded_external_score": PARENT_RECORDED_SCORE,
    }


def build_think_only_dataset(mixed_dataset):
    indices = [
        index for index, row in enumerate(mixed_dataset)
        if row["route"] == "think"
    ]
    dataset = mixed_dataset.select(indices)
    if not dataset or any(row["route"] != "think" for row in dataset):
        raise RuntimeError("Think-only dataset filter produced an invalid route set")
    return dataset


def prepare_think_suffix_run_plan(args):
    plan = _BASE_PREPARE_RUN_PLAN(args)
    if plan["probe_group_ids"]:
        raise ValueError("Think suffix v1 does not enable legacy Beam probes")
    dataset = build_think_only_dataset(plan["dataset"])
    sampler = ThinkG8SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_think_g8_sampler(dataset, sampler)
    max_steps = args.max_steps if args.max_steps is not None else audit["optimizer_steps"]
    if not 1 <= max_steps <= audit["optimizer_steps"]:
        raise ValueError(f"--max-steps must be in [1, {audit['optimizer_steps']}]")
    if max_steps % 2:
        raise ValueError("--max-steps must end on a num_iterations=2 boundary")
    plan["dataset"] = dataset
    plan["audit"] = audit
    plan["max_steps"] = max_steps
    return plan


def make_suffix_grpo_config(
    output_dir, max_steps, lr, seed, *, save_strategy="no",
    save_steps=500, save_total_limit=None, use_cpu=False,
):
    cfg = GRPOConfig(**suffix_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))
    cfg.generation_kwargs = None
    return cfg


def suffix_config_kwargs(**overrides):
    values = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": GROUP_SIZE,
        "generation_batch_size": GROUP_SIZE,
        "max_prompt_length": 8192,
        "max_completion_length": 2048,
        "num_iterations": 2,
        "beta": 0.0,
        "epsilon": 0.2,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "disable_dropout": True,
        "importance_sampling_level": "token",
        "top_entropy_quantile": 1.0,
        "mask_truncated_completions": False,
        "use_vllm": False,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "report_to": "none",
        "temperature": 0.9,
        "top_p": 0.95,
        "generation_kwargs": None,
        "shuffle_dataset": False,
    }
    values.update(overrides)
    return values


class ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        frozen = dict(payload.get("frozen_contract") or {})
        frozen["beam32"] = {"enabled": False}
        payload.update({
            "experiment": "GR_REC_ThinkSuffixSID_Resample_v1",
            "experiment_type": "Think-only CoT-to-SID suffix GRPO",
            "parent_adapter_label": "GR_REC_v1 checkpoint-1500 (recorded 1.3510)",
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_recorded_external_score": PARENT_RECORDED_SCORE,
            "optimizer_initialization": "fresh",
            "training_routes": ["think"],
            "group_size": GROUP_SIZE,
            "temperature": 0.9,
            "top_p": 0.95,
            "reward_formula": "NoThink-v1 final SID hierarchy: -1/-0.25/0/0.5/2/8",
            "advantage_formula": "(R-mean)/population_std+1e-4",
            "loss_scope": "tokens strictly after first </think>",
            "cot_tokens_receive_loss": False,
            "full_completion_attention": True,
            "zero_std_trigger": "reward_population_std == 0 before backward",
            "zero_std_resample_candidates": GROUP_SIZE,
            "zero_std_max_extra_rounds": MAX_RESAMPLE_ROUNDS,
            "zero_std_max_candidate_budget": GROUP_SIZE * (MAX_RESAMPLE_ROUNDS + 1),
            "multi_sid_policy": "monitor_only; reward final complete suffix SID",
            "training_beam32": False,
            "fixed_probe": {"enabled": False, "reason": "legacy probe uses Beam32"},
            "frozen_contract": frozen,
        })
        if _RUNTIME_IMPORT_PROVENANCE:
            payload.update(_RUNTIME_IMPORT_PROVENANCE)
        self._writer.write_manifest(payload)


def suffix_monitor_from_env(run_id, rank):
    return ManifestWriter(base_monitor_from_env(run_id, rank))


def install_experiment_bindings():
    baseline_runner.prepare_run_plan = prepare_think_suffix_run_plan
    baseline_runner.RecGRPOTrainer = ThinkSuffixSIDTrainer
    baseline_runner.make_think_reward_func = make_think_suffix_reward_func
    baseline_runner.make_grpo_config = make_suffix_grpo_config
    baseline_runner.monitor_from_env = suffix_monitor_from_env


def contract_report(max_steps=None) -> dict:
    parent = validate_parent_adapter()
    return {
        "experiment": "GR_REC_ThinkSuffixSID_Resample_v1",
        "parent": parent,
        "group_size": GROUP_SIZE,
        "max_extra_resample_rounds": MAX_RESAMPLE_ROUNDS,
        "max_candidates_per_group": GROUP_SIZE * (MAX_RESAMPLE_ROUNDS + 1),
        "max_steps": max_steps,
        "training_started": False,
    }


def main(argv=None):
    global _RUNTIME_IMPORT_PROVENANCE
    if argv and argv == ["--contract-only"]:
        print(json.dumps(contract_report(), indent=2, sort_keys=True))
        return
    from .runtime_import_provenance import assert_runtime_import_provenance
    _RUNTIME_IMPORT_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    install_experiment_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
