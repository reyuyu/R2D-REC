#!/usr/bin/env python3
"""Formal runner for Think G4 + per-CoT Beam8 dual-objective GRPO."""
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
EXPECTED_RAW_GROUPS = 1549
EXPECTED_TRAIN_GROUPS = 1545
EXPECTED_FORMAL_STEPS = 3090
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)

# Fail closed on the intended immutable parent before importing grpo_model.
os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .dual_beam8_trainer import (
    COT_G,
    SID_G,
    DualBeam8Runtime,
    ThinkDualBeam8Trainer,
    ThinkG4SingleGroupSampler,
    audit_dual_sampler,
    make_dual_beam8_reward_func,
)
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import (
    assert_runtime_import_provenance,
)

_BASE_PREPARE_RUN_PLAN = baseline_runner.prepare_run_plan
_BASE_LOAD_MODEL = baseline_runner.load_model
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_RUNTIME_PROVENANCE = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_adapter():
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (PARENT_ADAPTER / name).is_file()]
    if missing:
        raise FileNotFoundError(f"DUAL_BEAM8_PARENT_INCOMPLETE: {missing}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(
            f"DUAL_BEAM8_PARENT_SHA_MISMATCH expected={PARENT_ADAPTER_SHA256} actual={actual}"
        )
    return {
        "path": str(PARENT_ADAPTER),
        "adapter_sha256": actual,
        "recorded_external_score": PARENT_RECORDED_SCORE,
    }


def load_model_and_capture_runtime(*args, **kwargs):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER
    result = _BASE_LOAD_MODEL(*args, **kwargs)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER = result[0], result[1]
    return result


def make_runtime_dual_reward(beam32_fn=None):
    # The baseline runner still constructs its historical training Beam32 helper;
    # this ablation intentionally ignores it. Probe4 keeps a separate Beam32 path.
    del beam32_fn
    if _RUNTIME_MODEL is None or _RUNTIME_TOKENIZER is None:
        raise RuntimeError("DUAL_BEAM8_RUNTIME_MODEL_NOT_CAPTURED")
    runtime = DualBeam8Runtime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_dual_beam8_reward_func(runtime)


def prepare_dual_run_plan(args):
    # Keep the original fixed Probe4 selection and exclusion logic intact.
    plan = _BASE_PREPARE_RUN_PLAN(args)
    if plan["raw_groups"] != EXPECTED_RAW_GROUPS:
        raise RuntimeError(f"DUAL_BEAM8_RAW_TOPOLOGY_DRIFT: {plan['raw_groups']}")
    if tuple(plan["probe_group_ids"]) != FIXED_PROBE4_IDS:
        raise RuntimeError(
            f"DUAL_BEAM8_PROBE4_ID_DRIFT expected={FIXED_PROBE4_IDS} "
            f"actual={tuple(plan['probe_group_ids'])}"
        )
    think_indices = [
        index for index, row in enumerate(plan["dataset"])
        if row["route"] == "think"
    ]
    dataset = plan["dataset"].select(think_indices)
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_dual_sampler(dataset, sampler)
    if audit["trained_groups"] != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError(f"DUAL_BEAM8_TRAIN_TOPOLOGY_DRIFT: {audit['trained_groups']}")
    if set(plan["probe_group_ids"]) & set(dataset["recommendation_group_id"]):
        raise RuntimeError("DUAL_BEAM8_PROBE_TRAIN_OVERLAP")
    max_steps = args.max_steps if args.max_steps is not None else audit["optimizer_steps"]
    if not 1 <= max_steps <= audit["optimizer_steps"]:
        raise ValueError(f"--max-steps must be in [1, {audit['optimizer_steps']}]")
    if max_steps % 2:
        raise ValueError("--max-steps must end on a num_iterations=2 boundary")
    plan["dataset"] = dataset
    plan["audit"] = audit
    plan["max_steps"] = max_steps
    plan["probe_train_overlap"] = []
    return plan


def dual_config_kwargs(**overrides):
    values = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": COT_G,
        "generation_batch_size": COT_G,
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


def make_dual_grpo_config(
    output_dir, max_steps, lr, seed, *, save_strategy="no",
    save_steps=500, save_total_limit=None, use_cpu=False,
):
    return GRPOConfig(**dual_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))


class DualManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_ThinkDualBeam8_v2",
            "experiment_type": "Think G4 + per-CoT Beam8 dual-objective GRPO",
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_recorded_external_score": PARENT_RECORDED_SCORE,
            "optimizer_initialization": "fresh",
            "optimizer": {"name": "AdamW", "learning_rate": 1e-6,
                          "weight_decay": 0.0, "scheduler": "constant"},
            "training_routes": ["think"],
            "probe4_retained": True,
            "probe4_excluded_from_training": True,
            "cot_group_size": COT_G,
            "sid_group_size_per_cot": SID_G,
            "sid_candidates_per_business_group": COT_G * SID_G,
            "cot_sampling": {"temperature": 0.9, "top_p": 0.95},
            "cot_reward": "Beam8 aggregate hierarchical think_reward",
            "cot_advantage": "G4 population-normalized across the four CoTs",
            "cot_loss_scope": "sampled CoT action tokens through first </think>",
            "sid_reward": "NoThink-v1 q_reward -1/-0.25/0/0.5/2/8",
            "sid_advantage": "independent G8 population-normalized inside each CoT",
            "sid_loss_scope": "three ABC tokens only; domain prefix is fixed context",
            "objective_weighting": "cot_loss + sid_loss (1:1)",
            "sid_objective_semantics": "Beam-selected PPO-style / group-relative optimization; not strict on-policy GRPO",
            "num_iterations": 2,
            "iteration2_reuse": ["CoT", "Beam8", "reward", "advantage", "old_logp"],
            "zero_std_reroll": False,
            "training_beam": {"num_beams": SID_G, "num_return_sequences": SID_G, "max_new_tokens": 3},
            "fixed_probe": {
                "enabled": True,
                "count": 4,
                "group_ids": list(FIXED_PROBE4_IDS),
                "evaluation_contract": "existing production-shaped Probe4/Beam32",
            },
            "expected_raw_groups": EXPECTED_RAW_GROUPS,
            "expected_training_groups": EXPECTED_TRAIN_GROUPS,
            "expected_fresh_rollouts": EXPECTED_TRAIN_GROUPS,
            "expected_optimizer_steps": EXPECTED_FORMAL_STEPS,
            "explicit_final_checkpoint_step": EXPECTED_FORMAL_STEPS,
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_dual_beam8(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append("dual_beam8.jsonl", {"type": "dual_beam8", **event})


def dual_monitor_from_env(run_id, rank):
    return DualManifestWriter(base_monitor_from_env(run_id, rank))


def install_bindings():
    baseline_runner.prepare_run_plan = prepare_dual_run_plan
    baseline_runner.load_model = load_model_and_capture_runtime
    baseline_runner.RecGRPOTrainer = ThinkDualBeam8Trainer
    baseline_runner.make_think_reward_func = make_runtime_dual_reward
    baseline_runner.make_grpo_config = make_dual_grpo_config
    baseline_runner.monitor_from_env = dual_monitor_from_env


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
