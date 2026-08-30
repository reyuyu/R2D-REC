#!/usr/bin/env python3
"""Formal runner for Think G4 + per-CoT Official Sample8 ABC3 GRPO."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = GRPO_ROOT / "scripts"
PARENT_ADAPTER = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
PARENT_ADAPTER_SHA256 = "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3"
SOURCE_DATASET = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
SOURCE_DATASET_SHA256 = "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
TRAIN_GROUP_ID_SHA256 = "c1416069c076469252668bb27f69b0904946bd2d2154cbe3024115a17175be5f"
TRAIN_CANONICAL_SHA256 = "1ff7965aab7ac0126f1898699b704e5df65b9ad270f8f50c40cf12b5f6d49055"
EXPECTED_DOMAIN_GROUPS = {"ad": 426, "living": 189, "prod": 381, "video": 549}
EXPECTED_RAW_GROUPS = 1549
EXPECTED_TRAIN_GROUPS = 1545
EXPECTED_FORMAL_STEPS = 3090
FORMAL_OUTPUT_ROOT = "/root/GRPO-checkpoints"
FORMAL_SAVE_STEPS = 50
FORMAL_PROBE_EVERY_STEPS = 50
FORMAL_SAVE_TOTAL_LIMIT = 64
FORMAL_RUN_ID_PREFIX = "GR-REC-THINK-OFFICIAL-SAMPLE8-V3-FORMAL-"
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

from .official_sample8_trainer import (
    COT_G,
    SID_G,
    OFFICIAL_SID_TOKENS,
    MAX_COT_CLOSURE_RETRIES,
    OfficialSample8Runtime,
    ThinkOfficialSample8Trainer,
    ThinkG4SingleGroupSampler,
    audit_official_sampler,
    make_official_sample8_reward_func,
)
from .official_probe import OfficialProbeEvaluator
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import (
    assert_runtime_import_provenance,
)

_BASE_PREPARE_RUN_PLAN = baseline_runner.prepare_run_plan
_BASE_LOAD_MODEL = baseline_runner.load_model
_BASE_BUILD_ARG_PARSER = baseline_runner.build_arg_parser
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_RUNTIME_PROVENANCE = None
_TRUSTED_RESUME = None


def _resume_checkpoint_arg(argv):
    values = list(argv or [])
    for index, value in enumerate(values):
        if value == "--resume-from-checkpoint" and index + 1 < len(values):
            return values[index + 1]
        if value.startswith("--resume-from-checkpoint="):
            return value.split("=", 1)[1]
    return None


def _run_id_arg(argv):
    values = list(argv or [])
    for index, value in enumerate(values):
        if value == "--run-id" and index + 1 < len(values):
            return values[index + 1]
        if value.startswith("--run-id="):
            return value.split("=", 1)[1]
    raise ValueError("V3-Official requires --run-id")


def validate_trusted_resume_checkpoint(
    checkpoint_path, run_id, output_root=FORMAL_OUTPUT_ROOT,
):
    if not run_id.startswith(FORMAL_RUN_ID_PREFIX):
        raise RuntimeError(f"V3_OFFICIAL_UNTRUSTED_RESUME_RUN_ID: {run_id}")
    root = Path(output_root).resolve(strict=True)
    expected_run_dir = (root / run_id).resolve(strict=False)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if checkpoint.parent != expected_run_dir:
        raise RuntimeError(
            "V3_OFFICIAL_UNTRUSTED_RESUME_PATH "
            f"expected_parent={expected_run_dir} checkpoint={checkpoint}"
        )
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if not match or int(match.group(1)) % 2:
        raise RuntimeError(f"V3_OFFICIAL_INVALID_RESUME_BOUNDARY: {checkpoint}")
    step = int(match.group(1))
    required = [
        "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
        "scheduler.pt", "training_args.bin", "trainer_state.json",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ]
    incomplete = [
        name for name in required
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    if incomplete:
        raise RuntimeError(f"V3_OFFICIAL_RESUME_INCOMPLETE: {incomplete}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError(
            f"V3_OFFICIAL_RESUME_STEP_MISMATCH path={step} "
            f"state={state.get('global_step')}"
        )
    return {
        "path": str(checkpoint),
        "run_id": run_id,
        "step": step,
        "trusted_v3_official_checkpoint": True,
    }


def enable_trusted_torch_load_for_resume(checkpoint_path, run_id):
    audit = validate_trusted_resume_checkpoint(checkpoint_path, run_id)
    # The optimizer/RNG files are pickle-backed. They are loaded only after the
    # checkpoint has passed the same-run, boundary, state, and completeness guards.
    import numpy as np
    import torch
    import transformers.trainer as transformers_trainer

    np_core = getattr(np, "_core", np.core)
    numpy_allowlist = [
        np_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]

    def trusted_numpy_safe_globals():
        return torch.serialization.safe_globals(numpy_allowlist)

    transformers_trainer.check_torch_load_is_safe = lambda: None
    transformers_trainer.safe_globals = trusted_numpy_safe_globals
    return audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_train_sha256(rows):
    digest = hashlib.sha256()
    keys = ("prompt", "route", "recommendation_group_id", "target_domain", "all_gold_sids")
    for row in rows:
        digest.update((json.dumps(
            {key: row[key] for key in keys}, ensure_ascii=False, separators=(",", ":")
        ) + "\n").encode())
    return digest.hexdigest()


def validate_official_parent():
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (PARENT_ADAPTER / name).is_file()]
    if missing:
        raise FileNotFoundError(f"V3_OFFICIAL_PARENT_INCOMPLETE: {missing}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(
            f"V3_OFFICIAL_PARENT_SHA_MISMATCH expected={PARENT_ADAPTER_SHA256} actual={actual}"
        )
    return {
        "path": str(PARENT_ADAPTER),
        "adapter_sha256": actual,
    }


def load_model_and_capture_runtime(*args, **kwargs):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER
    result = _BASE_LOAD_MODEL(*args, **kwargs)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER = result[0], result[1]
    return result


def make_runtime_official_reward(beam32_fn=None):
    # The baseline runner still constructs its historical training Beam32 helper;
    # this ablation intentionally ignores it. Probe4 keeps a separate Beam32 path.
    del beam32_fn
    if _RUNTIME_MODEL is None or _RUNTIME_TOKENIZER is None:
        raise RuntimeError("V3_OFFICIAL_RUNTIME_MODEL_NOT_CAPTURED")
    runtime = OfficialSample8Runtime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_official_sample8_reward_func(runtime)


def prepare_official_run_plan(args):
    # Keep the original fixed Probe4 selection and exclusion logic intact.
    if Path(baseline_runner.DATA).resolve() != SOURCE_DATASET.resolve():
        raise RuntimeError("V3_OFFICIAL_SOURCE_PATH_DRIFT")
    if sha256_file(SOURCE_DATASET) != SOURCE_DATASET_SHA256:
        raise RuntimeError("V3_OFFICIAL_SOURCE_SHA_DRIFT")
    plan = _BASE_PREPARE_RUN_PLAN(args)
    if plan["raw_groups"] != EXPECTED_RAW_GROUPS:
        raise RuntimeError(f"V3_OFFICIAL_RAW_TOPOLOGY_DRIFT: {plan['raw_groups']}")
    if tuple(plan["probe_group_ids"]) != FIXED_PROBE4_IDS:
        raise RuntimeError(
            f"V3_OFFICIAL_PROBE4_ID_DRIFT expected={FIXED_PROBE4_IDS} "
            f"actual={tuple(plan['probe_group_ids'])}"
        )
    think_indices = [
        index for index, row in enumerate(plan["dataset"])
        if row["route"] == "think"
    ]
    dataset = plan["dataset"].select(think_indices)
    rows = [dict(row) for row in dataset]
    group_ids = [row["recommendation_group_id"] for row in rows]
    group_id_sha = hashlib.sha256(("\n".join(sorted(group_ids)) + "\n").encode()).hexdigest()
    canonical_sha = canonical_train_sha256(rows)
    domains = dict(sorted(Counter(row["target_domain"] for row in rows).items()))
    if group_id_sha != TRAIN_GROUP_ID_SHA256 or canonical_sha != TRAIN_CANONICAL_SHA256:
        raise RuntimeError("V3_OFFICIAL_TRAIN_DATASET_PARITY_FAILED")
    if domains != EXPECTED_DOMAIN_GROUPS:
        raise RuntimeError(f"V3_OFFICIAL_DOMAIN_DISTRIBUTION_DRIFT: {domains}")
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_official_sampler(dataset, sampler)
    if audit["trained_groups"] != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError(f"V3_OFFICIAL_TRAIN_TOPOLOGY_DRIFT: {audit['trained_groups']}")
    if set(plan["probe_group_ids"]) & set(dataset["recommendation_group_id"]):
        raise RuntimeError("V3_OFFICIAL_PROBE_TRAIN_OVERLAP")
    max_steps = args.max_steps if args.max_steps is not None else audit["optimizer_steps"]
    if not 1 <= max_steps <= audit["optimizer_steps"]:
        raise ValueError(f"--max-steps must be in [1, {audit['optimizer_steps']}]")
    if max_steps % 2:
        raise ValueError("--max-steps must end on a num_iterations=2 boundary")
    plan["dataset"] = dataset
    plan["audit"] = audit
    plan["max_steps"] = max_steps
    plan["probe_train_overlap"] = []
    plan["dataset_guard"] = {
        "source_path": str(SOURCE_DATASET),
        "source_sha256": SOURCE_DATASET_SHA256,
        "raw_business_groups": EXPECTED_RAW_GROUPS,
        "train_business_groups": EXPECTED_TRAIN_GROUPS,
        "train_group_id_sha256": group_id_sha,
        "train_canonical_sha256": canonical_sha,
        "domain_group_counts": domains,
        "think_only": True,
        "probe4_overlap": 0,
    }
    return plan


def official_config_kwargs(**overrides):
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


def build_official_arg_parser():
    parser = _BASE_BUILD_ARG_PARSER()
    parser.set_defaults(
        output_dir=FORMAL_OUTPUT_ROOT,
        save_steps=FORMAL_SAVE_STEPS,
        save_total_limit=FORMAL_SAVE_TOTAL_LIMIT,
        probe_every_steps=FORMAL_PROBE_EVERY_STEPS,
    )
    return parser


def make_official_grpo_config(
    output_dir, max_steps, lr, seed, *, save_strategy="no",
    save_steps=500, save_total_limit=None, use_cpu=False,
):
    if float(lr) != 1e-6:
        raise ValueError("V3-Official learning rate is frozen at 1e-6")
    return GRPOConfig(**official_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))


class OfficialSample8ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        base_fixed_probe = dict(payload.get("fixed_probe") or {})
        payload.update({
            "experiment": "GR_REC_ThinkOfficialSample8_v3",
            "experiment_type": "Think G4 + per-CoT stochastic Official Sample8 ABC3 GRPO",
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_contract": "Beta baseline checkpoint-1106; never a GRPO checkpoint",
            "optimizer_initialization": "restored" if _TRUSTED_RESUME else "fresh",
            "trusted_resume": _TRUSTED_RESUME,
            "resume_policy": (
                "fresh Beta checkpoint-1106, or complete even-step checkpoint from "
                "the identical V3-Official formal run-id"
            ),
            "optimizer": {"name": "AdamW", "learning_rate": 1e-6,
                          "weight_decay": 0.0, "scheduler": "constant"},
            "training_routes": ["think"],
            "probe4_retained": True,
            "probe4_excluded_from_training": True,
            "cot_group_size": COT_G,
            "sid_group_size_per_cot": SID_G,
            "sid_candidates_per_business_group": COT_G * SID_G,
            "cot_sampling": {"temperature": 0.9, "top_p": 0.95},
            "cot_reward": "arithmetic sum of the eight sampled SID q_reward values",
            "cot_advantage": "G4 population-normalized across the four CoTs",
            "cot_loss_scope": "sampled CoT action tokens through first </think>",
            "sid_reward": "NoThink-v1 q_reward -1/-0.25/0/0.5/2/8",
            "sid_advantage": "independent G8 population-normalized inside each CoT",
            "sid_loss_scope": "ABC3 action tokens only; fixed target-domain begin is context",
            "sid_parser": "strict ABC3 token parse joined with the fixed target domain",
            "objective_weighting": "cot_loss + sid_loss (1:1)",
            "sid_objective_semantics": "on-policy stochastic Sample8 group-relative PPO",
            "num_iterations": 2,
            "iteration2_reuse": ["CoT", "Official Sample8 ABC3", "reward", "advantage", "old_logp"],
            "zero_std_reroll": False,
            "cot_closure_recovery": {
                "all_rank_acceptance": True,
                "max_extra_retries": MAX_COT_CLOSURE_RETRIES,
                "discard_failed_attempt_before_sid_sampling": True,
                "iteration2_reuses_only_accepted_rollout": True,
            },
            "training_sid_sampling": {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "num_return_sequences": SID_G,
                "min_new_tokens": OFFICIAL_SID_TOKENS,
                "max_new_tokens": OFFICIAL_SID_TOKENS,
                "select_sid": "strict ABC3 joined with fixed target domain",
                "fixed_domain_prefix": True,
                "natural_language_bridge": False,
            },
            "fixed_probe": {
                **base_fixed_probe,
                "enabled": True,
                "count": 4,
                "group_ids": list(FIXED_PROBE4_IDS),
                "every_steps": FORMAL_PROBE_EVERY_STEPS,
                "evaluation_contract": "Official: sampled CoT + fixed target-domain begin + production Beam32 ABC3",
            },
            "expected_raw_groups": EXPECTED_RAW_GROUPS,
            "expected_training_groups": EXPECTED_TRAIN_GROUPS,
            "expected_fresh_rollouts": EXPECTED_TRAIN_GROUPS,
            "expected_optimizer_steps": EXPECTED_FORMAL_STEPS,
            "explicit_final_checkpoint_step": EXPECTED_FORMAL_STEPS,
            "dataset_guard": {
                "source_path": str(SOURCE_DATASET),
                "source_sha256": SOURCE_DATASET_SHA256,
                "raw_business_groups": EXPECTED_RAW_GROUPS,
                "train_business_groups": EXPECTED_TRAIN_GROUPS,
                "train_group_id_sha256": TRAIN_GROUP_ID_SHA256,
                "train_canonical_sha256": TRAIN_CANONICAL_SHA256,
                "domain_group_counts": EXPECTED_DOMAIN_GROUPS,
                "think_only": True,
                "probe4_overlap": 0,
            },
            "reward_engineering": {
                "copy_shaping": False,
                "warmup": False,
                "fine_grained_frontier": False,
                "duplicate_penalty": False,
                "saturation": False,
                "consistency": False,
            },
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_official_sample8_v3(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append("official_sample8.jsonl", {"type": "official_sample8", **event})


def official_monitor_from_env(run_id, rank):
    return OfficialSample8ManifestWriter(base_monitor_from_env(run_id, rank))


def install_official_bindings():
    baseline_runner.build_arg_parser = build_official_arg_parser
    baseline_runner.prepare_run_plan = prepare_official_run_plan
    baseline_runner.load_model = load_model_and_capture_runtime
    baseline_runner.RecGRPOTrainer = ThinkOfficialSample8Trainer
    baseline_runner.make_think_reward_func = make_runtime_official_reward
    baseline_runner.make_grpo_config = make_official_grpo_config
    baseline_runner.monitor_from_env = official_monitor_from_env
    baseline_runner.FixedProbeEvaluator = OfficialProbeEvaluator


def main(argv=None):
    global _RUNTIME_PROVENANCE, _TRUSTED_RESUME
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_official_parent()
    resume_checkpoint = _resume_checkpoint_arg(argv)
    if resume_checkpoint:
        _TRUSTED_RESUME = enable_trusted_torch_load_for_resume(
            resume_checkpoint, _run_id_arg(argv)
        )
        _RUNTIME_PROVENANCE["trusted_resume"] = _TRUSTED_RESUME
    else:
        _TRUSTED_RESUME = None
    install_official_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
