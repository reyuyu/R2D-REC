#!/usr/bin/env python3
"""Run fail-closed GRPO-2 Think-only validation or formal training."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import peft
import transformers
import trl

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
NATIVE_DIR = PACKAGE_DIR.parent
FULLBASE_SCRIPTS = NATIVE_DIR / "grpo_fullbase_conservative_v1" / "scripts"
GRPO_DIR = NATIVE_DIR / "grpo"
GRPO_SCRIPTS = NATIVE_DIR / "grpo" / "scripts"
for directory in (FULLBASE_SCRIPTS, GRPO_SCRIPTS, GRPO_DIR):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from grpo_probe import load_probe_records
from monitor.writer import monitor_from_env as base_monitor_from_env
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from checkpointing import assert_training_checkpoint, write_json_atomic
from modeling import enforce_lora_only_trainable, fresh_lora_config
from retention import summarize_retention
from run_grpo_trl_smoke import current_git_commit
from ablations.gr_rec_think_sample8_fullsid_positive_a0_v3.a0_reward import q_reward_without_a_only
from ablations.gr_rec_think_sample8_fullsid_v3 import sample8_fullsid_trainer as historical_trainer
from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import (
    Sample8FullSIDRuntime,
    ThinkG4SingleGroupSampler,
    audit_sample8_sampler,
    make_sample8_fullsid_reward_func,
)

from contracts import canonical_model_identity, file_sha256, validate_config, validate_dataset, validate_parent_manifest
from trainer import GRPO2ThinkTrainer

_CONFIG: dict = {}
_PARENT_MANIFEST: dict = {}
_DATASET_GUARD: dict = {}
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_PARSED = None
_MODEL_AUDIT: dict = {}


def is_formal() -> bool:
    return _CONFIG.get("run_kind") == "grpo2_formal_300"


def working_tree_clean() -> bool:
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        check=True, capture_output=True, text=True,
    )
    return not result.stdout.strip()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser():
    parser = baseline_runner.build_arg_parser_original() if hasattr(baseline_runner, "build_arg_parser_original") else baseline_runner.build_arg_parser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--probe-data-path")
    parser.add_argument("--allow-test-parent", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def prepare_plan(args):
    guard = validate_dataset(args.data_path)
    records = guard.pop("rows")
    dataset = Dataset.from_list(records)
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_sample8_sampler(dataset, sampler)
    max_steps = int(_CONFIG["optimization"]["max_steps"])
    if args.max_steps is not None and int(args.max_steps) != max_steps:
        raise RuntimeError("GRPO2_MAX_STEPS_CLI_CONFIG_MISMATCH")
    retention = _CONFIG["retention_probe"]
    group_ids = list(retention.get("group_ids", [])) if retention.get("enabled") else []
    if group_ids and not args.probe_data_path:
        raise RuntimeError("GRPO2_RETENTION_REQUIRES_PROBE_DATA_PATH")
    probe_records = load_probe_records(args.probe_data_path, group_ids) if group_ids else {}
    return {
        "raw_groups": len(dataset),
        "selected_groups_arg": len(dataset),
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "output_dir": Path(args.output_dir) / args.run_id,
        "resume_step": 0,
        "probe_group_ids": group_ids,
        "probe_records": probe_records,
        "secondary_probe_group_ids": [],
        "secondary_probe_records": {},
    }


def load_model(device):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER, _MODEL_AUDIT
    parent_dir = Path(_PARSED.parent_manifest).parent
    tokenizer = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        parent_dir, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2",
    )
    seed = int(_CONFIG["seeds"]["lora_initialization"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(base, fresh_lora_config())
    audit = enforce_lora_only_trainable(model)
    if audit["lora_trainable_tensor_count"] != 504 or audit["lora_trainable_parameter_count"] != 87_293_952:
        raise RuntimeError(f"GRPO2_LORA_PARAMETER_CONTRACT_FAILED: {audit}")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    train_seed = int(_CONFIG["seeds"]["training"]) + rank
    random.seed(train_seed)
    np.random.seed(train_seed % (2**32))
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed(train_seed)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER, _MODEL_AUDIT = model, tokenizer, audit
    print(json.dumps({
        "BASE_TRAINABLE_PARAMS": audit.get("base_trainable_parameter_count", 0),
        "GRPO2_LORA_TRAINABLE_PARAMS": audit["lora_trainable_parameter_count"],
        "GRPO2_LORA_TENSOR_COUNT": audit["lora_trainable_tensor_count"],
    }, sort_keys=True), flush=True)
    return model, tokenizer, audit["lora_trainable_parameter_count"]


def make_reward(beam32_fn=None):
    del beam32_fn
    if _RUNTIME_MODEL is None:
        raise RuntimeError("GRPO2_RUNTIME_MODEL_NOT_LOADED")
    runtime = Sample8FullSIDRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_sample8_fullsid_reward_func(runtime)


def make_disabled_nothink_reward_func(tokenizer=None):
    del tokenizer

    def disabled_nothink_reward(completions, **kwargs):
        routes = kwargs.get("route") or []
        if any(route != "think" for route in routes):
            raise RuntimeError("GRPO2_NOTHINK_TRAINING_ROUTE_FORBIDDEN")
        return [float("nan")] * len(completions)

    disabled_nothink_reward.__name__ = "nothink_reward_disabled_for_grpo2"
    return disabled_nothink_reward


def make_config(output_dir, max_steps, lr, seed, *, save_strategy="steps", save_steps=10, save_total_limit=2, use_cpu=False):
    del lr
    values = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "learning_rate": float(_CONFIG["optimization"]["learning_rate"]),
        "seed": seed,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 4,
        "generation_batch_size": 4,
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
        "shuffle_dataset": False,
        "save_strategy": "no" if is_formal() else save_strategy,
        "save_steps": save_steps,
        "save_total_limit": 5 if is_formal() else save_total_limit,
        "use_cpu": use_cpu,
    }
    return GRPOConfig(**values)


class MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "schema": "grpo2_think_fullbase_conservative_v1",
            "stage": "GRPO2_REC_THINK",
            "test_parent_only": _PARENT_MANIFEST["test_parent_only"],
            "canonical_grpo1_parent": _PARENT_MANIFEST["canonical_grpo1_parent"],
            "canonical_for_this_run": _PARENT_MANIFEST.get("canonical_for_this_run", False),
            "parent_canonical_model_identity": _PARENT_MANIFEST["canonical_model_identity"],
            "dataset_guard": _DATASET_GUARD,
            "training_routes": ["think"],
            "train_think_rows": 611,
            "nothink_training_rows": 0,
            "think_reward": "ON",
            "nothink_reward": "OFF_FOR_TRAINING_RETENTION_ONLY",
            "historical_math": "Positive-A0 V3: G4 CoT + four independent G8 FullSID; L_cot + L_sid",
            "historical_learning_rate": 1e-6,
            "formal_learning_rate" if is_formal() else "test_learning_rate": 2e-7,
            "fresh_lora": True,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "trl": trl.__version__,
                "peft": peft.__version__,
                "cuda": torch.version.cuda,
            },
            "deterministic_environment": {
                key: os.environ.get(key) for key in (
                    "CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS",
                    "FLASH_ATTENTION_DETERMINISTIC", "NVIDIA_TF32_OVERRIDE",
                    "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_DISABLE",
                )
            },
        })
        self._writer.write_manifest(payload)

    def write_sample8_fullsid(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append("sample8_fullsid.jsonl", {"type": "sample8_fullsid", **event})


def monitor_from_env(run_id, rank):
    return MonitorWriter(base_monitor_from_env(run_id, rank))


class BoundTrainer(GRPO2ThinkTrainer):
    def __init__(self, *args, **kwargs):
        config_sha = file_sha256(_PARSED.config)
        lineage = {
            "stage": "GRPO2_REC_THINK",
            "test_parent_only": _PARENT_MANIFEST["test_parent_only"],
            "canonical_grpo1_parent": _PARENT_MANIFEST["canonical_grpo1_parent"],
            "canonical_for_this_run": _PARENT_MANIFEST.get("canonical_for_this_run", False),
            "parent_canonical_model_identity": _PARENT_MANIFEST["canonical_model_identity"],
            "parent_full_model_sha256": _PARENT_MANIFEST["canonical_model_identity"],
            "source_grpo1_checkpoint": _PARENT_MANIFEST["source_grpo1_checkpoint"],
            "selection_basis": _PARENT_MANIFEST.get("selection_basis"),
            "external_best_confirmed": _PARENT_MANIFEST.get("external_best_confirmed", False),
            "source_sft_model_sha256": _PARENT_MANIFEST["source_sft_model_sha256"],
            "source_grpo1_adapter_sha256": _PARENT_MANIFEST["source_grpo1_adapter_sha256"],
            "dataset_sha256": _CONFIG["dataset"]["sha256"],
            "config_sha256": config_sha,
            "code_commit": current_git_commit(),
            "seed": _CONFIG["seeds"]["training"],
            "lora_initialization_seed": _CONFIG["seeds"]["lora_initialization"],
            "learning_rate_base": _CONFIG["optimization"]["learning_rate"],
        }
        output = Path(kwargs["args"].output_dir)
        checkpoint_steps = _CONFIG.get("checkpoint", {}).get("steps", [])
        super().__init__(
            *args, lineage=lineage, evidence_dir=output,
            checkpoint_steps=checkpoint_steps, **kwargs,
        )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            write_json_atomic(output / "run_manifest.json", {
                "schema": "grpo2_formal_run_v1" if is_formal() else "grpo2_validation_run_v1",
                "stage": "GRPO2_REC_THINK",
                "source_grpo1_checkpoint": _PARENT_MANIFEST["source_grpo1_checkpoint"],
                "source_grpo1_adapter_sha256": _PARENT_MANIFEST["source_grpo1_adapter_sha256"],
                "source_grpo1_external_best_confirmed": False,
                "parent_full_model_sha256": _PARENT_MANIFEST["canonical_model_identity"],
                "dataset_sha256": _CONFIG["dataset"]["sha256"],
                "config_sha256": config_sha,
                "code_commit": current_git_commit(),
                "max_steps": _CONFIG["optimization"]["max_steps"],
                "learning_rate": _CONFIG["optimization"]["learning_rate"],
                "checkpoint_steps": checkpoint_steps,
                "seeds": _CONFIG["seeds"],
                "train_think_rows": 611,
                "train_nothink_rows": 0,
                "base_trainable_params": _MODEL_AUDIT.get("base_trainable_parameter_count", 0),
                "grpo2_lora_trainable_params": _MODEL_AUDIT.get("lora_trainable_parameter_count"),
                "grpo2_lora_tensor_count": _MODEL_AUDIT.get("lora_trainable_tensor_count"),
                "fresh_lora": True,
            })


def _validate_preflight(args) -> dict:
    global _CONFIG, _PARENT_MANIFEST, _DATASET_GUARD
    _CONFIG = validate_config(_load_json(args.config))
    _PARENT_MANIFEST = validate_parent_manifest(args.parent_manifest, allow_test_parent=args.allow_test_parent)
    guard = validate_dataset(args.data_path)
    _DATASET_GUARD = {key: value for key, value in guard.items() if key != "rows"}
    if args.resume_from_checkpoint:
        raise RuntimeError("GRPO2_MUST_FRESH_START")
    if int(args.seed) != int(_CONFIG["seeds"]["training"]):
        raise RuntimeError("GRPO2_TRAINING_SEED_DRIFT")
    if float(args.lr) != float(_CONFIG["optimization"]["learning_rate"]):
        raise RuntimeError("GRPO2_LEARNING_RATE_CLI_CONFIG_MISMATCH")
    if not working_tree_clean():
        raise RuntimeError("GRPO2_WORKING_TREE_MUST_BE_CLEAN")
    identity, _ = canonical_model_identity(Path(args.parent_manifest).parent)
    return {
        "status": "READY_TO_EXECUTE",
        "run_id": args.run_id,
        "test_parent_only": _PARENT_MANIFEST["test_parent_only"],
        "canonical_grpo1_parent": _PARENT_MANIFEST["canonical_grpo1_parent"],
        "parent_canonical_model_identity": identity,
        "dataset_sha256": _DATASET_GUARD["sha256"],
        "max_steps": _CONFIG["optimization"]["max_steps"],
        "learning_rate": _CONFIG["optimization"]["learning_rate"],
        "git_commit": current_git_commit(),
    }


def install_bindings():
    historical_trainer.q_reward = q_reward_without_a_only
    if not hasattr(baseline_runner, "build_arg_parser_original"):
        baseline_runner.build_arg_parser_original = baseline_runner.build_arg_parser
    baseline_runner.build_arg_parser = build_parser
    baseline_runner.prepare_run_plan = prepare_plan
    baseline_runner.load_model = load_model
    baseline_runner.RecGRPOTrainer = BoundTrainer
    baseline_runner.make_think_reward_func = make_reward
    baseline_runner.make_nothink_reward_func = make_disabled_nothink_reward_func
    baseline_runner.make_grpo_config = make_config
    baseline_runner.monitor_from_env = monitor_from_env


def _write_checkpoint_manifest(checkpoint: Path, world_size: int) -> None:
    required = [
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
        "scheduler.pt", "trainer_state.json", "training_args.bin",
        *[f"rng_state_{rank}.pth" for rank in range(world_size)], "lineage.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"GRPO2_CHECKPOINT_INCOMPLETE: {missing}")
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {
        "schema": "grpo2_checkpoint_manifest_v1",
        "step": int(checkpoint.name.rsplit("-", 1)[1]),
        "adapter_only": True,
        "resume_capable": True,
        "world_size": world_size,
        "files": [
            {
                "name": name,
                "size": (checkpoint / name).stat().st_size,
                "sha256": file_sha256(checkpoint / name),
            }
            for name in required
        ],
    })


def _finalize(args) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    output = Path(args.output_dir) / args.run_id
    if is_formal():
        expected_steps = list(_CONFIG["checkpoint"]["steps"])
    else:
        expected_steps = [5] if _CONFIG["optimization"]["max_steps"] == 5 else [10, 20]
    for step in expected_steps:
        _write_checkpoint_manifest(output / f"checkpoint-{step}", 4)
    checkpoints = [assert_training_checkpoint(output / f"checkpoint-{step}", 4) for step in expected_steps]
    for checkpoint, step in zip(checkpoints, expected_steps):
        directory = output / f"checkpoint-{step}"
        if not (directory / "checkpoint_manifest.json").is_file():
            raise RuntimeError(f"GRPO2_CHECKPOINT_MANIFEST_MISSING: {step}")
        state = _load_json(directory / "trainer_state.json")
        if int(state.get("max_steps", -1)) != int(_CONFIG["optimization"]["max_steps"]):
            raise RuntimeError(f"GRPO2_CHECKPOINT_MAX_STEPS_MISMATCH: {step}")
    parent_identity_after, _ = canonical_model_identity(Path(args.parent_manifest).parent)
    if parent_identity_after != _PARENT_MANIFEST["canonical_model_identity"]:
        raise RuntimeError("GRPO2_PARENT_MUTATED_DURING_TRAINING")
    retention = None
    if _CONFIG["retention_probe"].get("enabled"):
        monitor_dir = Path(os.environ["GRPO_MONITOR_DIR"]) / args.run_id
        retention = summarize_retention(monitor_dir / "probes.jsonl", suite=None)
        write_json_atomic(output / "retention-summary.json", retention)
    summary = {
        "status": "PASS",
        "run_id": args.run_id,
        "global_step": _CONFIG["optimization"]["max_steps"],
        "test_parent_only": _PARENT_MANIFEST["test_parent_only"],
        "canonical_grpo1_parent": _PARENT_MANIFEST["canonical_grpo1_parent"],
        "parent_unchanged": True,
        "checkpoints": checkpoints,
        "retention": retention,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(output / "summary.json", summary)
    if is_formal():
        write_json_atomic(output / "checkpoint_summary.json", {"checkpoints": checkpoints})
        rank_summary = _load_json(output / "run-summary-rank0.json")
        training_summary = {
            **summary,
            "final_step": rank_summary["global_step"],
            "final_loss": rank_summary["train_loss"],
            "max_base_parameter_delta": rank_summary["base_delta"],
            "base_unchanged": rank_summary["base_delta"] == 0.0,
            "nonfinite_count": 0,
            "oom_count": 0,
            "nccl_fatal_count": 0,
            "final_state": "READY_FOR_GRPO2_CHECKPOINT_EVALUATION",
        }
        write_json_atomic(output / "training_summary.json", training_summary)
        report = (
            "# Formal GRPO-2 report\n\n"
            "Status: `READY_FOR_GRPO2_CHECKPOINT_EVALUATION`\n\n"
            f"Parent: GRPO-1 checkpoint-500 (`{_PARENT_MANIFEST['canonical_model_identity']}`)\n\n"
            "Training: Think-only, fresh LoRA, LR `2e-7`, steps `0..300`.\n"
        )
        (output / "FORMAL_GRPO2_REPORT.md").write_text(report, encoding="utf-8")


def main(argv=None) -> int:
    global _PARSED
    install_bindings()
    args = build_parser().parse_args(argv)
    _PARSED = args
    record = _validate_preflight(args)
    if args.preflight_only:
        preflight_dir = Path(args.output_dir) / "preflight"
        write_json_atomic(preflight_dir / f"{args.run_id}.json", record)
        print(json.dumps(record, sort_keys=True))
        return 0
    preflight_path = Path(args.output_dir) / "preflight" / f"{args.run_id}.json"
    if not preflight_path.is_file() or _load_json(preflight_path) != record:
        raise RuntimeError("GRPO2_EXECUTION_PREFLIGHT_RECORD_MISMATCH")
    os.environ.setdefault("GRPO_MONITOR", "1")
    os.environ.setdefault("GRPO_DETAILED_MONITOR", "1")
    os.environ.setdefault("GRPO_PARITY_AUDIT", "1")
    os.environ.setdefault("GRPO_TRACE_EVERY", "1")
    baseline_runner.main(argv)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    _finalize(args)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
