#!/usr/bin/env python3
"""Validate GRPO-2 by continuing the trainable GRPO-1 checkpoint-500 adapter."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import peft
import torch
import transformers
import trl

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
NATIVE_DIR = PACKAGE_DIR.parent
FULLBASE_SCRIPTS = NATIVE_DIR / "grpo_fullbase_conservative_v1" / "scripts"
GRPO_DIR = NATIVE_DIR / "grpo"
GRPO_SCRIPTS = GRPO_DIR / "scripts"
for directory in (FULLBASE_SCRIPTS, GRPO_SCRIPTS, GRPO_DIR):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from grpo_probe import load_probe_records
from monitor.writer import monitor_from_env as base_monitor_from_env
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from adapter_identity import audit_loaded_adapter
from checkpointing import assert_training_checkpoint, write_json_atomic
from continued_retention import summarize as summarize_retention
from continued_contracts import (
    GRPO1_STEP500_ADAPTER_SHA256,
    SFT_MODEL_SHA256,
    file_sha256,
    load_json,
    validate_config,
    validate_dataset,
    validate_grpo2_resume,
    validate_parent_sources,
)
from modeling import enforce_lora_only_trainable
from run_grpo_trl_smoke import current_git_commit
from trainer import GRPO2ContinuedAdapterTrainer
from ablations.gr_rec_think_sample8_fullsid_positive_a0_v3.a0_reward import q_reward_without_a_only
from ablations.gr_rec_think_sample8_fullsid_v3 import sample8_fullsid_trainer as historical_trainer
from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import (
    Sample8FullSIDRuntime,
    ThinkG4SingleGroupSampler,
    audit_sample8_sampler,
    make_sample8_fullsid_reward_func,
)

_CONFIG: dict = {}
_SOURCE_AUDIT: dict = {}
_DATASET_GUARD: dict = {}
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_PARSED = None
_MODEL_AUDIT: dict = {}
_STEP0_AUDIT: dict = {}


def working_tree_clean() -> bool:
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True)
    return not result.stdout.strip()


def build_parser():
    parser = baseline_runner.build_arg_parser_original() if hasattr(baseline_runner, "build_arg_parser_original") else baseline_runner.build_arg_parser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-parent", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--probe-data-path", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--step0-only", action="store_true")
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
    group_ids = list(_CONFIG["retention_probe"]["group_ids"])
    probe_records = load_probe_records(args.probe_data_path, group_ids)
    resume_step = validate_grpo2_resume(args.resume_from_checkpoint)["step"] if args.resume_from_checkpoint else 0
    return {
        "raw_groups": len(dataset),
        "selected_groups_arg": len(dataset),
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "output_dir": Path(args.output_dir) / args.run_id,
        "resume_step": resume_step,
        "probe_group_ids": group_ids,
        "probe_records": probe_records,
        "secondary_probe_group_ids": [],
        "secondary_probe_records": {},
    }


def load_model(device):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER, _MODEL_AUDIT, _STEP0_AUDIT
    tokenizer = AutoTokenizer.from_pretrained(_PARSED.base_model, local_files_only=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        _PARSED.base_model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base,
        _PARSED.adapter_parent,
        is_trainable=True,
        autocast_adapter_dtype=False,
        local_files_only=True,
    )
    audit = enforce_lora_only_trainable(model)
    if audit["lora_trainable_tensor_count"] != 504 or audit["lora_trainable_parameter_count"] != 87_293_952:
        raise RuntimeError(f"GRPO2_INHERITED_LORA_PARAMETER_CONTRACT_FAILED: {audit}")
    step0 = audit_loaded_adapter(
        model,
        Path(_PARSED.adapter_parent) / "adapter_model.safetensors",
        device=device,
    )
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    output = Path(_PARSED.output_dir) / _PARSED.run_id
    write_json_atomic(output / f"step0-adapter-parity-rank{rank}.json", {**step0, **audit})
    train_seed = int(_CONFIG["seeds"]["training"]) + rank
    random.seed(train_seed)
    np.random.seed(train_seed % (2**32))
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed(train_seed)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER = model, tokenizer
    _MODEL_AUDIT, _STEP0_AUDIT = audit, step0
    print(json.dumps({
        "BASE_TRAINABLE_PARAMS": audit["base_trainable_parameter_count"],
        "LORA_TRAINABLE_PARAMS": audit["lora_trainable_parameter_count"],
        "LORA_TRAINABLE_TENSORS": audit["lora_trainable_tensor_count"],
        "STEP0_ADAPTER_PARITY": step0["status"],
    }, sort_keys=True), flush=True)
    return model, tokenizer, audit["lora_trainable_parameter_count"]


def make_reward(beam32_fn=None):
    del beam32_fn
    runtime = Sample8FullSIDRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_sample8_fullsid_reward_func(runtime)


def make_disabled_nothink_reward_func(tokenizer=None):
    del tokenizer

    def disabled(completions, **kwargs):
        routes = kwargs.get("route") or []
        if any(route != "think" for route in routes):
            raise RuntimeError("GRPO2_NOTHINK_TRAINING_ROUTE_FORBIDDEN")
        return [float("nan")] * len(completions)

    disabled.__name__ = "nothink_reward_disabled_for_grpo2"
    return disabled


def make_config(output_dir, max_steps, lr, seed, *, save_strategy="steps", save_steps=10, save_total_limit=2, use_cpu=False):
    del lr
    return GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=float(_CONFIG["optimization"]["learning_rate"]),
        seed=seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        generation_batch_size=4,
        max_prompt_length=8192,
        max_completion_length=2048,
        num_iterations=2,
        beta=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        disable_dropout=True,
        importance_sampling_level="token",
        top_entropy_quantile=1.0,
        mask_truncated_completions=False,
        use_vllm=False,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        report_to="none",
        temperature=0.9,
        top_p=0.95,
        shuffle_dataset=False,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        use_cpu=use_cpu,
    )


class MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "schema": "grpo2_think_continued_adapter_v1",
            "stage": "GRPO2_REC_THINK",
            "adapter_inheritance": "CONTINUE_PARENT_ADAPTER",
            "adapter_initialization": "INHERITED_FROM_GRPO1",
            "fresh_lora": False,
            "fresh_optimizer": True,
            "training_resume_from_grpo1": False,
            "base_full_model_sha256": SFT_MODEL_SHA256,
            "grpo1_parent_checkpoint": 500,
            "grpo1_parent_adapter_sha256": GRPO1_STEP500_ADAPTER_SHA256,
            "dataset_guard": _DATASET_GUARD,
            "training_routes": ["think"],
            "train_think_rows": 611,
            "train_nothink_rows": 0,
            "historical_math": "Positive-A0 V3: G4 CoT + four independent G8 FullSID; L_cot + L_sid",
            "learning_rate": 2e-7,
            "versions": {
                "torch": torch.__version__, "transformers": transformers.__version__,
                "trl": trl.__version__, "peft": peft.__version__, "cuda": torch.version.cuda,
            },
        })
        self._writer.write_manifest(payload)

    def write_sample8_fullsid(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append("sample8_fullsid.jsonl", {"type": "sample8_fullsid", **event})


def monitor_from_env(run_id, rank):
    return MonitorWriter(base_monitor_from_env(run_id, rank))


class BoundTrainer(GRPO2ContinuedAdapterTrainer):
    def __init__(self, *args, **kwargs):
        config_sha = file_sha256(_PARSED.config)
        lineage = {
            "base_full_model_sha256": SFT_MODEL_SHA256,
            "adapter_initialization_source_stage": "GRPO1_REC_BILATERAL",
            "adapter_initialization_source_checkpoint": 500,
            "adapter_initialization_source_sha256": GRPO1_STEP500_ADAPTER_SHA256,
            "optimizer_initialization": "fresh",
            "trainer_state_initialization": "fresh",
            "learning_rate": 2e-7,
            "dataset_sha256": _CONFIG["dataset"]["sha256"],
            "config_sha256": config_sha,
            "code_commit": current_git_commit(),
            "seed": _CONFIG["seeds"]["training"],
        }
        output = Path(kwargs["args"].output_dir)
        super().__init__(*args, lineage=lineage, evidence_dir=output, **kwargs)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            write_json_atomic(output / "run_manifest.json", {
                "schema": "grpo2_continued_adapter_validation_v1",
                "stage": "GRPO2_REC_THINK",
                "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
                "adapter_weight_parent": "GRPO1 checkpoint-500",
                "adapter_initialization_source_sha256": GRPO1_STEP500_ADAPTER_SHA256,
                "optimizer_parent": "NONE",
                "trainer_state_parent": "NONE",
                "rng_parent": "NONE",
                "training_resume": bool(_PARSED.resume_from_checkpoint),
                "stage_fresh_start": not bool(_PARSED.resume_from_checkpoint),
                "adapter_continuation": True,
                "fresh_lora": False,
                "fresh_optimizer": not bool(_PARSED.resume_from_checkpoint),
                "base_full_model_sha256": SFT_MODEL_SHA256,
                "dataset_sha256": _CONFIG["dataset"]["sha256"],
                "config_sha256": config_sha,
                "code_commit": current_git_commit(),
                "max_steps": _CONFIG["optimization"]["max_steps"],
                "learning_rate": 2e-7,
                "train_think_rows": 611,
                "train_nothink_rows": 0,
                "base_trainable_params": _MODEL_AUDIT["base_trainable_parameter_count"],
                "lora_trainable_params": _MODEL_AUDIT["lora_trainable_parameter_count"],
                "lora_tensor_count": _MODEL_AUDIT["lora_trainable_tensor_count"],
                "step0_adapter_parity": _STEP0_AUDIT,
            })


def _validate_preflight(args) -> dict:
    global _CONFIG, _SOURCE_AUDIT, _DATASET_GUARD
    _CONFIG = validate_config(load_json(args.config))
    _SOURCE_AUDIT = validate_parent_sources(args.base_model, args.adapter_parent)
    dataset = validate_dataset(args.data_path)
    _DATASET_GUARD = {key: value for key, value in dataset.items() if key != "rows"}
    if args.resume_from_checkpoint:
        validate_grpo2_resume(args.resume_from_checkpoint)
    if int(args.seed) != int(_CONFIG["seeds"]["training"]):
        raise RuntimeError("GRPO2_TRAINING_SEED_DRIFT")
    if float(args.lr) != 2e-7:
        raise RuntimeError("GRPO2_LEARNING_RATE_CLI_CONFIG_MISMATCH")
    if not working_tree_clean():
        raise RuntimeError("GRPO2_WORKING_TREE_MUST_BE_CLEAN")
    return {
        "status": "READY_TO_EXECUTE",
        "mode": "CONTINUED_SINGLE_ADAPTER",
        "run_id": args.run_id,
        "base_full_sft_sha256": _SOURCE_AUDIT["base_full_model_sha256"],
        "grpo1_parent_checkpoint": 500,
        "grpo1_parent_adapter_sha256": _SOURCE_AUDIT["adapter_sha256"],
        "adapter_initialization": "INHERITED_FROM_GRPO1",
        "fresh_lora": False,
        "fresh_optimizer": not bool(args.resume_from_checkpoint),
        "training_resume_from_grpo1": False,
        "grpo2_checkpoint_resume": bool(args.resume_from_checkpoint),
        "dataset_sha256": _DATASET_GUARD["sha256"],
        "max_steps": _CONFIG["optimization"]["max_steps"],
        "learning_rate": 2e-7,
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


def _write_checkpoint_manifest(checkpoint: Path) -> dict:
    required = [
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
        "trainer_state.json", "training_args.bin", *[f"rng_state_{rank}.pth" for rank in range(4)], "lineage.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"GRPO2_CHECKPOINT_INCOMPLETE: {missing}")
    record = assert_training_checkpoint(checkpoint, 4)
    manifest = {
        "schema": "grpo2_continued_adapter_checkpoint_manifest_v1",
        "step": record["step"],
        "adapter_only": True,
        "resume_capable": True,
        "files": [{"name": name, "size": (checkpoint / name).stat().st_size, "sha256": file_sha256(checkpoint / name)} for name in required],
    }
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return record


def _finalize(args) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    output = Path(args.output_dir) / args.run_id
    expected = [5] if _CONFIG["optimization"]["max_steps"] == 5 else [10, 20]
    checkpoints = [_write_checkpoint_manifest(output / f"checkpoint-{step}") for step in expected]
    if checkpoints[-1]["adapter_sha256"] == GRPO1_STEP500_ADAPTER_SHA256:
        raise RuntimeError("GRPO2_INHERITED_ADAPTER_FINAL_WEIGHT_DID_NOT_CHANGE")
    source_after = validate_parent_sources(args.base_model, args.adapter_parent)
    if source_after["base_full_model_sha256"] != _SOURCE_AUDIT["base_full_model_sha256"] or source_after["adapter_sha256"] != _SOURCE_AUDIT["adapter_sha256"]:
        raise RuntimeError("GRPO2_PARENT_SOURCE_MUTATED_DURING_TRAINING")
    monitor_dir = Path(os.environ["GRPO_MONITOR_DIR"]) / args.run_id
    retention = summarize_retention(monitor_dir / "probes.jsonl")
    write_json_atomic(output / "retention-summary.json", retention)
    rank_summary = load_json(output / "run-summary-rank0.json")
    trainer_evidence = load_json(output / "trainer-evidence-rank0.json")
    summary = {
        "status": "PASS",
        "mode": "CONTINUED_SINGLE_ADAPTER",
        "run_id": args.run_id,
        "global_step": _CONFIG["optimization"]["max_steps"],
        "base_full_sft_sha256": SFT_MODEL_SHA256,
        "grpo1_parent_checkpoint": 500,
        "grpo1_parent_adapter_sha256": GRPO1_STEP500_ADAPTER_SHA256,
        "adapter_initialization": "INHERITED_FROM_GRPO1",
        "fresh_lora": False,
        "fresh_optimizer": True,
        "training_resume_from_grpo1": False,
        "base_trainable_params": _MODEL_AUDIT["base_trainable_parameter_count"],
        "lora_trainable_params": _MODEL_AUDIT["lora_trainable_parameter_count"],
        "lora_tensor_count": _MODEL_AUDIT["lora_trainable_tensor_count"],
        "optimizer_lora_only": trainer_evidence["optimizer_audit"]["optimizer_lora_only"],
        "step0_adapter_parity": _STEP0_AUDIT["status"],
        "adapter_delta_norm": trainer_evidence["adapter_delta_norm"],
        "grpo1_to_smoke_adapter_changed": True,
        "base_changed": False,
        "base_max_parameter_delta": trainer_evidence["base_max_parameter_delta"],
        "final_loss": rank_summary["train_loss"],
        "checkpoints": checkpoints,
        "retention": retention,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(output / "summary.json", summary)


def main(argv=None) -> int:
    global _PARSED
    install_bindings()
    args = build_parser().parse_args(argv)
    _PARSED = args
    record = _validate_preflight(args)
    preflight_path = Path(args.output_dir) / "preflight" / f"{args.run_id}.json"
    if args.preflight_only:
        write_json_atomic(preflight_path, record)
        print(json.dumps(record, sort_keys=True))
        return 0
    if not preflight_path.is_file() or load_json(preflight_path) != record:
        raise RuntimeError("GRPO2_EXECUTION_PREFLIGHT_RECORD_MISMATCH")
    if args.step0_only:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        output = Path(args.output_dir) / args.run_id
        output.mkdir(parents=True, exist_ok=True)
        torch.cuda.set_device(rank)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        load_model(f"cuda:{rank}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank == 0:
            write_json_atomic(output / "STEP0_INHERITED_ADAPTER_PARITY.json", {
                "status": "PASS",
                "mode": "CONTINUED_SINGLE_ADAPTER",
                "optimizer_created": False,
                "optimizer_steps": 0,
                "base_trainable_params": _MODEL_AUDIT["base_trainable_parameter_count"],
                "lora_trainable_params": _MODEL_AUDIT["lora_trainable_parameter_count"],
                "lora_tensor_count": _MODEL_AUDIT["lora_trainable_tensor_count"],
                "step0_adapter_parity": _STEP0_AUDIT,
            })
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return 0
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
