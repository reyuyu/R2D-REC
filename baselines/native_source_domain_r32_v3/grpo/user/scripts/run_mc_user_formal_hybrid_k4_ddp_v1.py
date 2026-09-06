#!/usr/bin/env python3
"""Four-GPU candidate-parallel formal trainer for MC_USER hybrid GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from run_mc_user_e2e_smoke import cleanup_generation_cache, set_generation_mode, set_training_mode
from run_mc_user_formal_v1 import (
    FORMAL_RESUME,
    OUTPUT_ROOT,
    _append_jsonl,
    _write_json,
    assert_run_target_writable,
    candidate_evaluator_means,
    git_reproducibility_state,
    save_formal_checkpoint,
    select_formal_rows,
    summarize_metrics,
)
from run_mc_user_generation_smoke import _candidate_overlap_metadata, stop_token_ids
from run_mc_user_pilot_v1 import (
    _selected_gpu_processes,
    assert_only_lora_trainable,
    candidate_record,
    display_rollout_records,
)
from run_mc_user_real_smoke import (
    MEMORY_THRESHOLD_MIB,
    batch_to_device,
    file_sha256,
    gpu_preflight,
    load_tokenizer,
    optimizer_state_is_lora_only,
    parameter_sha256,
    read_jsonl,
    validate_paths,
    validate_trainable,
)
from run_user_grpo_smoke import render_prompt, trim_at_stop
from user_mc_batch import make_mc_policy_batch
from user_action_reward import score_action
from user_chain_reward import score_chain
from user_mc_hybrid_distributed import distributed_hybrid_optimizer_step
from user_mc_rollout import prepare_mc_scored_rollout
from user_mc_train_step import _gradient_norm


WORLD_SIZE = 4
K = 4
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
ALGORITHM = "mc_user_hybrid_grpo_v1"
REPRO_ROOT = Path(os.environ.get("MC_USER_REPRO_ROOT", "/root/onereason_final_reproduction_20260901"))
OUTPUT_ROOT = Path(
    os.environ.get("MC_USER_OUTPUT_ROOT", "/data/GRPO_USER/runs/mc_user_v1_hybrid_formal")
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_user_formal_stage1_512_hybrid_k4.json"
STRONG_PARENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "mc_user_hybrid_strongparent_lr3e7_200.json"
)
FROZEN_CONFIG = {
    "experiment_type": "formal",
    "stage": "stage1_512_hybrid_k4",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": "/data/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500",
    "parent_experiment": "GR_REC_v1",
    "parent_checkpoint_step": 1500,
    "parent_recorded_external_score": 1.3510,
    "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "prompt_count": 512,
    "action_count": 256,
    "chain_count": 256,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [128, 256, 384, 512],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}
STRONG_PARENT_CONFIG = {
    "experiment_type": "formal",
    "stage": "strongparent_lr3e7_200",
    "base_model": "/data/models/onereason-8b-pretrain-competition",
    "adapter": (
        "/root/GRPO-checkpoints/"
        "GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-250"
    ),
    "parent_experiment": "GR_REC_ThinkSample8_FullSID_v3",
    "parent_checkpoint_step": 250,
    "parent_recorded_external_score": None,
    "train_data": "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "prompt_count": 200,
    "action_count": 100,
    "chain_count": 100,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 3e-7,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [25, 50, 75, 100, 150, 200],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}
REPRO_STEP100_CONFIG = {
    "experiment_type": "formal",
    "stage": "strongparent_lr3e7_step100_repro",
    "base_model": os.environ.get("MC_USER_BASE_MODEL", "/data/models/onereason-8b-pretrain-competition"),
    "adapter": os.environ.get(
        "MC_USER_PARENT_ADAPTER",
        str(REPRO_ROOT / "outputs/03_grpo_tk/GRPO-TK-REPRO-TO250/checkpoint-250"),
    ),
    "parent_experiment": "GR_REC_ThinkSample8_FullSID_v3",
    "parent_checkpoint_step": 250,
    "parent_recorded_external_score": 1.3579,
    "train_data": os.environ.get(
        "MC_USER_TRAIN_DATA",
        "/root/reproduce_datasets/onereason_final_chain_20260901/03_user_grpo/train_3000.jsonl",
    ),
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "prompt_count": 100,
    "action_count": 50,
    "chain_count": 50,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 3e-7,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [25, 50, 75, 100],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}
GRPO3_DETERMINISM_CONFIG = {
    "experiment_type": "formal",
    "stage": "grpo3_user_from_grpo2_step300_determinism_v1",
    "base_model": "/root/rec_fdr_v43_runs/REC-FDR-V43-STRICTDET-20260904-173024/work/output",
    "adapter": (
        "/root/grpo2_think_continued_adapter_formal300_20260905/"
        "GRPO2-REC-THINK-CONTINUED-ADAPTER-FROM-GRPO1-STEP500-LR2E7-300/"
        "checkpoint-300"
    ),
    "parent_adapter_sha256": "1a9d441a8936dd8814c899193d60515143099b2824287aa87c0b6770bfd11c45",
    "parent_experiment": "GRPO2_REC_THINK_CONTINUED_SINGLE_ADAPTER",
    "parent_checkpoint_step": 300,
    "parent_recorded_external_score": None,
    "train_data": "/root/reproduce_datasets/onereason_final_chain_20260901/03_user_grpo/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "registered_dataset_name": "user_grpo",
    "registered_dataset_split": "train",
    "registered_dataset_rows": 3000,
    "runtime_seed": 20260823,
    "prompt_count": 100,
    "action_count": 50,
    "chain_count": 50,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 3e-7,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [25, 50, 75, 100],
    "resume_supported": False,
    "resume_policy": "continuous_run_only",
}
GRPO3_FORMAL_CONFIG = {
    "experiment_type": "formal",
    "stage": "grpo3_user_from_grpo2_step250_formal_v1",
    "base_model": "/root/rec_fdr_v43_runs/REC-FDR-V43-STRICTDET-20260904-173024/work/output",
    "adapter": (
        "/root/grpo2_think_continued_adapter_formal300_20260905/"
        "GRPO2-REC-THINK-CONTINUED-ADAPTER-FROM-GRPO1-STEP500-LR2E7-300/"
        "checkpoint-250"
    ),
    "parent_adapter_sha256": "4cb382ad9a18de37c52787391ab3910166b22f99b89c7c6f69c4e3a52da4b45a",
    "parent_experiment": "GRPO2_REC_THINK_CONTINUED_SINGLE_ADAPTER",
    "parent_checkpoint_step": 250,
    "probe_parent_label": "GRPO2-step250",
    "parent_selection_basis": "USER_SELECTED",
    "external_best_confirmed": False,
    "parent_recorded_external_score": None,
    "train_data": "/root/reproduce_datasets/onereason_final_chain_20260901/03_user_grpo/train_3000.jsonl",
    "train_sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "registered_dataset_name": "user_grpo",
    "registered_dataset_split": "train",
    "registered_dataset_rows": 3000,
    "runtime_seed": 20260823,
    "prompt_count": 200,
    "action_count": 100,
    "chain_count": 100,
    "selection_seed": 20260823,
    "route_schedule": "strict_alternating",
    "K": K,
    "world_size": WORLD_SIZE,
    "parallelism": "candidate_parallel",
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "learning_rate": 3e-7,
    "weight_decay": 0.0,
    "optimizer": "AdamW",
    "scheduler": "constant",
    "forward_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "sequence_weight": 1.0,
    "local_weight": 0.3,
    "checkpoint_steps": [25, 50, 75, 100, 150, 200],
    "resume_supported": True,
    "resume_policy": "complete_checkpoint_state",
}
GRPO3_FROM_GRPO1_STEP300_FORMAL_CONFIG = {
    **GRPO3_FORMAL_CONFIG,
    "stage": "grpo3_user_from_grpo1_step300_formal_v1",
    "adapter": (
        "/root/grpo1_formal_500_20260905/outputs/"
        "GRPO1-REC-BILATERAL-FULLBASE-CONSERVATIVE-R32-LR5E7-500/"
        "checkpoint-300"
    ),
    "parent_adapter_sha256": "fbae37f3892c414a7c86f2285a4568f2a5bb526c8d249616f0445bd9b90f6c19",
    "parent_experiment": "GRPO1_REC_BILATERAL_FULLBASE_CONSERVATIVE",
    "parent_checkpoint_step": 300,
    "probe_parent_label": "GRPO1-step300",
    "parent_stage": "GRPO1",
}
SUPPORTED_FROZEN_CONFIGS = {
    FROZEN_CONFIG["stage"]: FROZEN_CONFIG,
    STRONG_PARENT_CONFIG["stage"]: STRONG_PARENT_CONFIG,
    REPRO_STEP100_CONFIG["stage"]: REPRO_STEP100_CONFIG,
    GRPO3_DETERMINISM_CONFIG["stage"]: GRPO3_DETERMINISM_CONFIG,
    GRPO3_FORMAL_CONFIG["stage"]: GRPO3_FORMAL_CONFIG,
    GRPO3_FROM_GRPO1_STEP300_FORMAL_CONFIG["stage"]: GRPO3_FROM_GRPO1_STEP300_FORMAL_CONFIG,
}


class MCK4Error(RuntimeError):
    pass


def load_k4_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("stage") == "grpo3_user_pipeline_final_only_v1":
        return validate_pipeline_k4_config(config)
    expected_config = SUPPORTED_FROZEN_CONFIGS.get(config.get("stage"))
    if expected_config is None:
        raise MCK4Error(f"unsupported frozen K4 stage: {config.get('stage')}")
    keys = set(config).union(expected_config)
    mismatches = {
        key: {"actual": config.get(key), "expected": expected_config.get(key)}
        for key in sorted(keys)
        if config.get(key) != expected_config.get(key)
    }
    if mismatches:
        raise MCK4Error(f"frozen K4 config mismatch: {mismatches}")
    return config


def validate_pipeline_k4_config(config: dict[str, Any]) -> dict[str, Any]:
    prompt_count = int(config.get("prompt_count", 0))
    expected = {
        "experiment_type": "formal",
        "runtime_seed": 20260823,
        "selection_seed": 20260823,
        "route_schedule": "strict_alternating",
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "learning_rate": 3e-7,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "scheduler": "constant",
        "forward_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_weight": 1.0,
        "local_weight": 0.3,
        "registered_dataset_name": "user_grpo",
        "registered_dataset_split": "train",
        "registered_dataset_rows": 3000,
        "resume_supported": True,
        "resume_policy": "complete_checkpoint_state",
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if config.get("parent_stage") not in {"GRPO1", "GRPO2"}:
        mismatches["parent_stage"] = {
            "actual": config.get("parent_stage"),
            "expected": "GRPO1 or GRPO2",
        }
    if prompt_count <= 0:
        mismatches["prompt_count"] = {"actual": prompt_count, "expected": "positive"}
    if config.get("action_count") != (prompt_count + 1) // 2:
        mismatches["action_count"] = {
            "actual": config.get("action_count"),
            "expected": (prompt_count + 1) // 2,
        }
    if config.get("chain_count") != prompt_count // 2:
        mismatches["chain_count"] = {
            "actual": config.get("chain_count"),
            "expected": prompt_count // 2,
        }
    if config.get("checkpoint_steps") != [prompt_count]:
        mismatches["checkpoint_steps"] = {
            "actual": config.get("checkpoint_steps"),
            "expected": [prompt_count],
        }
    if mismatches:
        raise MCK4Error(f"pipeline K4 config mismatch: {mismatches}")
    return config


def validate_parent_adapter_contract(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise MCK4Error("BLOCKED_PARENT_CONTRACT: adapter_model.safetensors missing")
    with safe_open(weights, framework="pt", device="cpu") as handle:
        tensor_names = list(handle.keys())
    lora_names = [name for name in tensor_names if "lora_" in name.lower()]
    if len(tensor_names) != 504 or len(lora_names) != 504:
        raise MCK4Error(
            "BLOCKED_PARENT_CONTRACT: expected 504/504 LoRA tensors, "
            f"found {len(tensor_names)}/{len(lora_names)}"
        )
    return {
        "trainable_lora_tensor_count": len(lora_names),
        "lora_tensor_count": len(lora_names),
        "adapter_tensor_count": len(tensor_names),
    }


def validate_grpo3_parent_lineage(
    path: Path, *, expected_step: int, expected_sha256: str, expected_parent_stage: str = "GRPO2"
) -> dict[str, Any]:
    lineage_path = path / "lineage.json"
    if not lineage_path.is_file():
        raise MCK4Error("BLOCKED_PARENT_LINEAGE: lineage.json missing")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if expected_parent_stage == "GRPO1":
        checks = {
            "recipe": lineage.get("recipe") == "grpo_fullbase_conservative_v1",
            "adapter_stage": lineage.get("adapter_stage") == "GR_REC",
            "parent_mode": lineage.get("parent_mode") == "full_model",
            "step": int(lineage.get("step", -1)) == expected_step,
            "adapter_sha256": lineage.get("adapter_sha256") == expected_sha256,
        }
    elif expected_parent_stage == "GRPO2":
        checks = {
            "adapter_semantics": lineage.get("adapter_semantics") == "CONTINUED_SINGLE_ADAPTER",
            "adapter_continuation": lineage.get("adapter_continuation") is True,
            "adapter_only": lineage.get("adapter_only") is True,
            "contains_grpo1_and_grpo2_effect": lineage.get("contains_grpo1_and_grpo2_effect") is True,
            "grpo2_step": int(lineage.get("grpo2_step", -1)) == expected_step,
            "adapter_sha256": lineage.get("adapter_sha256") == expected_sha256,
            "training_resume": lineage.get("training_resume") is False,
        }
    else:
        raise MCK4Error(f"BLOCKED_PARENT_LINEAGE: unsupported parent stage {expected_parent_stage}")
    if not all(checks.values()):
        raise MCK4Error(f"BLOCKED_PARENT_LINEAGE: {checks}")
    return {
        "status": "PASS",
        "schema": lineage.get("schema"),
        "parent_stage": expected_parent_stage,
        "parent_step": expected_step,
        "adapter_sha256": expected_sha256,
        "contains_grpo1_and_grpo2_effect": expected_parent_stage == "GRPO2",
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_tensor_bytes(value: torch.Tensor, count: int = 64) -> bytes:
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return b""
    count = min(count, flat.numel())
    positions = (
        torch.zeros(1, dtype=torch.long, device=flat.device)
        if count == 1
        else torch.arange(count, dtype=torch.long, device=flat.device)
        * (flat.numel() - 1)
        // (count - 1)
    )
    sample = flat.index_select(0, positions).contiguous().cpu()
    return sample.view(torch.uint8).numpy().tobytes()


def sampled_lora_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(_sample_tensor_bytes(parameter))
        count += 1
    if count != 504:
        raise MCK4Error(f"expected 504 LoRA tensors for fingerprint, found {count}")
    return digest.hexdigest()


def gradient_evidence(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], float]:
    gradients: dict[str, dict[str, Any]] = {}
    global_squared = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or "lora_" not in name.lower():
            continue
        gradient = parameter.grad
        if gradient is None:
            gradients[name] = {"norm": 0.0, "fingerprint": "NONE"}
            continue
        norm = float(gradient.detach().float().norm().item())
        if not math.isfinite(norm):
            raise MCK4Error(f"non-finite gradient norm for {name}")
        global_squared += norm * norm
        gradients[name] = {
            "norm": norm,
            "fingerprint": hashlib.sha256(_sample_tensor_bytes(gradient)).hexdigest(),
        }
    if len(gradients) != 504:
        raise MCK4Error(f"expected 504 LoRA gradients, found {len(gradients)}")
    return gradients, math.sqrt(global_squared)


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(str(group_index).encode("ascii"))
        for key in sorted(key for key in group if key != "params"):
            digest.update(key.encode("utf-8"))
            digest.update(str(group[key]).encode("utf-8"))
        for parameter_index, parameter in enumerate(group["params"]):
            digest.update(str(parameter_index).encode("ascii"))
            for key in sorted(optimizer.state.get(parameter, {})):
                value = optimizer.state[parameter][key]
                digest.update(key.encode("utf-8"))
                if torch.is_tensor(value):
                    digest.update(_sample_tensor_bytes(value, count=32))
                else:
                    digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def rng_fingerprint(device: torch.device) -> dict[str, str]:
    import numpy as np

    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256()
    numpy_digest.update(str(numpy_state[0]).encode("ascii"))
    numpy_digest.update(numpy_state[1].tobytes())
    numpy_digest.update(str(numpy_state[2:]).encode("ascii"))
    return {
        "python": _canonical_sha256(random.getstate()),
        "numpy": numpy_digest.hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": hashlib.sha256(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
        ).hexdigest(),
    }


def seed_deterministic_runtime(seed: int) -> None:
    import numpy as np
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def validate_checkpoint_root(path: Path, *, minimum_free_bytes: int = 2 * 1024**3) -> Path:
    declared = str(path).replace("\\", "/")
    if declared == "/data" or declared.startswith("/data/"):
        raise MCK4Error("checkpoint-root must not be under /data")
    root = path.expanduser().resolve()
    if root == Path("/data") or Path("/data") in root.parents:
        raise MCK4Error("checkpoint-root must not be under /data")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise MCK4Error("checkpoint-root is not writable")
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    if free < minimum_free_bytes:
        raise MCK4Error("checkpoint-root has insufficient free space")
    return root


def candidate_seed(selection_seed: int, prompt_step: int, sample_id: str, candidate_index: int) -> int:
    raw = f"{selection_seed}:{prompt_step}:{sample_id}:{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def probe_queue_value(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frozen = FROZEN_CONFIG if config is None else config
    parent_label = frozen.get("probe_parent_label") or (
        "Parent"
        if frozen["parent_experiment"] == "GR_REC_ThinkSample8_FullSID_v3"
        else "BETA"
    )
    return {
        "status": "PENDING_TRAINING_CHECKPOINTS",
        "mode": "POST_TRAINING_INFERENCE_ONLY",
        "training_blocked_by_probe": False,
        "items": [
            {"step": step, "label": parent_label if step == 0 else str(step), "status": "pending" if step == 0 else "waiting", "available_for_probe": step == 0}
            for step in (0, *frozen["checkpoint_steps"])
        ],
    }


def update_probe_queue(queue: Mapping[str, Any], checkpoint_step: int) -> dict[str, Any]:
    updated = json.loads(json.dumps(queue))
    found = False
    for item in updated["items"]:
        if int(item["step"]) == int(checkpoint_step):
            item["status"] = "ready"
            item["available_for_probe"] = True
            found = True
    if not found:
        raise MCK4Error(f"probe queue lacks checkpoint {checkpoint_step}")
    return updated


def _all_gpu_preflight(threshold: int, checker: Callable[..., Mapping[str, Any]] = gpu_preflight) -> list[dict[str, Any]]:
    return [dict(checker(gpu_id, threshold)) for gpu_id in range(WORLD_SIZE)]


def build_manifest(
    run_id: str,
    config_path: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    git_commit: str,
    *,
    smoke_prompts: int = 0,
) -> dict[str, Any]:
    selected_count = smoke_prompts or int(config["prompt_count"])
    selected = list(rows[:selected_count])
    expected_action = (selected_count + 1) // 2 if smoke_prompts else int(config["action_count"])
    expected_chain = selected_count // 2 if smoke_prompts else int(config["chain_count"])
    if len(selected) != selected_count:
        raise MCK4Error("selected prompt count is smaller than frozen contract")
    if (
        sum(row["route"] == "action" for row in selected) != expected_action
        or sum(row["route"] == "chain" for row in selected) != expected_chain
    ):
        raise MCK4Error("selected route counts violate frozen contract")
    return {
        "status": "READY_TO_EXECUTE",
        "run_kind": "user_grpo",
        "algorithm": ALGORITHM,
        "experiment_type": "formal_k4_smoke" if smoke_prompts else "formal",
        "stage": f"{config['stage']}_smoke" if smoke_prompts else config["stage"],
        "run_id": run_id,
        "base_model": config["base_model"],
        "adapter": config["adapter"],
        "parent_experiment": config["parent_experiment"],
        "parent_checkpoint_step": config["parent_checkpoint_step"],
        "parent_selection_basis": config.get("parent_selection_basis"),
        "external_best_confirmed": config.get("external_best_confirmed"),
        "parent_recorded_external_score": config["parent_recorded_external_score"],
        "probe_parent_label": config.get("probe_parent_label") or (
            "Parent"
            if config["parent_experiment"] == "GR_REC_ThinkSample8_FullSID_v3"
            else "BETA"
        ),
        "probe_parent_adapter": config["adapter"] if config["parent_experiment"] == "GR_REC_ThinkSample8_FullSID_v3" else None,
        "train_data": config["train_data"],
        "train_sha256": config["train_sha256"],
        "registered_dataset_name": config.get("registered_dataset_name"),
        "registered_dataset_split": config.get("registered_dataset_split"),
        "registered_dataset_rows": config.get("registered_dataset_rows"),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "prompt_count": len(selected),
        "smoke_prompts": smoke_prompts,
        "max_steps": len(selected),
        "action_count": sum(row["route"] == "action" for row in selected),
        "chain_count": sum(row["route"] == "chain" for row in selected),
        "selection_seed": config["selection_seed"],
        "route_schedule": config["route_schedule"],
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_new_tokens": config["max_new_tokens"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "optimizer": config.get("optimizer", "AdamW"),
        "scheduler": config.get("scheduler", "none"),
        "forward_batch_size": config["forward_batch_size"],
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "sequence_weight": config["sequence_weight"],
        "local_weight": config["local_weight"],
        "runtime_seed": config.get("runtime_seed"),
        "checkpoint_steps": [] if smoke_prompts else list(config["checkpoint_steps"]),
        "resume_supported": bool(config.get("resume_supported", False)),
        "resume_policy": config.get("resume_policy", "continuous_run_only"),
        "FORMAL_RESUME": (
            "SUPPORTED" if config.get("resume_supported") else FORMAL_RESUME
        ),
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
        "fresh_lora": False,
        "fresh_optimizer": True,
        "training_resume_from_grpo2": False,
        "prompts": [
            {"prompt_step": index, "route": row["route"], "sample_id": row["sample_id"]}
            for index, row in enumerate(selected, 1)
        ],
    }


def run_preflight(args: argparse.Namespace, *, gpu_checker: Callable[..., Mapping[str, Any]] = gpu_preflight, git_checker: Callable[..., Mapping[str, Any]] = git_reproducibility_state) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = load_k4_config(config_path)
    paths = validate_paths(Path(config["base_model"]), Path(config["adapter"]), Path(config["train_data"]))
    parent_contract = validate_parent_adapter_contract(Path(config["adapter"]))
    if paths["train_sha256"] != config["train_sha256"]:
        raise MCK4Error("frozen train SHA mismatch")
    repo_root = Path(__file__).resolve().parents[5]
    git_state = dict(git_checker(repo_root))
    rows = select_formal_rows(read_jsonl(Path(config["train_data"])), int(config["selection_seed"]))
    if args.smoke_prompts < 0 or args.smoke_prompts > int(config["prompt_count"]):
        raise MCK4Error("smoke-prompts must be between 0 and prompt_count")
    if config["stage"] == GRPO3_DETERMINISM_CONFIG["stage"] and (
        args.smoke_prompts != 5 or not args.determinism_evidence
    ):
        raise MCK4Error("GRPO3 determinism stage only permits a five-step evidence smoke")
    gpus = _all_gpu_preflight(args.memory_threshold_mib, checker=gpu_checker)
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise MCK4Error("run-id contains unsupported characters")
    run_dir = args.output_root / args.run_id
    checkpoint_root = validate_checkpoint_root(args.checkpoint_root)
    assert_run_target_writable(run_dir)
    config_sha256 = file_sha256(config_path)
    parent_sha256 = file_sha256(Path(config["adapter"]) / "adapter_model.safetensors")
    if config.get("parent_adapter_sha256") and parent_sha256 != config["parent_adapter_sha256"]:
        raise MCK4Error("parent adapter SHA256 mismatch")
    if config.get("registered_dataset_rows"):
        row_count = sum(1 for _ in Path(config["train_data"]).open("r", encoding="utf-8"))
        if row_count != int(config["registered_dataset_rows"]):
            raise MCK4Error("registered dataset row count mismatch")
    parent_lineage = None
    if config["stage"] in {
        GRPO3_FORMAL_CONFIG["stage"],
        GRPO3_FROM_GRPO1_STEP300_FORMAL_CONFIG["stage"],
        "grpo3_user_pipeline_final_only_v1",
    }:
        parent_lineage = validate_grpo3_parent_lineage(
            Path(config["adapter"]),
            expected_step=int(config["parent_checkpoint_step"]),
            expected_sha256=parent_sha256,
            expected_parent_stage=str(config.get("parent_stage", "GRPO2")),
        )
    manifest = build_manifest(args.run_id, config_path, config_sha256, config, rows, str(git_state["git_commit"]), smoke_prompts=args.smoke_prompts)
    manifest["parent_adapter_sha256"] = parent_sha256
    manifest["parent_lineage"] = parent_lineage
    manifest["determinism_evidence_enabled"] = bool(args.determinism_evidence)
    manifest["checkpoint_root"] = str(checkpoint_root)
    manifest["checkpoint_run_dir"] = str(checkpoint_root / args.run_id)
    existing = run_dir / "manifest.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous != manifest:
            raise MCK4Error("BLOCKED_RUN_ID_CONTRACT_MISMATCH")
    _write_json(existing, manifest)
    preflight = {
        "status": "READY_TO_EXECUTE",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "gpus": gpus,
        "git_commit": git_state["git_commit"],
        "working_tree_clean": True,
        "config_sha256": config_sha256,
        "train_sha256": config["train_sha256"],
        "prompt_count": manifest["prompt_count"],
        "action_count": manifest["action_count"],
        "chain_count": manifest["chain_count"],
        "K": K,
        "world_size": WORLD_SIZE,
        "parallelism": "candidate_parallel",
        "checkpoint_root": str(checkpoint_root),
        "parent_contract": parent_contract,
        "parent_adapter_sha256": parent_sha256,
        "parent_lineage": parent_lineage,
        "registered_dataset_used_by_trainer": bool(config.get("registered_dataset_name")),
        "execute_required": True,
    }
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(run_dir / "evaluations" / "user_light_probe" / "probe_queue.json", probe_queue_value(config))
    return preflight


def _load_execute_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    run_dir = args.output_root / args.run_id
    preflight_path, manifest_path = run_dir / "preflight.json", run_dir / "manifest.json"
    if not preflight_path.is_file() or not manifest_path.is_file():
        raise MCK4Error("execute requires existing preflight.json and manifest.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY_TO_EXECUTE" or manifest.get("git_commit") != preflight.get("git_commit"):
        raise MCK4Error("preflight contract mismatch")
    if manifest.get("K") != K or manifest.get("world_size") != WORLD_SIZE or manifest.get("parallelism") != "candidate_parallel":
        raise MCK4Error("execute manifest is not K4 candidate-parallel")
    if manifest.get("smoke_prompts", 0) != args.smoke_prompts:
        raise MCK4Error("execute smoke-prompts differs from preflight")
    if bool(manifest.get("determinism_evidence_enabled")) != bool(args.determinism_evidence):
        raise MCK4Error("execute determinism-evidence differs from preflight")
    checkpoint_root = validate_checkpoint_root(args.checkpoint_root)
    if str(checkpoint_root) != manifest.get("checkpoint_root"):
        raise MCK4Error("execute checkpoint-root differs from preflight")
    config_path = args.config.expanduser().resolve()
    config = load_k4_config(config_path)
    if validate_parent_adapter_contract(Path(config["adapter"])) != preflight.get("parent_contract"):
        raise MCK4Error("execute parent adapter contract differs from preflight")
    if file_sha256(Path(config["adapter"]) / "adapter_model.safetensors") != preflight.get("parent_adapter_sha256"):
        raise MCK4Error("execute parent adapter SHA differs from preflight")
    if file_sha256(config_path) != manifest.get("config_sha256"):
        raise MCK4Error("execute config SHA differs from preflight")
    if file_sha256(Path(config["train_data"])) != manifest.get("train_sha256"):
        raise MCK4Error("execute train SHA differs from preflight")
    repo_root = Path(__file__).resolve().parents[5]
    git_state = git_reproducibility_state(repo_root)
    if git_state["git_commit"] != manifest.get("git_commit"):
        raise MCK4Error("execute git commit differs from preflight")
    selected_ids = [item["sample_id"] for item in manifest["prompts"]]
    all_rows = {str(row["sample_id"]): row for row in read_jsonl(Path(config["train_data"]))}
    rows = [dict(all_rows[sample_id]) for sample_id in selected_ids]
    return preflight, config, rows, run_dir


def validate_resume_checkpoint(
    checkpoint: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    required = [
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
        "scheduler.pt", "trainer_state.json", "training_args.bin",
        "formal_state.json", "lineage.json", "checkpoint_manifest.json",
        *[f"rng_state_{rank}.pth" for rank in range(WORLD_SIZE)],
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise MCK4Error(f"resume checkpoint missing files: {missing}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    formal_state = json.loads((checkpoint / "formal_state.json").read_text(encoding="utf-8"))
    lineage = json.loads((checkpoint / "lineage.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    prompt_step = int(trainer_state.get("prompt_step", -1))
    optimizer_step = int(trainer_state.get("optimizer_step", -1))
    expected_ids = [str(row["sample_id"]) for row in rows[:prompt_step]]
    checks = {
        "checkpoint_status": checkpoint_manifest.get("status") == "PASS",
        "resume_capable": checkpoint_manifest.get("resume_capable") is True,
        "prompt_step": prompt_step == checkpoint_manifest.get("prompt_step") == formal_state.get("prompt_step"),
        "optimizer_step": optimizer_step == checkpoint_manifest.get("global_step") == formal_state.get("optimizer_step"),
        "step_range": 0 < prompt_step < int(config["prompt_count"]),
        "max_steps": trainer_state.get("max_steps") == int(config["prompt_count"]),
        "processed_ids": formal_state.get("processed_sample_ids") == expected_ids,
        "selection_seed": formal_state.get("selection_seed") == int(config["selection_seed"]),
        "train_sha": formal_state.get("train_sha256") == config["train_sha256"],
        "lineage_schema": lineage.get("schema") == "grpo3_user_continued_adapter_lineage_v1",
        "lineage_parent": (
            lineage.get("parent_stage") == config.get("parent_stage")
            and lineage.get("parent_checkpoint_step") == config.get("parent_checkpoint_step")
            and lineage.get("parent_adapter_sha256") == config.get("parent_adapter_sha256")
        ),
        "adapter_sha": lineage.get("adapter_sha256") == file_sha256(checkpoint / "adapter_model.safetensors"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise MCK4Error(f"resume checkpoint contract mismatch: {failed}")
    return {
        "path": checkpoint,
        "prompt_step": prompt_step,
        "optimizer_step": optimizer_step,
        "records": list(trainer_state.get("log_history", [])),
        "adapter_sha256": lineage["adapter_sha256"],
    }


def restore_resume_state(
    checkpoint: Path,
    rank: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    import numpy as np

    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    rng = torch.load(checkpoint / f"rng_state_{rank}.pth", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"], device)


def load_beta_for_rank(
    config: Mapping[str, Any],
    local_rank: int,
    adapter_path: Path | None = None,
) -> torch.nn.Module:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path or config["adapter"]),
        is_trainable=True,
        local_files_only=True,
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    return model


@torch.inference_mode()
def generate_one_candidate(model: torch.nn.Module, tokenizer: Any, row: Mapping[str, Any], device: torch.device, seed: int, config: Mapping[str, Any]) -> list[int]:
    prompt = render_prompt(tokenizer, row)
    encoded = tokenizer([prompt], add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt")
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    stops = stop_token_ids(tokenizer)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            num_return_sequences=1,
            max_new_tokens=int(config["max_new_tokens"]),
            eos_token_id=sorted(stops),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return trim_at_stop(output[0, prompt_width:].cpu().tolist(), stops)


def score_rank_candidate(route: str, row: Mapping[str, Any], generated_ids: Sequence[int], tokenizer: Any, candidate_index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rollout = prepare_mc_scored_rollout([row], [generated_ids], tokenizer, candidates_per_prompt=1)
    policy_batch = make_mc_policy_batch(tokenizer, rollout)
    marginal = rollout["marginal_results"][0]
    units = rollout["credit_units_per_candidate"][0]
    overlap = _candidate_overlap_metadata(units, len(generated_ids))
    raw = {
        "candidate_index": candidate_index,
        "completion": rollout["completions"][0],
        "generated_token_count": len(generated_ids),
        "format_valid": bool(marginal["valid"]),
        "full_reward": float(marginal["full_reward"]),
        "projection_required": bool(rollout["projection_per_candidate"][0]["projection_required"]),
        **{key: int(overlap[key]) for key in ("overlap_token_count", "same_sign_overlap_token_count", "mixed_sign_overlap_token_count", "max_active_units_per_token")},
    }
    if route == "action":
        evaluator = score_action(raw["completion"], dict(row))
        raw.update(f1=float(evaluator.f1), precision=float(evaluator.precision), recall=float(evaluator.recall), predicted_sid_count=len(evaluator.pred_sids_unique))
    else:
        evaluator = score_chain(raw["completion"], dict(row))
        raw["full_action_alignment"] = float(evaluator.action_f1)
        raw["full_logic_alignment"] = float(evaluator.logic_f1)
        raw["predicted_event_count"] = len(evaluator.predicted_events)
    metric_candidate = candidate_record(raw, units, marginal, route)
    rollout_record = display_rollout_records(prompt_step=0, optimizer_step=0, route=route, sample_id=str(row["sample_id"]), candidates=[raw], units_per_candidate=[units])[0]
    if route == "action":
        evaluator_fields = {
            "precision": raw["precision"],
            "recall": raw["recall"],
            "predicted_sid_count": raw["predicted_sid_count"],
        }
    else:
        evaluator_fields = {"predicted_event_count": raw["predicted_event_count"]}
    metric_candidate.update(evaluator_fields)
    rollout_record.update(evaluator_fields)
    return policy_batch, metric_candidate, rollout_record


def _gather(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    return [dict(item) for item in gathered]


def append_rank0_prompt_artifacts(
    rank: int,
    metrics_path: Path,
    rollouts_path: Path,
    metric: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
) -> None:
    if rank != 0:
        return
    _append_jsonl(metrics_path, metric)
    for rollout in rollouts:
        _append_jsonl(rollouts_path, rollout)


def validate_ddp_owner_claims(claims: Sequence[Mapping[str, Any]]) -> None:
    if len(claims) != WORLD_SIZE or any(not claim.get("nvidia_host_pids") for claim in claims):
        raise MCK4Error("GPU_OWNER_NOT_VISIBLE")
    pid_sets = [set(map(int, claim["nvidia_host_pids"])) for claim in claims]
    if len(set().union(*pid_sets)) != WORLD_SIZE:
        raise MCK4Error("GPU_FOREIGN_PROCESS_AFTER_LOAD")


def claim_ddp_gpu_process_ownership(gpu_id: int) -> dict[str, Any]:
    gpu_uuid, processes = _selected_gpu_processes(gpu_id)
    local_claim = {
        "gpu_id": gpu_id,
        "gpu_uuid": gpu_uuid,
        "nvidia_host_pids": sorted(int(item["nvidia_host_pid"]) for item in processes),
        "processes": processes,
    }
    claims = _gather(local_claim)
    validate_ddp_owner_claims(claims)
    return local_claim


def assert_ddp_gpu_process_owned(claim: Mapping[str, Any]) -> None:
    gpu_uuid, processes = _selected_gpu_processes(int(claim["gpu_id"]))
    current_pids = sorted(int(item["nvidia_host_pid"]) for item in processes)
    if gpu_uuid != claim["gpu_uuid"] or current_pids != list(claim["nvidia_host_pids"]):
        raise MCK4Error("GPU_OWNERSHIP_CHANGED")


def _rank_hashes(model: torch.nn.Module) -> list[dict[str, Any]]:
    lora_hash, lora_count = parameter_sha256(model, lora=True)
    gathered = _gather({"lora_hash": lora_hash, "lora_count": lora_count})
    if len({item["lora_hash"] for item in gathered}) != 1:
        raise MCK4Error("rank LoRA parameters diverged")
    return gathered


def _rng_checkpoint_state(
    rank: int, prompt_step: int, optimizer_step: int, device: torch.device
) -> dict[str, Any]:
    import numpy as np

    return {
        "rank": rank,
        "prompt_step": prompt_step,
        "global_step": optimizer_step,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def save_resumable_formal_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_run_dir: Path,
    prompt_step: int,
    optimizer_step: int,
    processed_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    *,
    rank: int,
    device: torch.device,
    config: Mapping[str, Any],
    config_sha256: str,
    train_sha256: str,
    git_commit: str,
) -> Path:
    checkpoint_dir = checkpoint_run_dir / "checkpoints" / f"prompt-step-{prompt_step:04d}"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        torch.save(dict(config), checkpoint_dir / "training_args.bin")
        trainer_state = {
            "global_step": optimizer_step,
            "prompt_step": prompt_step,
            "max_steps": int(config["prompt_count"]),
            "optimizer_step": optimizer_step,
            "log_history": list(records),
        }
        _write_json(checkpoint_dir / "trainer_state.json", trainer_state)
        _write_json(
            checkpoint_dir / "formal_state.json",
            {
                "prompt_step": prompt_step,
                "optimizer_step": optimizer_step,
                "processed_sample_ids": [row["sample_id"] for row in processed_rows],
                "selection_seed": int(config["selection_seed"]),
                "config_sha256": config_sha256,
                "train_sha256": train_sha256,
                "git_commit": git_commit,
                "running_metrics": summarize_metrics(records, wall_seconds),
                "resume_supported": True,
                "resume_policy": "complete_checkpoint_state",
                "FORMAL_RESUME": "SUPPORTED",
            },
        )
    dist.barrier()
    torch.save(
        _rng_checkpoint_state(rank, prompt_step, optimizer_step, device),
        checkpoint_dir / f"rng_state_{rank}.pth",
    )
    dist.barrier()
    if rank == 0:
        adapter_sha256 = file_sha256(checkpoint_dir / "adapter_model.safetensors")
        parent_stage = str(config.get("parent_stage", "GRPO2"))
        parent_step = int(config["parent_checkpoint_step"])
        lineage = {
            "schema": "grpo3_user_continued_adapter_lineage_v1",
            "stage": "GRPO3_USER",
            "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
            "adapter_weight_parent": f"{parent_stage} checkpoint-{parent_step}",
            "parent_stage": parent_stage,
            "parent_checkpoint_step": parent_step,
            "parent_adapter_sha256": config["parent_adapter_sha256"],
            "parent_selection_basis": config["parent_selection_basis"],
            "external_best_confirmed": config["external_best_confirmed"],
            "contains_grpo1_and_grpo3_effect": parent_stage == "GRPO1",
            "contains_grpo1_grpo2_and_grpo3_effect": parent_stage == "GRPO2",
            "fresh_lora": False,
            "fresh_optimizer": True,
            "training_resume_from_grpo2": False,
            "training_resume_from_parent": False,
            "adapter_only": True,
            "resume_supported": True,
            "grpo3_prompt_step": prompt_step,
            "grpo3_optimizer_step": optimizer_step,
            "adapter_sha256": adapter_sha256,
            "base_model": config["base_model"],
            "dataset_sha256": train_sha256,
            "config_sha256": config_sha256,
            "code_commit": git_commit,
        }
        _write_json(checkpoint_dir / "lineage.json", lineage)
        required = [
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            *[f"rng_state_{value}.pth" for value in range(WORLD_SIZE)],
            "formal_state.json",
            "lineage.json",
        ]
        missing = [name for name in required if not (checkpoint_dir / name).is_file()]
        if missing:
            raise MCK4Error(f"resumable checkpoint missing files: {missing}")
        forbidden = [
            item.name
            for item in checkpoint_dir.iterdir()
            if item.name == "model.safetensors"
            or item.name.startswith("model-")
            or item.name.startswith("pytorch_model")
        ]
        if forbidden:
            raise MCK4Error(f"formal checkpoint contains base weights: {forbidden}")
        _write_json(
            checkpoint_dir / "checkpoint_manifest.json",
            {
                "schema": "grpo3_user_resumable_checkpoint_v1",
                "status": "PASS",
                "prompt_step": prompt_step,
                "global_step": optimizer_step,
                "max_steps": int(config["prompt_count"]),
                "adapter_only": True,
                "resume_capable": True,
                "files": [
                    {
                        "name": name,
                        "size": (checkpoint_dir / name).stat().st_size,
                        "sha256": file_sha256(checkpoint_dir / name),
                    }
                    for name in required
                ],
            },
        )
    dist.barrier()
    return checkpoint_dir


def execute_distributed(args: argparse.Namespace) -> dict[str, Any] | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise MCK4Error("execute requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise MCK4Error("execute requires torchrun world_size=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank not in range(WORLD_SIZE):
        raise MCK4Error("LOCAL_RANK must be 0..3")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # This is a single-node job. Binding bootstrap traffic to loopback avoids
    # the development container's unroutable external interface.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    preflight, config, rows, run_dir = _load_execute_contract(args)
    checkpoint_run_dir = Path(preflight["checkpoint_root"]) / args.run_id
    if config.get("runtime_seed") is not None:
        seed_deterministic_runtime(int(config["runtime_seed"]))
    dist.barrier()
    tokenizer = load_tokenizer(config["base_model"])
    resume = (
        validate_resume_checkpoint(args.resume_from_checkpoint, config, rows)
        if args.resume_from_checkpoint is not None else None
    )
    model = load_beta_for_rank(config, local_rank, resume["path"] if resume else None)
    trainable = validate_trainable(model)
    if len(trainable) != 504:
        raise MCK4Error("K4 formal requires 504 LoRA trainable tensors")
    dist.barrier()
    owner = claim_ddp_gpu_process_ownership(local_rank)
    initial_base_hash = parameter_sha256(model, lora=False)[0] if rank == 0 else None
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    _rank_hashes(ddp.module)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = None
    if config.get("scheduler") == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    if resume is not None:
        if scheduler is None:
            raise MCK4Error("resume requires a checkpointed scheduler")
        restore_resume_state(resume["path"], rank, device, optimizer, scheduler)
    records: list[dict[str, Any]] = list(resume["records"]) if resume else []
    optimizer_step = int(resume["optimizer_step"]) if resume else 0
    starting_prompt_step = int(resume["prompt_step"]) if resume else 0
    started = time.perf_counter()
    metrics_path, rollouts_path = run_dir / "metrics.jsonl", run_dir / "rollouts.jsonl"
    evidence_path = run_dir / "determinism_evidence.jsonl"
    initial_lora_fingerprint = sampled_lora_fingerprint(ddp.module)
    queue_path = run_dir / "evaluations" / "user_light_probe" / "probe_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    try:
        for prompt_step, row in enumerate(rows[starting_prompt_step:], starting_prompt_step + 1):
            assert_ddp_gpu_process_owned(owner)
            seed = candidate_seed(int(config["selection_seed"]), prompt_step, str(row["sample_id"]), rank)
            prompt = render_prompt(tokenizer, row)
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_token_sha256 = _canonical_sha256(prompt_token_ids)
            rng_before = rng_fingerprint(device) if args.determinism_evidence else None
            set_generation_mode(ddp.module)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            generation_started = time.perf_counter()
            generated_ids = generate_one_candidate(ddp.module, tokenizer, row, device, seed, config)
            torch.cuda.synchronize(device)
            generation_wall = time.perf_counter() - generation_started
            generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            policy_batch, metric_candidate, rollout_record = score_rank_candidate(str(row["route"]), row, generated_ids, tokenizer, rank)
            cleanup_generation_cache(device)
            set_training_mode(ddp.module)
            training_batch = batch_to_device(policy_batch, device)
            torch.cuda.reset_peak_memory_stats(device)
            training_started = time.perf_counter()
            update = distributed_hybrid_optimizer_step(
                ddp,
                optimizer,
                training_batch,
                metric_candidate["reward"],
                training_batch["credit_units_per_candidate"],
                sequence_weight=float(config["sequence_weight"]),
                local_weight=float(config["local_weight"]),
                forward_batch_size=int(config["forward_batch_size"]),
            )
            torch.cuda.synchronize(device)
            training_wall = time.perf_counter() - training_started
            training_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if update["optimizer_step_performed"]:
                optimizer_step += 1
                if scheduler is not None:
                    scheduler.step()
            if args.determinism_evidence:
                per_lora_gradients, measured_grad_norm = gradient_evidence(ddp.module)
                step_evidence = {
                    "rank": rank,
                    "sample_id": str(row["sample_id"]),
                    "user_id": row.get("user_id"),
                    "route": str(row["route"]),
                    "candidate_seed": seed,
                    "prompt_token_sha256": prompt_token_sha256,
                    "generation_token_ids": list(generated_ids),
                    "generation_token_sha256": _canonical_sha256(list(generated_ids)),
                    "reward": float(metric_candidate["reward"]),
                    "group_rewards": list(update["group_rewards"]),
                    "sequence_advantage": float(update["sequence_advantage"]),
                    "policy_loss": float(update["total_loss"]),
                    "sequence_loss": float(update["sequence_loss"]),
                    "local_loss": float(update["local_loss"]),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "per_lora_gradients": per_lora_gradients,
                    "global_grad_norm": float(update["grad_norm"]),
                    "per_lora_norm_aggregate": measured_grad_norm,
                    "optimizer_fingerprint": optimizer_fingerprint(optimizer),
                    "rng_before": rng_before,
                    "rng_after": rng_fingerprint(device),
                    "lora_parameter_fingerprint": sampled_lora_fingerprint(ddp.module),
                    "optimizer_step_performed": bool(update["optimizer_step_performed"]),
                }
            else:
                step_evidence = None
            local = {
                "rank": rank,
                "candidate_seed": seed,
                "candidate": metric_candidate,
                "rollout": rollout_record,
                "metadata": {
                    "active_unit_count": update["local_objective_metadata"]["active_unit_count"],
                    "active_token_count": update["local_objective_metadata"]["active_token_count"],
                    "active_token_assignment_count": update["local_objective_metadata"]["active_token_assignment_count"],
                },
                "hybrid": {key: update[key] for key in (
                    "local_reward", "group_reward_mean", "group_reward_std",
                    "group_reward_spread", "sequence_advantage", "sequence_loss",
                    "local_loss", "total_loss",
                )},
                "grad_norm": float(update["grad_norm"]),
                "generation_wall_seconds": generation_wall,
                "training_wall_seconds": training_wall,
                "generation_peak_vram_mib": generation_peak,
                "training_peak_vram_mib": training_peak,
            }
            gathered = _gather(local)
            gathered_evidence = _gather(step_evidence) if step_evidence is not None else []
            if rank == 0:
                candidates = [item["candidate"] for item in sorted(gathered, key=lambda value: value["rank"])]
                for candidate, item in zip(candidates, sorted(gathered, key=lambda value: value["rank"])):
                    candidate.update(item["hybrid"])
                evaluator = candidate_evaluator_means(str(row["route"]), candidates)
                hybrid_mean = {
                    key: sum(float(item["hybrid"][key]) for item in gathered) / K
                    for key in ("sequence_loss", "local_loss", "total_loss")
                }
                if row["route"] == "action":
                    route_means = {
                        "candidate_mean_precision": sum(float(item["precision"]) for item in candidates) / K,
                        "candidate_mean_recall": sum(float(item["recall"]) for item in candidates) / K,
                        "candidate_mean_sid_count": sum(float(item["predicted_sid_count"]) for item in candidates) / K,
                        "candidate_mean_total": None,
                        "candidate_mean_event_count": None,
                    }
                else:
                    route_means = {
                        "candidate_mean_precision": None,
                        "candidate_mean_recall": None,
                        "candidate_mean_sid_count": None,
                        "candidate_mean_total": sum(float(item["reward"]) for item in candidates) / K,
                        "candidate_mean_event_count": sum(float(item["predicted_event_count"]) for item in candidates) / K,
                    }
                record = {
                    "prompt_step": prompt_step,
                    "optimizer_step": optimizer_step,
                    "step": prompt_step,
                    "route": row["route"],
                    "sample_id": row["sample_id"],
                    "candidates": candidates,
                    **evaluator,
                    "active_unit_count": sum(item["metadata"]["active_unit_count"] for item in gathered),
                    "active_token_count": sum(item["metadata"]["active_token_count"] for item in gathered),
                    "active_token_assignment_count": sum(item["metadata"]["active_token_assignment_count"] for item in gathered),
                    "algorithm": ALGORITHM,
                    "group_reward_mean": float(update["group_reward_mean"]),
                    "group_reward_std": float(update["group_reward_std"]),
                    "group_reward_min": min(float(item["reward"]) for item in candidates),
                    "group_reward_max": max(float(item["reward"]) for item in candidates),
                    "group_reward_spread": float(update["group_reward_spread"]),
                    **hybrid_mean,
                    **route_means,
                    "loss": hybrid_mean["total_loss"],
                    "grad_norm": float(update["grad_norm"]),
                    "skipped_update": bool(update["skipped_update"]),
                    "optimizer_step_performed": bool(update["optimizer_step_performed"]),
                    "generation_wall_seconds": max(item["generation_wall_seconds"] for item in gathered),
                    "training_wall_seconds": max(item["training_wall_seconds"] for item in gathered),
                    "generation_peak_vram_mib": max(item["generation_peak_vram_mib"] for item in gathered),
                    "training_peak_vram_mib": max(item["training_peak_vram_mib"] for item in gathered),
                }
                records.append(record)
                display_rows = []
                for item in sorted(gathered, key=lambda value: value["rank"]):
                    display = dict(item["rollout"], prompt_step=prompt_step, optimizer_step=optimizer_step, rank=item["rank"], candidate_index=item["rank"], candidate_seed=item["candidate_seed"], **item["hybrid"])
                    display["completion_token_count"] = display["generated_token_count"]
                    display["reward"] = display["full_reward"]
                    if display["route"] == "chain":
                        display["total_reward"] = display["full_reward"]
                        display["action_alignment"] = display["full_action_alignment"]
                        display["logic_alignment"] = display["full_logic_alignment"]
                    display_rows.append(display)
                append_rank0_prompt_artifacts(rank, metrics_path, rollouts_path, record, display_rows)
                if gathered_evidence:
                    ordered_evidence = sorted(gathered_evidence, key=lambda value: value["rank"])
                    evidence_record = {
                        "prompt_step": prompt_step,
                        "optimizer_step": optimizer_step,
                        "sample_id": str(row["sample_id"]),
                        "user_id": row.get("user_id"),
                        "route": str(row["route"]),
                        "prompt_token_sha256": prompt_token_sha256,
                        "batch_fingerprint": _canonical_sha256({
                            "sample_id": str(row["sample_id"]),
                            "route": str(row["route"]),
                            "prompt_token_sha256": prompt_token_sha256,
                            "candidate_seeds": [item["candidate_seed"] for item in ordered_evidence],
                            "generation_token_sha256": [item["generation_token_sha256"] for item in ordered_evidence],
                        }),
                        "lora_init_fingerprint": initial_lora_fingerprint,
                        "advantages": [item["sequence_advantage"] for item in ordered_evidence],
                        "ranks": ordered_evidence,
                    }
                    _append_jsonl(evidence_path, evidence_record)
            if prompt_step in set(config["checkpoint_steps"]) and not args.smoke_prompts:
                dist.barrier()
                _rank_hashes(ddp.module)
                if config.get("resume_supported"):
                    if scheduler is None:
                        raise MCK4Error("formal resumable checkpoint requires constant scheduler")
                    save_resumable_formal_checkpoint(
                        ddp.module,
                        optimizer,
                        scheduler,
                        checkpoint_run_dir,
                        prompt_step,
                        optimizer_step,
                        rows[:prompt_step],
                        records,
                        time.perf_counter() - started,
                        rank=rank,
                        device=device,
                        config=config,
                        config_sha256=preflight["config_sha256"],
                        train_sha256=config["train_sha256"],
                        git_commit=preflight["git_commit"],
                    )
                elif rank == 0:
                    save_formal_checkpoint(
                        ddp.module,
                        checkpoint_run_dir,
                        prompt_step,
                        optimizer_step,
                        rows[:prompt_step],
                        records,
                        time.perf_counter() - started,
                        selection_seed=int(config["selection_seed"]),
                        config_sha256=preflight["config_sha256"],
                        train_sha256=config["train_sha256"],
                        git_commit=preflight["git_commit"],
                    )
                if rank == 0:
                    queue = update_probe_queue(queue, prompt_step)
                    _write_json(queue_path, queue)
                dist.barrier()
        hashes = _rank_hashes(ddp.module)
        if rank == 0:
            final_adapter_sha256 = None
            if args.smoke_prompts:
                if optimizer_step != args.smoke_prompts:
                    raise MCK4Error(
                        f"smoke required {args.smoke_prompts} optimizer steps, got {optimizer_step}"
                    )
                final_adapter_dir = run_dir / "final_adapter"
                ddp.module.save_pretrained(final_adapter_dir, safe_serialization=True)
                final_adapter_sha256 = file_sha256(
                    final_adapter_dir / "adapter_model.safetensors"
                )
            final_base_hash = parameter_sha256(ddp.module, lora=False)[0]
            summary = summarize_metrics(records, time.perf_counter() - started)
            summary.update({
                "status": "PASS",
                "run_id": args.run_id,
                "prompt_step": len(rows),
                "optimizer_step": optimizer_step,
                "metrics_row_count": len(records),
                "rollout_row_count": len(rows) * K,
                "trainable_tensor_count": len(trainable),
                "lora_tensor_count": hashes[0]["lora_count"],
                "rank_lora_hashes_equal": True,
                "base_hash_unchanged": initial_base_hash == final_base_hash,
                "optimizer_state_lora_only": optimizer_state_is_lora_only(optimizer, trainable),
                "optimizer": config.get("optimizer", "AdamW"),
                "scheduler": config.get("scheduler", "none"),
                "learning_rate": config["learning_rate"],
                "max_steps": int(config["prompt_count"]),
                "checkpoint_steps": list(config["checkpoint_steps"]),
                "resume_supported": bool(config.get("resume_supported", False)),
                "parallelism": "candidate_parallel",
                "K": K,
                "world_size": WORLD_SIZE,
                "algorithm": ALGORITHM,
                "checkpoint_root": preflight["checkpoint_root"],
                "sequence_weight": config["sequence_weight"],
                "local_weight": config["local_weight"],
                "registered_dataset_used_by_trainer": bool(config.get("registered_dataset_name")),
                "training_semantics_changed": False,
                "determinism_evidence_row_count": len(records) if args.determinism_evidence else 0,
                "initial_lora_fingerprint": initial_lora_fingerprint,
                "final_adapter_sha256": final_adapter_sha256,
                "resumed_from_checkpoint": str(resume["path"]) if resume else None,
                "resume_start_prompt_step": starting_prompt_step,
            })
            if not summary["base_hash_unchanged"]:
                raise MCK4Error("base parameters changed")
            _write_json(run_dir / "summary.json", summary)
            return summary
        return None
    except Exception as exc:
        if rank == 0:
            _write_json(run_dir / "summary.json", {"status": "STOPPED", "run_id": args.run_id, "error": f"{type(exc).__name__}: {exc}", "prompt_step": len(records), "optimizer_step": optimizer_step})
        raise
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-prompts", type=int, default=0)
    parser.add_argument("--determinism-evidence", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.execute:
        return execute_distributed(args)
    result = run_preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("READY_TO_EXECUTE")
    return result


if __name__ == "__main__":
    main()
