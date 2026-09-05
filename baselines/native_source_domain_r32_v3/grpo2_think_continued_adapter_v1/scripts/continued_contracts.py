"""Fail-closed source, data, and configuration contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SFT_MODEL_SHA256 = "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"
GRPO1_STEP500_ADAPTER_SHA256 = "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"
GRPO1_DATASET_SHA256 = "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
DATASET_SHA256 = "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
EXPECTED_ROWS = 611
EXPECTED_DOMAINS = {"ad": 160, "living": 73, "prod": 118, "video": 260}
EXPECTED_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_parent_sources(
    base_model: str | Path,
    adapter_parent: str | Path,
    parent_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_model)
    adapter_dir = Path(adapter_parent)
    base_weight = base_dir / "model.safetensors"
    adapter_weight = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    lineage_path = adapter_dir / "lineage.json"
    missing = [str(path.name) for path in (base_weight, adapter_weight, adapter_config_path, lineage_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"GRPO2_PARENT_SOURCE_FILES_MISSING: {missing}")
    base_sha = file_sha256(base_weight)
    adapter_sha = file_sha256(adapter_weight)
    contract = parent_contract or {
        "base_model_sha256": SFT_MODEL_SHA256,
        "adapter_sha256": GRPO1_STEP500_ADAPTER_SHA256,
        "adapter_step": 500,
        "adapter_dataset_sha256": GRPO1_DATASET_SHA256,
    }
    expected_base_sha = str(contract["base_model_sha256"])
    expected_adapter_sha = str(contract["adapter_sha256"])
    expected_step = int(contract["adapter_step"])
    expected_dataset_sha = str(contract["adapter_dataset_sha256"])
    if base_sha != expected_base_sha:
        raise RuntimeError(f"GRPO2_SFT_SHA_MISMATCH expected={expected_base_sha} actual={base_sha}")
    if adapter_sha != expected_adapter_sha:
        raise RuntimeError(f"GRPO2_GRPO1_ADAPTER_SHA_MISMATCH expected={expected_adapter_sha} actual={adapter_sha}")
    lineage = load_json(lineage_path)
    required_lineage = {
        "recipe": "grpo_fullbase_conservative_v1",
        "parent_mode": "full_model",
        "parent_base_sha256": expected_base_sha,
        "adapter_stage": "GR_REC",
        "adapter_sha256": expected_adapter_sha,
        "step": expected_step,
        "dataset_sha256": expected_dataset_sha,
    }
    for key, expected in required_lineage.items():
        if lineage.get(key) != expected:
            raise RuntimeError(f"GRPO2_GRPO1_LINEAGE_MISMATCH key={key}")
    adapter_config = load_json(adapter_config_path)
    if (
        adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("r") != 32
        or adapter_config.get("lora_alpha") != 64
        or adapter_config.get("bias") != "none"
        or set(adapter_config.get("target_modules") or []) != EXPECTED_LORA_TARGETS
    ):
        raise RuntimeError("GRPO2_GRPO1_ADAPTER_CONFIG_MISMATCH")
    return {
        "base_full_model_sha256": base_sha,
        "adapter_sha256": adapter_sha,
        "adapter_checkpoint": expected_step,
        "lineage": lineage,
        "adapter_config": adapter_config,
    }


def validate_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    actual = file_sha256(source)
    if actual != DATASET_SHA256:
        raise RuntimeError(f"GRPO2_DATASET_SHA_MISMATCH expected={DATASET_SHA256} actual={actual}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    group_ids = [row.get("recommendation_group_id") for row in rows]
    domains: dict[str, int] = {}
    for row in rows:
        domain = row.get("target_domain")
        domains[domain] = domains.get(domain, 0) + 1
    if len(rows) != EXPECTED_ROWS or len(set(group_ids)) != EXPECTED_ROWS:
        raise RuntimeError("GRPO2_DATASET_GROUP_TOPOLOGY_MISMATCH")
    if any(row.get("route") != "think" for row in rows):
        raise RuntimeError("GRPO2_NOTHINK_TRAINING_ROUTE_FORBIDDEN")
    if dict(sorted(domains.items())) != EXPECTED_DOMAINS:
        raise RuntimeError(f"GRPO2_DATASET_DOMAIN_MISMATCH actual={domains}")
    return {"sha256": actual, "rows": rows, "unique_groups": len(set(group_ids)), "domains": domains}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("adapter_inheritance") != "CONTINUE_PARENT_ADAPTER":
        raise RuntimeError("GRPO2_ADAPTER_INHERITANCE_MODE_REQUIRED")
    if config.get("fresh_lora") is not False or config.get("fresh_optimizer") is not True:
        raise RuntimeError("GRPO2_INITIALIZATION_SEMANTICS_DRIFT")
    if config.get("training_resume") is not False:
        raise RuntimeError("GRPO1_TO_GRPO2_IS_NOT_TRAINER_RESUME")
    dataset = config.get("dataset", {})
    expected_dataset = {
        "name": "grpo_tk_positive_groups_1946_20260829",
        "sha256": DATASET_SHA256,
        "rows": EXPECTED_ROWS,
        "route": "think",
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise RuntimeError("GRPO2_DATASET_CONFIG_DRIFT")
    if config.get("run_kind") == "grpo2_continued_adapter_pipeline_final_only":
        registry = {
            "registry_key": "recommendation_grpo_think_only",
            "split": "train",
            "registered_dataset_used_by_trainer": True,
        }
        if any(dataset.get(key) != value for key, value in registry.items()):
            raise RuntimeError("GRPO2_REGISTERED_DATASET_CONTRACT_DRIFT")
    if config.get("seeds") != {
        "training": 20260816, "dataset": 20260816, "sampler": 20260816,
        "generation": 20260816, "probe": 20260818,
    }:
        raise RuntimeError("GRPO2_SEED_CONTRACT_DRIFT")
    optimization = config.get("optimization", {})
    required_optimization = {
        "learning_rate": 2e-7, "historical_learning_rate": 1e-6,
        "weight_decay": 0.0, "lr_scheduler_type": "constant",
        "max_grad_norm": 1.0, "beta": 0.0, "epsilon": 0.2,
        "loss_type": "grpo", "scale_rewards": "group", "num_iterations": 2,
    }
    for key, expected in required_optimization.items():
        if optimization.get(key) != expected:
            raise RuntimeError(f"GRPO2_OPTIMIZATION_CONTRACT_DRIFT key={key}")
    max_steps = optimization.get("max_steps")
    pipeline = config.get("run_kind") == "grpo2_continued_adapter_pipeline_final_only"
    if not pipeline and max_steps not in (5, 20, 300):
        raise RuntimeError("GRPO2_STEPS_MUST_BE_5_20_OR_300")
    if pipeline and (not isinstance(max_steps, int) or max_steps <= 0):
        raise RuntimeError("GRPO2_PIPELINE_STEPS_MUST_BE_POSITIVE")
    formal = config.get("run_kind") == "grpo2_continued_adapter_formal_300"
    if not pipeline and formal != (max_steps == 300):
        raise RuntimeError("GRPO2_FORMAL_RUN_KIND_STEPS_MISMATCH")
    if formal:
        checkpoint = config.get("checkpoint", {})
        if checkpoint.get("steps") != [100, 150, 200, 250, 300]:
            raise RuntimeError("GRPO2_FORMAL_CHECKPOINT_SCHEDULE_DRIFT")
        if checkpoint.get("save_total_limit") != 5 or checkpoint.get("adapter_only") is not True:
            raise RuntimeError("GRPO2_FORMAL_CHECKPOINT_CONTRACT_DRIFT")
        if config.get("retention_probe", {}).get("enabled") is not False:
            raise RuntimeError("GRPO2_FORMAL_INLINE_PROBE_FORBIDDEN")
    if pipeline:
        checkpoint = config.get("checkpoint", {})
        if checkpoint.get("steps") != [max_steps]:
            raise RuntimeError("GRPO2_PIPELINE_CHECKPOINT_SCHEDULE_DRIFT")
        if checkpoint.get("save_total_limit") != 1 or checkpoint.get("adapter_only") is not True:
            raise RuntimeError("GRPO2_PIPELINE_CHECKPOINT_CONTRACT_DRIFT")
        if checkpoint.get("full_resume_state") is not True:
            raise RuntimeError("GRPO2_PIPELINE_RESUME_STATE_REQUIRED")
        parent = config.get("parent", {})
        required_parent = {
            "base_model_sha256",
            "base_config_sha256",
            "adapter_sha256",
            "adapter_step",
            "adapter_dataset_sha256",
        }
        if set(parent) != required_parent:
            raise RuntimeError("GRPO2_PIPELINE_PARENT_CONTRACT_DRIFT")
    lora = config.get("lora", {})
    if (lora.get("r"), lora.get("alpha"), lora.get("bias"), lora.get("disable_dropout")) != (32, 64, "none", True):
        raise RuntimeError("GRPO2_LORA_CONTRACT_DRIFT")
    if set(lora.get("target_modules") or []) != EXPECTED_LORA_TARGETS:
        raise RuntimeError("GRPO2_LORA_TARGETS_DRIFT")
    return config


def validate_grpo2_resume(
    checkpoint: str | Path, expected_parent_sha256: str = GRPO1_STEP500_ADAPTER_SHA256
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    lineage = load_json(checkpoint / "lineage.json")
    if lineage.get("stage") != "GRPO2_REC_THINK" or lineage.get("adapter_semantics") != "CONTINUED_SINGLE_ADAPTER":
        raise RuntimeError("GRPO2_RESUME_LINEAGE_INVALID")
    if lineage.get("adapter_initialization_source_sha256") != expected_parent_sha256:
        raise RuntimeError("GRPO2_RESUME_PARENT_ADAPTER_MISMATCH")
    if lineage.get("resume_supported") is not True:
        raise RuntimeError("GRPO2_RESUME_NOT_SUPPORTED")
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"]
    required.extend(f"rng_state_{rank}.pth" for rank in range(4))
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"GRPO2_RESUME_CHECKPOINT_INCOMPLETE: {missing}")
    state = load_json(checkpoint / "trainer_state.json")
    if int(state["global_step"]) != int(lineage.get("grpo2_optimizer_step", -1)):
        raise RuntimeError("GRPO2_RESUME_STEP_MISMATCH")
    if lineage.get("adapter_sha256") != file_sha256(checkpoint / "adapter_model.safetensors"):
        raise RuntimeError("GRPO2_RESUME_ADAPTER_SHA_MISMATCH")
    return {"step": int(state["global_step"]), "lineage": lineage}
