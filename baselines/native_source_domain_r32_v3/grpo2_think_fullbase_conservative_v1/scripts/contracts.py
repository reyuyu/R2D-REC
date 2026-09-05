"""Fail-closed contracts for GRPO-2 parents and Think-only data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_SHA256 = "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
GRPO1_STEP300_ADAPTER_SHA256 = "fbae37f3892c414a7c86f2285a4568f2a5bb526c8d249616f0445bd9b90f6c19"
GRPO1_STEP500_ADAPTER_SHA256 = "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"
GRPO1_ADAPTER_SHA256 = GRPO1_STEP300_ADAPTER_SHA256
SFT_MODEL_SHA256 = "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"
EXPECTED_ROWS = 611
EXPECTED_DOMAINS = {"ad": 160, "living": 73, "prod": 118, "video": 260}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_model_identity(model_dir: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise RuntimeError("GRPO2_PARENT_HAS_NO_SAFETENSORS")
    records = [
        {"name": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}
        for path in files
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), records


def validate_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    actual = file_sha256(source)
    if actual != DATASET_SHA256:
        raise RuntimeError(f"GRPO2_DATASET_SHA_MISMATCH expected={DATASET_SHA256} actual={actual}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    gids = [row.get("recommendation_group_id") for row in rows]
    domains: dict[str, int] = {}
    for row in rows:
        domain = row.get("target_domain")
        domains[domain] = domains.get(domain, 0) + 1
    if len(rows) != EXPECTED_ROWS or len(set(gids)) != EXPECTED_ROWS:
        raise RuntimeError("GRPO2_DATASET_GROUP_TOPOLOGY_MISMATCH")
    if any(row.get("route") != "think" for row in rows):
        raise RuntimeError("GRPO2_TRAIN_DATA_MUST_BE_THINK_ONLY")
    if dict(sorted(domains.items())) != EXPECTED_DOMAINS:
        raise RuntimeError(f"GRPO2_DATASET_DOMAIN_MISMATCH actual={domains}")
    return {"sha256": actual, "rows": rows, "unique_groups": len(set(gids)), "domains": domains}


def validate_parent_manifest(path: str | Path, *, allow_test_parent: bool) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_parent = value.get("test_parent_only") is True
    if test_parent:
        if value.get("canonical_grpo1_parent") is not False or value.get("canonical") is not False:
            raise RuntimeError("GRPO2_TEST_PARENT_LINEAGE_FLAGS_INVALID")
        if not allow_test_parent:
            raise RuntimeError("GRPO2_NONCANONICAL_PARENT_REQUIRES_ALLOW_TEST_PARENT")
        expected_step = 300
        expected_adapter = GRPO1_STEP300_ADAPTER_SHA256
    else:
        if allow_test_parent:
            raise RuntimeError("GRPO2_CANONICAL_PARENT_MUST_NOT_USE_ALLOW_TEST_PARENT")
        if not all((
            value.get("canonical") is True,
            value.get("canonical_for_this_run") is True,
            value.get("canonical_grpo1_parent") is True,
            value.get("external_best_confirmed") is False,
            value.get("selection_basis") == "user_provisional_selection",
        )):
            raise RuntimeError("GRPO2_CANONICAL_PARENT_LINEAGE_FLAGS_INVALID")
        expected_step = 500
        expected_adapter = GRPO1_STEP500_ADAPTER_SHA256
    if value.get("source_grpo1_checkpoint") != expected_step:
        raise RuntimeError("GRPO2_GRPO1_CHECKPOINT_SELECTION_MISMATCH")
    if value.get("source_sft_model_sha256") != SFT_MODEL_SHA256:
        raise RuntimeError("GRPO2_SFT_PARENT_SHA_MISMATCH")
    if value.get("source_grpo1_adapter_sha256") != expected_adapter:
        raise RuntimeError("GRPO2_GRPO1_ADAPTER_SHA_MISMATCH")
    model_dir = manifest_path.parent
    identity, files = canonical_model_identity(model_dir)
    if value.get("canonical_model_identity") != identity or value.get("weight_files") != files:
        raise RuntimeError("GRPO2_MERGED_PARENT_IDENTITY_MISMATCH")
    if value.get("functional_parity", {}).get("status") != "PASS":
        raise RuntimeError("GRPO2_PARENT_FUNCTIONAL_PARITY_NOT_PASS")
    if value.get("standalone_reload") != "PASS":
        raise RuntimeError("GRPO2_PARENT_STANDALONE_RELOAD_NOT_PASS")
    expected_auxiliary = []
    for record in value.get("auxiliary_files", []):
        artifact = model_dir / record["name"]
        expected_auxiliary.append({
            "name": artifact.name, "size": artifact.stat().st_size, "sha256": file_sha256(artifact)
        })
    if value.get("auxiliary_files") != expected_auxiliary:
        raise RuntimeError("GRPO2_PARENT_AUXILIARY_HASH_MISMATCH")
    canonical = dict(value)
    recorded_manifest_sha = canonical.pop("canonical_manifest_sha256", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    if recorded_manifest_sha != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("GRPO2_PARENT_CANONICAL_MANIFEST_SHA_MISMATCH")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    formal = config.get("run_kind") == "grpo2_formal_300"
    expected_flags = (False, True) if formal else (True, False)
    if (config.get("test_parent_only"), config.get("canonical_grpo1_parent")) != expected_flags:
        raise RuntimeError("GRPO2_CONFIG_PARENT_FLAGS_INVALID")
    if config["dataset"] != {
        "name": "grpo_tk_positive_groups_1946_20260829",
        "sha256": DATASET_SHA256,
        "rows": EXPECTED_ROWS,
        "route": "think",
    }:
        raise RuntimeError("GRPO2_DATASET_CONFIG_DRIFT")
    expected_seeds = {
        "training": 20260816, "dataset": 20260816, "sampler": 20260816,
        "generation": 20260816, "lora_initialization": 20260905, "probe": 20260818,
    }
    if config.get("seeds") != expected_seeds:
        raise RuntimeError("GRPO2_SEED_CONTRACT_DRIFT")
    optimization = config["optimization"]
    if float(optimization["learning_rate"]) != 2e-7 or float(optimization["historical_learning_rate"]) != 1e-6:
        raise RuntimeError("GRPO2_LEARNING_RATE_CONTRACT_DRIFT")
    if optimization.get("beta") != 0.0 or optimization.get("num_iterations") != 2:
        raise RuntimeError("GRPO2_PPO_CONTRACT_DRIFT")
    if formal:
        if optimization.get("max_steps") != 300:
            raise RuntimeError("GRPO2_FORMAL_MAX_STEPS_DRIFT")
        if config.get("checkpoint", {}).get("steps") != [100, 150, 200, 250, 300]:
            raise RuntimeError("GRPO2_FORMAL_CHECKPOINT_SCHEDULE_DRIFT")
        if config.get("retention_probe", {}).get("enabled") is not False:
            raise RuntimeError("GRPO2_FORMAL_INLINE_PROBE_FORBIDDEN")
    lora = config["lora"]
    if (lora["r"], lora["alpha"], lora["bias"], lora["disable_dropout"]) != (32, 64, "none", True):
        raise RuntimeError("GRPO2_LORA_CONTRACT_DRIFT")
    return config
