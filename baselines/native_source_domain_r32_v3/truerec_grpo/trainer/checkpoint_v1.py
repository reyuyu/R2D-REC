"""CPU checkpoint/resume contract for the single-process TrueRec V1 driver."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Sequence
import uuid

import numpy as np
import torch

from training_driver_v1 import DriverContractError, frozen_contract


CHECKPOINT_SCHEMA_VERSION = 2
PILOT_RECORD_COUNT = 4096
PILOT_RECORDS_SHA256 = "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879"
DEFAULT_ORDER_SEED = "truerec-v1-pilot4096-epoch-order-20260826"
METADATA_FILE = "metadata.json"
STATE_FILE = "state.pt"


class CheckpointContractError(DriverContractError):
    pass


def value_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def materialize_epoch_order(group_ids: Sequence[str], seed: str = DEFAULT_ORDER_SEED) -> tuple[list[str], str]:
    ids = [str(value) for value in group_ids]
    if len(ids) != len(set(ids)):
        raise CheckpointContractError("epoch order source contains duplicate recommendation_group_id")
    ordered = sorted(ids, key=lambda group_id: (hashlib.sha256(f"{seed}|{group_id}".encode()).digest(), group_id))
    return ordered, value_sha256(ordered)


def order_metadata(group_ids: Sequence[str], seed: str = DEFAULT_ORDER_SEED) -> tuple[list[str], dict[str, Any]]:
    ordered, digest = materialize_epoch_order(group_ids, seed)
    return ordered, {"seed": seed, "sha256": digest, "count": len(ordered)}


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_boundary(driver_state: dict[str, Any]) -> None:
    if driver_state.get("failed") is not False:
        raise CheckpointContractError("failed driver state cannot be checkpointed or resumed")
    if driver_state.get("groups_in_accumulation_window") != 0:
        raise CheckpointContractError("checkpoint requires a complete optimizer boundary")


def save_checkpoint_atomic(
    checkpoint_dir: Path,
    *,
    model: Any,
    optimizer: Any,
    driver: Any,
    dataset_identity: dict[str, Any],
    epoch: int,
    next_group_index: int,
    order: dict[str, Any],
    cuda_device: torch.device | str | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise CheckpointContractError("checkpoint target already exists")
    driver_state = driver.export_state()
    _validate_boundary(driver_state)
    expected_dataset = {
        "records_sha256": PILOT_RECORDS_SHA256,
        "record_count": PILOT_RECORD_COUNT,
        "unique_group_count": PILOT_RECORD_COUNT,
    }
    if dataset_identity != expected_dataset:
        raise CheckpointContractError("only frozen Pilot4096 can be checkpointed")
    if set(order) != {"seed", "sha256", "count"} or order["count"] != PILOT_RECORD_COUNT:
        raise CheckpointContractError("epoch order metadata is not a complete Pilot4096 order")
    if not order["seed"] or not isinstance(order["sha256"], str) or len(order["sha256"]) != 64:
        raise CheckpointContractError("epoch order identity is incomplete")
    if not 0 <= next_group_index <= int(order["count"]):
        raise CheckpointContractError("training cursor outside epoch order")
    cuda_rng_states = None
    if cuda_device is not None:
        device = torch.device(cuda_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise CheckpointContractError("CUDA RNG requested without an available CUDA device")
        cuda_rng_states = [torch.cuda.get_rng_state(device).cpu()]
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "frozen_contract": frozen_contract(),
        "dataset": dict(dataset_identity),
        "epoch_order": dict(order),
        "training_cursor": {"epoch": int(epoch), "next_group_index": int(next_group_index)},
        "driver_state": driver_state,
        "cuda_rng_tested": cuda_rng_states is not None,
        "cuda_device_count_saved": 0 if cuda_rng_states is None else len(cuda_rng_states),
    }
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": cuda_rng_states,
    }
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        torch.save(payload, temporary / STATE_FILE)
        os.replace(temporary, checkpoint_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Any,
    optimizer: Any,
    driver: Any,
    current_dataset_identity: dict[str, Any],
    current_group_ids: Sequence[str],
    current_contract: dict[str, Any] | None = None,
    cuda_device: torch.device | str | None = None,
) -> dict[str, int]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path, state_path = checkpoint_dir / METADATA_FILE, checkpoint_dir / STATE_FILE
    if not checkpoint_dir.is_dir() or not metadata_path.is_file() or not state_path.is_file():
        raise CheckpointContractError("checkpoint is missing required files")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = _torch_load(state_path)
    except Exception as error:
        raise CheckpointContractError("checkpoint is corrupt") from error
    required_metadata = {
        "schema_version", "frozen_contract", "dataset", "epoch_order",
        "training_cursor", "driver_state", "cuda_rng_tested", "cuda_device_count_saved",
    }
    required_payload = {
        "model_state_dict", "optimizer_state_dict", "python_random_state",
        "numpy_random_state", "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    if set(metadata) != required_metadata or set(payload) != required_payload:
        raise CheckpointContractError("checkpoint schema fields mismatch")
    if metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError("checkpoint schema version mismatch")
    contract = frozen_contract() if current_contract is None else current_contract
    if metadata["frozen_contract"] != contract:
        raise CheckpointContractError("frozen contract mismatch")
    if metadata["dataset"] != current_dataset_identity:
        raise CheckpointContractError("Pilot4096 dataset identity mismatch")
    if current_dataset_identity != {
        "records_sha256": PILOT_RECORDS_SHA256,
        "record_count": PILOT_RECORD_COUNT,
        "unique_group_count": PILOT_RECORD_COUNT,
    }:
        raise CheckpointContractError("current dataset is not frozen Pilot4096")
    saved_order = metadata["epoch_order"]
    _, reconstructed = order_metadata(current_group_ids, str(saved_order.get("seed", "")))
    if saved_order != reconstructed:
        raise CheckpointContractError("epoch order identity mismatch")
    _validate_boundary(metadata["driver_state"])
    cuda_tested = metadata["cuda_rng_tested"]
    cuda_count = metadata["cuda_device_count_saved"]
    cuda_states = payload["torch_cuda_rng_states"]
    if cuda_tested:
        if cuda_count != 1 or not isinstance(cuda_states, list) or len(cuda_states) != 1:
            raise CheckpointContractError("CUDA RNG checkpoint metadata/payload mismatch")
        if cuda_device is None or not torch.cuda.is_available() or torch.device(cuda_device).type != "cuda":
            raise CheckpointContractError("CUDA checkpoint requires one available resume device")
    elif cuda_count != 0 or cuda_states is not None:
        raise CheckpointContractError("CPU checkpoint contains inconsistent CUDA RNG state")
    cursor = metadata["training_cursor"]
    if set(cursor) != {"epoch", "next_group_index"} or not 0 <= int(cursor["next_group_index"]) <= reconstructed["count"]:
        raise CheckpointContractError("training cursor invalid")
    # All identity and boundary gates above pass before any mutable runtime state is restored.
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        driver.import_state(metadata["driver_state"])
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        if cuda_tested:
            torch.cuda.set_rng_state(cuda_states[0], torch.device(cuda_device))
    except Exception as error:
        raise CheckpointContractError("checkpoint state restoration failed") from error
    return {"epoch": int(cursor["epoch"]), "next_group_index": int(cursor["next_group_index"])}
