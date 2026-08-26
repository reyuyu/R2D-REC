"""Rank0-written world-size-4 checkpoint with per-rank RNG restoration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
from typing import Any, Sequence
import uuid

import numpy as np
import torch
import torch.distributed as dist

from checkpoint_v1 import PILOT_RECORD_COUNT, PILOT_RECORDS_SHA256, order_metadata
from distributed_trainer_v1 import DDP_WORLD_SIZE
from training_driver_v1 import frozen_contract


SCHEMA_VERSION = 1


class DistributedCheckpointError(RuntimeError):
    pass


def object_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def capture_rank_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
    }


def rng_fingerprints(value: dict[str, Any]) -> dict[str, str]:
    return {
        name: hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest()
        if isinstance(state, torch.Tensor) else object_sha256(state)
        for name, state in value.items()
    }


def restore_rank_rng(value: dict[str, Any], device: torch.device) -> dict[str, bool]:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state(value["torch_cuda"], device)
    expected = rng_fingerprints(value)
    actual = rng_fingerprints(capture_rank_rng(device))
    return {name: digest == actual[name] for name, digest in expected.items()}


def validate_world_size(world_size: int) -> None:
    if int(world_size) != DDP_WORLD_SIZE:
        raise DistributedCheckpointError(f"distributed checkpoint requires world_size={DDP_WORLD_SIZE}")


def save_distributed_checkpoint(
    checkpoint_dir: Path, *, model: Any, optimizer: Any, driver_state: dict[str, Any],
    dataset_identity: dict[str, Any], epoch: int, next_group_index: int,
    order: dict[str, Any], device: torch.device, allow_custom_dataset: bool = False,
) -> dict[str, Any]:
    if not dist.is_initialized():
        raise DistributedCheckpointError("process group is not initialized")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    validate_world_size(world_size)
    if not allow_custom_dataset and dataset_identity != {
        "records_sha256": PILOT_RECORDS_SHA256,
        "record_count": PILOT_RECORD_COUNT,
        "unique_group_count": PILOT_RECORD_COUNT,
    }:
        raise DistributedCheckpointError("dataset identity mismatch")
    expected_count = int(dataset_identity.get("record_count", 0))
    if (not allow_custom_dataset and order.get("count") != PILOT_RECORD_COUNT) or not 0 <= next_group_index <= expected_count:
        raise DistributedCheckpointError("order/cursor mismatch")
    local_rng = capture_rank_rng(device)
    all_rng = [None] * world_size
    dist.all_gather_object(all_rng, local_rng)
    checkpoint_dir = Path(checkpoint_dir)
    if rank == 0:
        if checkpoint_dir.exists():
            raise DistributedCheckpointError("checkpoint target already exists")
        model_state = model.state_dict()
        metadata = {
            "schema_version": SCHEMA_VERSION, "world_size": world_size,
            "frozen_contract": frozen_contract(), "dataset": dataset_identity,
            "epoch_order": order, "training_cursor": {"epoch": epoch, "next_group_index": next_group_index},
            "driver_state": driver_state, "model_state_sha256": tensor_state_sha256(model_state),
            "rank_rng_fingerprints": [rng_fingerprints(value) for value in all_rng],
        }
        payload = {
            "model_state_dict": model_state, "optimizer_state_dict": optimizer.state_dict(),
            "rank_rng_states": all_rng,
        }
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir()
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            torch.save(payload, temporary / "state.pt")
            os.replace(temporary, checkpoint_dir)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    dist.barrier()
    return {"rank": rank, "world_size": world_size, "rng_fingerprints": rng_fingerprints(local_rng)}


def load_distributed_checkpoint(
    checkpoint_dir: Path, *, model: Any, optimizer: Any,
    current_dataset_identity: dict[str, Any], current_group_ids: Sequence[str], device: torch.device,
    allow_custom_dataset: bool = False, expected_order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not dist.is_initialized():
        raise DistributedCheckpointError("process group is not initialized")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    validate_world_size(world_size)
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    payload = torch.load(checkpoint_dir / "state.pt", map_location="cpu", weights_only=False)
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("world_size") != world_size:
        raise DistributedCheckpointError("checkpoint schema/world size mismatch")
    if metadata.get("frozen_contract") != frozen_contract() or metadata.get("dataset") != current_dataset_identity:
        raise DistributedCheckpointError("checkpoint frozen contract/dataset mismatch")
    if allow_custom_dataset:
        if expected_order is None or metadata["epoch_order"] != expected_order:
            raise DistributedCheckpointError("custom checkpoint order mismatch")
    else:
        _, reconstructed = order_metadata(current_group_ids, metadata["epoch_order"]["seed"])
        if metadata["epoch_order"] != reconstructed:
            raise DistributedCheckpointError("checkpoint order mismatch")
    rank_rng = payload["rank_rng_states"]
    if len(rank_rng) != world_size or len(metadata["rank_rng_fingerprints"]) != world_size:
        raise DistributedCheckpointError("per-rank RNG inventory mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restored = restore_rank_rng(rank_rng[rank], device)
    if not all(restored.values()):
        raise DistributedCheckpointError("rank RNG restore mismatch")
    model_sha = tensor_state_sha256(model.state_dict())
    if model_sha != metadata["model_state_sha256"]:
        raise DistributedCheckpointError("model restore SHA mismatch")
    dist.barrier()
    return {
        "rank": rank, "world_size": world_size, "rng_restored": restored,
        "model_state_sha256": model_sha, "model_restore_exact": True,
        "driver_state": metadata["driver_state"], **metadata["training_cursor"],
    }
