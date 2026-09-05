"""Exact tensor identity checks for the inherited LoRA adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _normalized_key(name: str) -> str:
    return name.replace(".default.", ".")


def _tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()


def canonical_tensor_hash(tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def audit_loaded_adapter(model: Any, adapter_file: str | Path, *, device: str) -> dict[str, Any]:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    loaded_raw = get_peft_model_state_dict(model)
    loaded = {_normalized_key(name): tensor for name, tensor in loaded_raw.items()}
    with safe_open(str(adapter_file), framework="pt", device=device) as handle:
        source = {_normalized_key(name): handle.get_tensor(name) for name in handle.keys()}
    if len(loaded) != len(loaded_raw) or len(source) == 0:
        raise RuntimeError("GRPO2_ADAPTER_CANONICAL_KEY_COLLISION_OR_EMPTY")
    if set(loaded) != set(source):
        raise RuntimeError("GRPO2_STEP0_ADAPTER_KEY_MISMATCH")
    mismatches = []
    for name in sorted(source):
        left, right = source[name], loaded[name]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            mismatches.append(name)
            if len(mismatches) >= 8:
                break
    if mismatches:
        raise RuntimeError(f"GRPO2_STEP0_ADAPTER_TENSOR_MISMATCH: {mismatches}")
    source_hash = canonical_tensor_hash(source)
    loaded_hash = canonical_tensor_hash(loaded)
    if source_hash != loaded_hash:
        raise RuntimeError("GRPO2_STEP0_ADAPTER_CANONICAL_HASH_MISMATCH")
    return {
        "status": "PASS",
        "grpo1_to_grpo2_adapter_weight_parity": "PASS",
        "tensor_count": len(source),
        "tensor_names_exact": True,
        "tensor_shapes_exact": True,
        "tensor_dtypes_exact": True,
        "tensor_values_byte_exact": True,
        "source_canonical_tensor_sha256": source_hash,
        "loaded_canonical_tensor_sha256": loaded_hash,
    }
