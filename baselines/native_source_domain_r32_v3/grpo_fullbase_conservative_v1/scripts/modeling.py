"""Model construction and trainable-parameter enforcement for both parent modes."""
from __future__ import annotations

import hashlib
import random
from contextlib import nullcontext
from typing import Any, Iterable

from checkpointing import validate_adapter_lineage
from parent_contract import ParentSpec, validate_parent_files


LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "down_proj",
    "o_proj",
    "v_proj",
    "k_proj",
    "q_proj",
    "up_proj",
    "gate_proj",
)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fresh_lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )


def enforce_lora_only_trainable(model: Any) -> dict[str, Any]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    total = sum(parameter.numel() for parameter in model.parameters())
    base_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    )
    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    )
    lora_tensors = sum(
        1
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    )
    if base_trainable != 0:
        raise RuntimeError(f"base_trainable_parameter_count must be zero, got {base_trainable}")
    if lora_trainable <= 0:
        raise RuntimeError("no trainable LoRA parameters")
    return {
        "total_parameter_count": total,
        "base_trainable_parameter_count": base_trainable,
        "lora_trainable_parameter_count": lora_trainable,
        "lora_trainable_tensor_count": lora_tensors,
    }


def trainable_parameter_ids(model: Any) -> set[int]:
    return {id(parameter) for parameter in model.parameters() if parameter.requires_grad}


def optimizer_parameter_ids(optimizer: Any) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def assert_optimizer_lora_only(model: Any, optimizer: Any) -> dict[str, int | bool]:
    expected = trainable_parameter_ids(model)
    actual = optimizer_parameter_ids(optimizer)
    if actual != expected:
        raise RuntimeError(
            "optimizer parameters do not exactly match requires_grad=True LoRA parameters"
        )
    return {
        "optimizer_lora_only": True,
        "optimizer_parameter_tensor_count": len(actual),
    }


def load_training_parent(
    spec: ParentSpec,
    *,
    device: str,
    dtype: Any,
    lora_seed: int,
    local_files_only: bool = True,
):
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_parent_files(spec, hash_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    base = AutoModelForCausalLM.from_pretrained(
        spec.base_model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=local_files_only,
    )
    if spec.parent_mode == "adapter":
        model = PeftModel.from_pretrained(
            base,
            spec.parent_adapter_path,
            is_trainable=True,
            local_files_only=local_files_only,
        )
    else:
        _set_seed(lora_seed)
        model = get_peft_model(base, fresh_lora_config())
    audit = enforce_lora_only_trainable(model)
    if audit["lora_trainable_tensor_count"] != 504:
        raise RuntimeError(
            "frozen r32 seven-module Qwen3 contract requires 504 LoRA tensors, "
            f"got {audit['lora_trainable_tensor_count']}"
        )
    return model, tokenizer, audit


def load_probe_model(
    spec: ParentSpec,
    *,
    adapter_checkpoint: str | None,
    device: str,
    dtype: Any,
    local_files_only: bool = True,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_parent_files(spec, hash_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    base = AutoModelForCausalLM.from_pretrained(
        spec.base_model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=local_files_only,
    )
    if adapter_checkpoint is None:
        if spec.parent_mode == "adapter":
            adapter_checkpoint = spec.parent_adapter_path
        else:
            base.eval()
            return base, tokenizer
    if spec.parent_mode == "full_model":
        validate_adapter_lineage(
            adapter_checkpoint,
            expected_base_sha256=spec.base_model_sha256,
        )
    model = PeftModel.from_pretrained(
        base,
        adapter_checkpoint,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def adapter_disabled(model: Any):
    method = getattr(model, "disable_adapter", None)
    return method() if method is not None else nullcontext()


def sample_positions(numel: int, count: int, device: Any = None):
    import torch

    if numel <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    count = min(count, numel)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    # Integer arithmetic avoids float32 linspace rounding an endpoint to numel
    # for very large base tensors.
    return (
        torch.arange(count, dtype=torch.long, device=device) * (numel - 1)
        // (count - 1)
    )


def _sample_bytes(tensor: Any, sample_count: int = 64) -> bytes:
    import torch

    value = tensor.detach().reshape(-1)
    if value.numel() == 0:
        return b""
    positions = sample_positions(value.numel(), sample_count, value.device)
    sampled = value.index_select(0, positions).contiguous().cpu()
    return sampled.view(torch.uint8).numpy().tobytes()


def sampled_parameter_fingerprint(
    named_parameters: Iterable[tuple[str, Any]], *, include_lora: bool
) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in named_parameters:
        if ("lora_" in name.lower()) != include_lora:
            continue
        count += 1
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(_sample_bytes(parameter))
    if count == 0:
        raise RuntimeError("parameter fingerprint selected no tensors")
    return digest.hexdigest()


BASE_FINGERPRINT_SUFFIXES = (
    "embed_tokens.weight",
    "layers.0.self_attn.q_proj.base_layer.weight",
    "layers.18.mlp.down_proj.base_layer.weight",
    "layers.35.self_attn.o_proj.base_layer.weight",
    "lm_head.weight",
)


def base_tensor_fingerprints(model: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    for suffix in BASE_FINGERPRINT_SUFFIXES:
        matches = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "lora_" not in name.lower() and name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"base fingerprint suffix {suffix!r} matched {len(matches)} parameters"
            )
        name, parameter = matches[0]
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(_sample_bytes(parameter, sample_count=256))
        values[suffix] = digest.hexdigest()
    return values


def gradient_fingerprints(model: Any) -> tuple[dict[str, dict[str, float | str]], float]:
    import math

    result: dict[str, dict[str, float | str]] = {}
    global_squared = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or "lora_" not in name.lower():
            continue
        gradient = parameter.grad
        if gradient is None:
            result[name] = {"norm": 0.0, "fingerprint": "NONE"}
            continue
        norm = float(gradient.detach().float().norm().item())
        global_squared += norm * norm
        result[name] = {
            "norm": norm,
            "fingerprint": hashlib.sha256(_sample_bytes(gradient)).hexdigest(),
        }
    return result, math.sqrt(global_squared)
