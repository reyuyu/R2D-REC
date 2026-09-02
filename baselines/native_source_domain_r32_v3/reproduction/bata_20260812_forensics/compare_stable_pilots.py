#!/usr/bin/env python3
"""Offline A/B/C and historical-direction comparison for BATA-STABLE-V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file


LABELS = ("STABLE560-A", "STABLE560-B", "STABLE560-C")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
PROJECTION_RE = re.compile(r"\.(q|k|v|o|gate|up|down)_proj$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def canonical_tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def canonical_state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(b"tensor")
            digest.update(str(item.dtype).encode())
            digest.update(json.dumps(list(item.shape)).encode())
            digest.update(tensor_bytes(item.detach().cpu().contiguous()))
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())
        else:
            raise TypeError(f"unsupported canonical state value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def load_torch_state(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def normalize_key(name: str) -> str:
    return name.replace(".default.weight", ".weight")


def load_adapter(checkpoint: Path) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    path = checkpoint / "adapter_model.safetensors"
    tensors = {normalize_key(key): value for key, value in load_file(path, device="cpu").items()}
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = float(config["lora_alpha"]) / float(config["r"])
    return tensors, scaling, {
        "file_sha256": sha256_file(path),
        "canonical_tensor_sha256": canonical_tensor_hash(tensors),
        "tensor_count": len(tensors),
        "scaling": scaling,
    }


def vector_inner(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[float, float, float]:
    if set(left) != set(right):
        raise RuntimeError("LoRA tensor key mismatch")
    dot = left_norm = right_norm = 0.0
    for name in sorted(left):
        a = left[name].double()
        b = right[name].double()
        dot += torch.sum(a * b).item()
        left_norm += torch.sum(a * a).item()
        right_norm += torch.sum(b * b).item()
    return dot, left_norm, right_norm


def metrics_from_inner(dot: float, left_norm_sq: float, right_norm_sq: float) -> dict[str, float]:
    delta_sq = max(left_norm_sq + right_norm_sq - 2.0 * dot, 0.0)
    denominator = max(math.sqrt(left_norm_sq), math.sqrt(right_norm_sq), 1e-300)
    cosine_denominator = math.sqrt(left_norm_sq * right_norm_sq)
    return {
        "cosine": dot / cosine_denominator if cosine_denominator else 1.0,
        "relative_l2": math.sqrt(delta_sq) / denominator,
        "difference_l2": math.sqrt(delta_sq),
        "left_l2": math.sqrt(left_norm_sq),
        "right_l2": math.sqrt(right_norm_sq),
    }


def adapter_pair(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    dot, left_norm, right_norm = vector_inner(left, right)
    return {
        "exact": all(torch.equal(left[name], right[name]) for name in left),
        **metrics_from_inner(dot, left_norm, right_norm),
    }


def module_prefixes(tensors: dict[str, torch.Tensor]) -> list[str]:
    suffix = ".lora_A.weight"
    prefixes = sorted(name[: -len(suffix)] for name in tensors if name.endswith(suffix))
    if len(prefixes) * 2 != len(tensors):
        raise RuntimeError("adapter contains unpaired or unexpected LoRA tensors")
    return prefixes


def effective_inner(
    left: dict[str, torch.Tensor],
    left_scale: float,
    right: dict[str, torch.Tensor],
    right_scale: float,
    prefixes: Iterable[str] | None = None,
) -> float:
    selected = list(prefixes) if prefixes is not None else module_prefixes(left)
    if set(module_prefixes(left)) != set(module_prefixes(right)):
        raise RuntimeError("effective B@A module mismatch")
    total = 0.0
    for prefix in selected:
        la = left[prefix + ".lora_A.weight"].double()
        lb = left[prefix + ".lora_B.weight"].double()
        ra = right[prefix + ".lora_A.weight"].double()
        rb = right[prefix + ".lora_B.weight"].double()
        # trace((B_l^T B_r)(A_r A_l^T)); the elementwise form needs the
        # second rank-space factor transposed.
        total += left_scale * right_scale * torch.sum((lb.T @ rb) * (ra @ la.T).T).item()
    return total


def effective_pair(
    left: dict[str, torch.Tensor], left_scale: float,
    right: dict[str, torch.Tensor], right_scale: float,
) -> dict[str, Any]:
    dot = effective_inner(left, left_scale, right, right_scale)
    left_norm = effective_inner(left, left_scale, left, left_scale)
    right_norm = effective_inner(right, right_scale, right, right_scale)
    return metrics_from_inner(dot, left_norm, right_norm)


def linear_combination_inner(
    left_terms: list[tuple[float, dict[str, torch.Tensor], float]],
    right_terms: list[tuple[float, dict[str, torch.Tensor], float]],
    prefixes: Iterable[str] | None = None,
) -> float:
    return sum(
        left_coefficient * right_coefficient * effective_inner(
            left, left_scale, right, right_scale, prefixes
        )
        for left_coefficient, left, left_scale in left_terms
        for right_coefficient, right, right_scale in right_terms
    )


def direction_metrics(
    stable: tuple[dict[str, torch.Tensor], float],
    historical_553: tuple[dict[str, torch.Tensor], float],
    historical_1106: tuple[dict[str, torch.Tensor], float],
    prefixes: Iterable[str] | None = None,
) -> dict[str, float]:
    stable_tensors, stable_scale = stable
    h553_tensors, h553_scale = historical_553
    h1106_tensors, h1106_scale = historical_1106
    current = [(1.0, stable_tensors, stable_scale), (-1.0, h553_tensors, h553_scale)]
    historical = [(1.0, h1106_tensors, h1106_scale), (-1.0, h553_tensors, h553_scale)]
    dot = linear_combination_inner(current, historical, prefixes)
    current_norm = max(linear_combination_inner(current, current, prefixes), 0.0)
    historical_norm = max(linear_combination_inner(historical, historical, prefixes), 0.0)
    progress = dot / historical_norm if historical_norm else 0.0
    residual_sq = max(current_norm - (dot * dot / historical_norm if historical_norm else 0.0), 0.0)
    cosine_denominator = math.sqrt(current_norm * historical_norm)
    return {
        "hist_update_cosine": dot / cosine_denominator if cosine_denominator else 0.0,
        "hist_progress": progress,
        "hist_residual": math.sqrt(residual_sq) / math.sqrt(historical_norm) if historical_norm else 0.0,
        "stable_update_l2": math.sqrt(current_norm),
        "historical_update_l2": math.sqrt(historical_norm),
    }


def projection_groups(prefixes: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"attention": [], "mlp": []}
    for prefix in prefixes:
        match = PROJECTION_RE.search(prefix)
        if not match:
            raise RuntimeError(f"cannot classify LoRA projection: {prefix}")
        groups["attention" if match.group(1) in {"q", "k", "v", "o"} else "mlp"].append(prefix)
        layer = LAYER_RE.search(prefix)
        if not layer:
            raise RuntimeError(f"cannot classify LoRA layer: {prefix}")
        groups.setdefault(f"layer_{int(layer.group(1)):02d}", []).append(prefix)
    return groups


def checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    log = next(
        (row for row in reversed(trainer.get("log_history", [])) if row.get("step") == 560),
        {},
    )
    optimizer = load_torch_state(checkpoint / "optimizer.pt")
    scheduler = load_torch_state(checkpoint / "scheduler.pt")
    optimizer_steps = []
    for state in optimizer.get("state", {}).values():
        step = state.get("step") if isinstance(state, dict) else None
        if torch.is_tensor(step) and step.numel() == 1:
            optimizer_steps.append(float(step.item()))
        elif isinstance(step, (int, float)):
            optimizer_steps.append(float(step))
    rng = {
        f"rank{rank}": canonical_state_hash(load_torch_state(checkpoint / f"rng_state_{rank}.pth"))
        for rank in range(4)
    }
    return {
        "global_step": trainer.get("global_step"),
        "max_steps": trainer.get("max_steps"),
        "scalar_step560": {
            key: log.get(key) for key in ("loss", "grad_norm", "learning_rate", "epoch")
        },
        "optimizer_canonical_sha256": canonical_state_hash(optimizer),
        "optimizer_adam_step_min": min(optimizer_steps) if optimizer_steps else None,
        "optimizer_adam_step_max": max(optimizer_steps) if optimizer_steps else None,
        "scheduler_canonical_sha256": canonical_state_hash(scheduler),
        "rng_canonical_sha256": rng,
    }


def historical_scalar(path: Path) -> dict[str, Any]:
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    row = next((item for item in state.get("log_history", []) if item.get("step") == 560), {})
    return {key: row.get(key) for key in ("loss", "grad_norm", "learning_rate", "epoch")}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = {
        label: args.root / "runs" / label / "output" / "checkpoint-560" for label in LABELS
    }
    for label, checkpoint in checkpoints.items():
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing {label} checkpoint-560: {checkpoint}")

    adapters: dict[str, dict[str, torch.Tensor]] = {}
    scales: dict[str, float] = {}
    adapter_metadata: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        adapters[label], scales[label], adapter_metadata[label] = load_adapter(checkpoint)
    h553, h553_scale, h553_metadata = load_adapter(args.historical_553)
    h1106, h1106_scale, h1106_metadata = load_adapter(args.historical_1106)

    states = {label: checkpoint_state(checkpoint) for label, checkpoint in checkpoints.items()}
    pairs: dict[str, Any] = {}
    for index, left_label in enumerate(LABELS):
        for right_label in LABELS[index + 1 :]:
            key = f"{left_label}_vs_{right_label}"
            pairs[key] = {
                "raw_lora": adapter_pair(adapters[left_label], adapters[right_label]),
                "effective_ba": effective_pair(
                    adapters[left_label], scales[left_label], adapters[right_label], scales[right_label]
                ),
                "optimizer_exact": states[left_label]["optimizer_canonical_sha256"]
                == states[right_label]["optimizer_canonical_sha256"],
                "scheduler_exact": states[left_label]["scheduler_canonical_sha256"]
                == states[right_label]["scheduler_canonical_sha256"],
                "rng_exact": states[left_label]["rng_canonical_sha256"]
                == states[right_label]["rng_canonical_sha256"],
            }

    groups = projection_groups(module_prefixes(h553))
    historical_direction: dict[str, Any] = {}
    for label in LABELS:
        stable = (adapters[label], scales[label])
        historical_direction[label] = {
            "aggregate": direction_metrics(stable, (h553, h553_scale), (h1106, h1106_scale)),
            "attention": direction_metrics(
                stable, (h553, h553_scale), (h1106, h1106_scale), groups["attention"]
            ),
            "mlp": direction_metrics(
                stable, (h553, h553_scale), (h1106, h1106_scale), groups["mlp"]
            ),
            "layers": {
                key: direction_metrics(
                    stable, (h553, h553_scale), (h1106, h1106_scale), prefixes
                )
                for key, prefixes in groups.items()
                if key.startswith("layer_")
            },
        }

    exact = all(
        pair["raw_lora"]["exact"]
        and pair["effective_ba"]["relative_l2"] == 0.0
        and pair["optimizer_exact"]
        and pair["scheduler_exact"]
        and pair["rng_exact"]
        for pair in pairs.values()
    )
    verdict = "STABLE560_EXACT" if exact else "STABLE560_NUMERIC"
    return {
        "verdict": verdict,
        "recipe": "BATA-STABLE-V0",
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "historical_checkpoints": {
            "step553": {"path": str(args.historical_553), **h553_metadata},
            "step1106": {"path": str(args.historical_1106), **h1106_metadata},
        },
        "adapters": adapter_metadata,
        "states": states,
        "pairwise_stability": pairs,
        "historical_direction": historical_direction,
        "historical_scalar_step560": historical_scalar(args.historical_1106),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-553", type=Path, required=True)
    parser.add_argument("--historical-1106", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
