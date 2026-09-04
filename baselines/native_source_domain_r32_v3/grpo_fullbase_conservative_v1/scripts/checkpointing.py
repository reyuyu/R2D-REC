"""Adapter-only checkpoint lineage and fail-closed reload validation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from transformers import TrainerCallback
except ModuleNotFoundError:  # Pure contract tests do not require Transformers.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from parent_contract import file_sha256


RECIPE = "grpo_fullbase_conservative_v1"
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_step(path: str | Path) -> int:
    match = CHECKPOINT_RE.fullmatch(Path(path).name)
    if not match:
        raise RuntimeError(f"checkpoint directory must end in checkpoint-<step>: {path}")
    return int(match.group(1))


def assert_adapter_only_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    required = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"adapter checkpoint missing files: {missing}")
    forbidden = [
        name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (checkpoint / name).exists()
    ]
    forbidden.extend(path.name for path in checkpoint.glob("model-*.safetensors"))
    if forbidden:
        raise RuntimeError(f"checkpoint unexpectedly contains base weights: {forbidden}")
    return {
        "adapter_only": True,
        "adapter_sha256": file_sha256(checkpoint / "adapter_model.safetensors"),
        "step": checkpoint_step(checkpoint),
    }


def assert_training_checkpoint(path: str | Path, world_size: int) -> dict[str, Any]:
    checkpoint = Path(path)
    adapter = assert_adapter_only_checkpoint(checkpoint)
    required = [
        "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin",
        *[f"rng_state_{rank}.pth" for rank in range(world_size)],
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"training checkpoint missing files: {missing}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != adapter["step"]:
        raise RuntimeError("trainer_state global_step does not match checkpoint step")
    return {**adapter, "training_state_complete": True, "world_size": world_size}


def build_lineage(
    checkpoint: str | Path,
    *,
    parent_mode: str,
    parent_base_sha256: str,
    config_sha256: str,
    dataset_sha256: str,
    code_commit: str,
    seed: int,
    lora_seed: int,
    stage: str = "GR_REC",
) -> dict[str, Any]:
    audit = assert_adapter_only_checkpoint(checkpoint)
    step = audit["step"]
    return {
        "schema": "grpo_adapter_lineage_v1",
        "recipe": RECIPE,
        "parent_mode": parent_mode,
        "parent_base_sha256": parent_base_sha256,
        "adapter_stage": stage,
        "adapter_sha256": audit["adapter_sha256"],
        "step": step,
        "grpo_config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "code_commit": code_commit,
        "seed": int(seed),
        "lora_initialization_seed": int(lora_seed),
        "resume_supported": step % 2 == 0,
    }


def save_lineage(checkpoint: str | Path, **kwargs: Any) -> dict[str, Any]:
    value = build_lineage(checkpoint, **kwargs)
    write_json_atomic(Path(checkpoint) / "lineage.json", value)
    return value


def validate_adapter_lineage(
    checkpoint: str | Path,
    *,
    expected_base_sha256: str,
    expected_config_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_seed: int | None = None,
    require_resumable: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    lineage_path = checkpoint / "lineage.json"
    if not lineage_path.is_file():
        raise RuntimeError(f"checkpoint lineage missing: {lineage_path}")
    value = json.loads(lineage_path.read_text(encoding="utf-8"))
    if value.get("recipe") != RECIPE:
        raise RuntimeError(f"adapter recipe mismatch: {value.get('recipe')!r}")
    if value.get("parent_mode") != "full_model":
        raise RuntimeError("adapter was not derived from a full-model parent")
    if value.get("parent_base_sha256") != expected_base_sha256:
        raise RuntimeError("adapter parent base SHA mismatch")
    actual_adapter_sha = file_sha256(adapter_weight_path(checkpoint))
    if value.get("adapter_sha256") != actual_adapter_sha:
        raise RuntimeError("adapter weight SHA does not match lineage")
    checks = (
        ("grpo_config_sha256", expected_config_sha256),
        ("dataset_sha256", expected_dataset_sha256),
        ("seed", expected_seed),
    )
    for key, expected in checks:
        if expected is not None and value.get(key) != expected:
            raise RuntimeError(f"adapter lineage {key} mismatch")
    if int(value.get("step", -1)) != checkpoint_step(checkpoint):
        raise RuntimeError("adapter lineage step mismatch")
    if require_resumable and not value.get("resume_supported"):
        raise RuntimeError("checkpoint is inside a two-iteration rollout and is not resumable")
    assert_adapter_only_checkpoint(checkpoint)
    return value


def adapter_weight_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint) / "adapter_model.safetensors"
    if not path.is_file():
        raise RuntimeError(f"adapter weights missing: {path}")
    return path


class LineageCheckpointCallback(TrainerCallback):
    def __init__(self, lineage_kwargs: Mapping[str, Any]):
        self.lineage_kwargs = dict(lineage_kwargs)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        save_lineage(checkpoint, **self.lineage_kwargs)
