"""Parent identity contracts shared by training, probe, checkpoint, and resume."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PARENT_MODES = {"adapter", "full_model"}


def validate_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ParentSpec:
    parent_mode: str
    base_model_path: str
    base_model_sha256: str
    config_sha256: str | None = None
    parent_adapter_path: str | None = None
    parent_adapter_sha256: str | None = None
    lineage: str = "unspecified"

    def __post_init__(self) -> None:
        if self.parent_mode not in PARENT_MODES:
            raise ValueError(f"parent_mode must be one of {sorted(PARENT_MODES)}")
        if not self.base_model_path:
            raise ValueError("base_model_path is required")
        if not self.base_model_sha256:
            raise ValueError("base_model_sha256 is required")
        validate_sha256(self.base_model_sha256, "base_model_sha256")
        if self.config_sha256 is not None:
            validate_sha256(self.config_sha256, "config_sha256")
        if self.parent_mode == "adapter":
            if not self.parent_adapter_path or not self.parent_adapter_sha256:
                raise ValueError("adapter mode requires parent adapter path and SHA256")
            validate_sha256(self.parent_adapter_sha256, "parent_adapter_sha256")
        elif self.parent_adapter_path is not None or self.parent_adapter_sha256 is not None:
            raise ValueError("full_model mode forbids a parent adapter")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParentSpec":
        return cls(
            parent_mode=str(value["parent_mode"]),
            base_model_path=str(value["base_model_path"]),
            base_model_sha256=str(value["base_model_sha256"]),
            config_sha256=(
                str(value["config_sha256"])
                if value.get("config_sha256") is not None
                else None
            ),
            parent_adapter_path=(
                str(value["parent_adapter_path"])
                if value.get("parent_adapter_path") is not None
                else None
            ),
            parent_adapter_sha256=(
                str(value["parent_adapter_sha256"])
                if value.get("parent_adapter_sha256") is not None
                else None
            ),
            lineage=str(value.get("lineage", "unspecified")),
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["base_model_path"] = "<runtime-supplied>"
        if value["parent_adapter_path"] is not None:
            value["parent_adapter_path"] = "<runtime-supplied>"
        return value


def model_weight_path(base_model_path: str | Path) -> Path:
    path = Path(base_model_path)
    single = path / "model.safetensors"
    if single.is_file():
        return single
    raise RuntimeError(
        "this recipe requires one canonical model.safetensors file for exact identity"
    )


def adapter_weight_path(adapter_path: str | Path) -> Path:
    path = Path(adapter_path) / "adapter_model.safetensors"
    if not path.is_file():
        raise RuntimeError(f"adapter weights missing: {path}")
    return path


def validate_parent_files(spec: ParentSpec, *, hash_weights: bool = True) -> dict[str, Any]:
    base = Path(spec.base_model_path)
    required = [
        base / "config.json",
        base / "tokenizer.json",
        base / "tokenizer_config.json",
        base / "generation_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"parent files missing: {missing}")
    config_sha = file_sha256(base / "config.json")
    if spec.config_sha256 is not None and config_sha != spec.config_sha256:
        raise RuntimeError(
            f"base config SHA mismatch: expected {spec.config_sha256}, got {config_sha}"
        )
    result: dict[str, Any] = {
        "parent_mode": spec.parent_mode,
        "base_model_path": str(base.resolve()),
        "base_model_sha256": spec.base_model_sha256,
        "config_sha256": config_sha,
        "required_files": [path.name for path in required],
    }
    if spec.parent_mode == "full_model":
        weight = model_weight_path(base)
        result["model_size_bytes"] = weight.stat().st_size
        if hash_weights:
            actual = file_sha256(weight)
            if actual != spec.base_model_sha256:
                raise RuntimeError(
                    f"base model SHA mismatch: expected {spec.base_model_sha256}, got {actual}"
                )
            result["verified_model_sha256"] = actual
    else:
        adapter = Path(spec.parent_adapter_path or "")
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"adapter config missing: {adapter / 'adapter_config.json'}")
        weight = adapter_weight_path(adapter)
        actual = file_sha256(weight) if hash_weights else spec.parent_adapter_sha256
        if actual != spec.parent_adapter_sha256:
            raise RuntimeError(
                "parent adapter SHA mismatch: "
                f"expected {spec.parent_adapter_sha256}, got {actual}"
            )
        result.update(
            parent_adapter_path=str(adapter.resolve()),
            parent_adapter_sha256=actual,
        )
    return result
