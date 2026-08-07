"""Managed OneReason dataset-version resolution.

Managed versions form a parent chain. A version may override only the logical
subdatasets it changes; all other subdatasets resolve through its parent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class DatasetVersionManifestError(ValueError):
    pass


def _read_manifest(path: str | Path) -> Mapping[str, Any] | None:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetVersionManifestError(f"Cannot read dataset version manifest {manifest_path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("versions"), dict):
        raise DatasetVersionManifestError(f"Dataset version manifest {manifest_path} must contain a versions mapping.")
    return data


def _resolve_override(
    versions: Mapping[str, Any], version: str, logical_dataset: str, seen: set[str]
) -> Mapping[str, Any] | None:
    if version in seen:
        raise DatasetVersionManifestError(f"Cycle while resolving dataset version chain: {sorted(seen | {version})}")
    entry = versions.get(version)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise DatasetVersionManifestError(f"Version {version!r} must be an object.")
    overrides = entry.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise DatasetVersionManifestError(f"Version {version!r}.overrides must be a mapping.")
    override = overrides.get(logical_dataset)
    if override is None:
        extra_entries = entry.get("extra_registry_entries", {})
        if not isinstance(extra_entries, Mapping):
            raise DatasetVersionManifestError(f"Version {version!r}.extra_registry_entries must be a mapping.")
        override = extra_entries.get(logical_dataset)
    if override is not None:
        if not isinstance(override, Mapping) or not isinstance(override.get("registry_name"), str):
            raise DatasetVersionManifestError(
                f"Version {version!r} override for {logical_dataset!r} requires a registry_name."
            )
        return override
    parent = entry.get("parent")
    if parent is None:
        return None
    if not isinstance(parent, str) or not parent:
        raise DatasetVersionManifestError(f"Version {version!r}.parent must be a non-empty string or null.")
    return _resolve_override(versions, parent, logical_dataset, seen | {version})


def resolve_managed_dataset_registry_name(
    logical_dataset: str, version: str, manifest_path: str | Path
) -> str | None:
    """Return the registered physical dataset for a managed version.

    ``None`` deliberately means the requested version is not in the managed
    manifest, so legacy name construction remains backward compatible.
    """
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return None
    override = _resolve_override(manifest["versions"], version, logical_dataset, set())
    return None if override is None else override["registry_name"]


def validate_managed_dataset_version(
    version: str, logical_datasets: list[str], manifest_path: str | Path
) -> dict[str, str] | None:
    """Resolve every logical dataset, raising for an incomplete managed version."""
    manifest = _read_manifest(manifest_path)
    if manifest is None or version not in manifest["versions"]:
        return None
    resolved: dict[str, str] = {}
    for logical_dataset in logical_datasets:
        registry_name = resolve_managed_dataset_registry_name(logical_dataset, version, manifest_path)
        if registry_name is None:
            raise DatasetVersionManifestError(
                f"Managed version {version!r} does not resolve logical dataset {logical_dataset!r}."
            )
        resolved[logical_dataset] = registry_name
    return resolved
