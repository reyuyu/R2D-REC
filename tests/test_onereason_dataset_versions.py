from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from llamafactory.data.multitask import resolve_multitask_dataset_name
from llamafactory.data.onereason_dataset_versions import (
    DatasetVersionManifestError,
    resolve_managed_dataset_registry_name,
    validate_managed_dataset_version,
)


LOGICAL = ["onereason_material_cot", "onereason_recommendation_cot"]


def make_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "versions": {
                    "raw_all": {
                        "parent": None,
                        "overrides": {
                            "onereason_material_cot": {"registry_name": "onereason_material_cot"},
                            "onereason_recommendation_cot": {"registry_name": "onereason_recommendation_cot"},
                        },
                    },
                    "v1": {
                        "parent": "raw_all",
                        "overrides": {
                            "onereason_material_cot": {"registry_name": "onereason_material_cot_v1"},
                        },
                    },
                    "v2_recommendation_only": {
                        "parent": "v1",
                        "overrides": {
                            "onereason_recommendation_cot": {"registry_name": "onereason_recommendation_cot_v2"},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_parent_chain_resolves_only_changed_subdataset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        make_manifest(path)
        assert resolve_managed_dataset_registry_name("onereason_material_cot", "v2_recommendation_only", path) == "onereason_material_cot_v1"
        assert resolve_managed_dataset_registry_name("onereason_recommendation_cot", "v2_recommendation_only", path) == "onereason_recommendation_cot_v2"


def test_raw_full_resolves_canonical_names() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        make_manifest(path)
        assert validate_managed_dataset_version("raw_all", LOGICAL, path) == {
            "onereason_material_cot": "onereason_material_cot",
            "onereason_recommendation_cot": "onereason_recommendation_cot",
        }


def test_unknown_version_keeps_legacy_fallback() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        make_manifest(path)
        assert resolve_managed_dataset_registry_name("onereason_material_cot", "legacy_version", path) is None


def test_incomplete_managed_version_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps({"versions": {"broken": {"parent": None, "overrides": {}}}}), encoding="utf-8")
        try:
            validate_managed_dataset_version("broken", LOGICAL, path)
        except DatasetVersionManifestError as error:
            assert "does not resolve" in str(error)
        else:
            raise AssertionError("Expected incomplete managed version to fail")


def test_multitask_resolver_uses_managed_chain_and_ignores_train98_suffix() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        make_manifest(path)
        args = SimpleNamespace(
            multitask_task_layout="legacy",
            multitask_dataset_version="v2_recommendation_only",
            multitask_dataset_version_overrides={},
            multitask_dataset_version_manifest=str(path),
            multitask_train_dataset_suffix="_train98",
        )
        assert resolve_multitask_dataset_name("onereason_material_cot", args) == "onereason_material_cot_v1"
        assert resolve_multitask_dataset_name("onereason_recommendation_cot", args) == "onereason_recommendation_cot_v2"


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASS {name}")
