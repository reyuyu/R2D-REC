#!/usr/bin/env python3
"""Install and register the immutable BETA material-aligned dataset version."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "BETA_material_aligned_v1"
PARENT = "BETA"
PRIVATE_ROOT = Path("/data/baselines/native_source_domain_r32_v3/dataset_beta_material_aligned_v1")
TARGET_ROOT = Path("/data/lf_data_versions/alltrain") / VERSION
MATERIAL_SOURCE = Path("/data/lf_data_versions/source_uploads/material_understanding_20260810.jsonl")
DATASET_INFO = Path("/app/LLaMA-Factory/data/dataset_info.json")
VERSION_MANIFEST = Path("/app/LLaMA-Factory/data/onereason_dataset_versions.json")
EXPECTED_SOURCE_SHA256 = "a29042f9b997c1defe539f482cee0390848eda2dff2037ec05435c3095e23bd0"
EXPECTED_COUNTS = {
    "material_sample": 100_000,
    "sid_bucket_canonical_no_think": 11_298,
    "sid_bucket_reverse": 29_586,
}
COMPONENT_REGISTRY = {
    "cot": "onereason_material_cot_beta_material_aligned_v1",
    "nocot": "onereason_material_nocot_beta_material_aligned_v1",
}
COMBINED_REGISTRY = "onereason_beta_material_aligned"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.register-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return temporary


def dataset_entry(path: Path) -> dict[str, Any]:
    return {
        "file_name": str(path),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
            "system": "system",
        },
    }


def material_component(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt = row["prompt"]
    if prompt.rstrip().endswith("/no_think"):
        mode = "nocot"
    elif prompt.rstrip().endswith("/think"):
        mode = "cot"
    else:
        raise ValueError("Material prompt has no terminal /think or /no_think marker")
    return mode, {
        "instruction": prompt,
        "input": "",
        "output": row["response"],
        "history": [],
        "system": row["system"],
        "data_source": row["data_source"],
        "source_segment": row["data_source"],
        "aux_metadata_json": "",
    }


def write_components(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "cot": root / "onereason_material_cot.jsonl",
        "nocot": root / "onereason_material_nocot.jsonl",
    }
    handles = {name: path.open("x", encoding="utf-8") for name, path in paths.items()}
    digests = {name: hashlib.sha256() for name in paths}
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    try:
        with MATERIAL_SOURCE.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                source = row.get("data_source")
                if source not in EXPECTED_COUNTS:
                    raise ValueError(f"Unexpected data_source at line {line_number}: {source!r}")
                mode, converted = material_component(row)
                payload = json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n"
                handles[mode].write(payload)
                digests[mode].update(payload.encode("utf-8"))
                counts[mode] += 1
                source_counts[source] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if dict(source_counts) != EXPECTED_COUNTS:
        raise ValueError(f"Material route counts differ: {dict(source_counts)}")
    return {
        mode: {
            "file_name": str(paths[mode]),
            "records": counts[mode],
            "sha256": digests[mode].hexdigest(),
        }
        for mode in paths
    }


def main() -> None:
    if TARGET_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite registered dataset path: {TARGET_ROOT}")
    if digest(MATERIAL_SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Material source SHA256 differs from the audited original")

    private_manifest = json.loads((PRIVATE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    combined_source = PRIVATE_ROOT / "onereason_beta_material_aligned.jsonl"
    if digest(combined_source) != private_manifest["sha256"]:
        raise ValueError("Private combined dataset SHA256 differs from its manifest")

    registry = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    versions = json.loads(VERSION_MANIFEST.read_text(encoding="utf-8"))
    if VERSION in versions["versions"]:
        raise FileExistsError(f"Version is already registered: {VERSION}")
    if PARENT not in versions["versions"]:
        raise KeyError(f"Parent version is not registered: {PARENT}")
    reserved = set(COMPONENT_REGISTRY.values()) | {COMBINED_REGISTRY}
    collisions = sorted(reserved & set(registry))
    if collisions:
        raise FileExistsError(f"Dataset registry names already exist: {collisions}")

    temporary_root = TARGET_ROOT.with_name(f".{VERSION}.tmp-{os.getpid()}")
    temporary_root.mkdir(parents=True)
    try:
        for name in ("onereason_beta_material_aligned.jsonl", "manifest.json", "README.md"):
            shutil.copy2(PRIVATE_ROOT / name, temporary_root / name)
        components = write_components(temporary_root)

        final_components = copy.deepcopy(components)
        for item in final_components.values():
            item["file_name"] = str(TARGET_ROOT / Path(item["file_name"]).name)
        combined_target = TARGET_ROOT / "onereason_beta_material_aligned.jsonl"
        combined_stats = {
            "file_name": str(combined_target),
            "records": private_manifest["records"],
            "sha256": private_manifest["sha256"],
        }

        installed_manifest = copy.deepcopy(private_manifest)
        installed_manifest.update(
            {
                "registered_version": VERSION,
                "registered_parent": PARENT,
                "registry_name": COMBINED_REGISTRY,
                "component_files": final_components,
            }
        )
        (temporary_root / "manifest.json").write_text(
            json.dumps(installed_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_root / "dataset_info.json").write_text(
            json.dumps({COMBINED_REGISTRY: dataset_entry(combined_target)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(TARGET_ROOT)

        updated_registry = copy.deepcopy(registry)
        updated_registry[COMBINED_REGISTRY] = dataset_entry(combined_target)
        for mode, logical_name in (("cot", "onereason_material_cot"), ("nocot", "onereason_material_nocot")):
            updated_registry[COMPONENT_REGISTRY[mode]] = dataset_entry(Path(final_components[mode]["file_name"]))

        updated_versions = copy.deepcopy(versions)
        updated_versions["versions"][VERSION] = {
            "parent": PARENT,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "description": (
                "BETA with the audited original material upload, preserved three-route data_source labels, "
                "and packaged material-domain weights. User and recommendation data inherit from BETA."
            ),
            "world_included": False,
            "overrides": {
                "onereason_material_cot": {
                    "registry_name": COMPONENT_REGISTRY["cot"],
                    **final_components["cot"],
                },
                "onereason_material_nocot": {
                    "registry_name": COMPONENT_REGISTRY["nocot"],
                    **final_components["nocot"],
                },
            },
            "extra_registry_entries": {
                COMBINED_REGISTRY: {"registry_name": COMBINED_REGISTRY, **combined_stats}
            },
            "material_contract": {
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "source_counts": EXPECTED_COUNTS,
                "domain_counts": private_manifest["material_domain_counts"],
                "domain_weights": private_manifest["material_domain_weights"],
                "projection_preserved": private_manifest["material_projection_preserved"],
            },
        }

        dataset_info_tmp = atomic_json(DATASET_INFO, updated_registry)
        version_manifest_tmp = atomic_json(VERSION_MANIFEST, updated_versions)
        backup_suffix = f".pre-{VERSION}.bak"
        shutil.copy2(DATASET_INFO, DATASET_INFO.with_name(DATASET_INFO.name + backup_suffix))
        shutil.copy2(VERSION_MANIFEST, VERSION_MANIFEST.with_name(VERSION_MANIFEST.name + backup_suffix))
        os.replace(dataset_info_tmp, DATASET_INFO)
        os.replace(version_manifest_tmp, VERSION_MANIFEST)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "version": VERSION,
                "target": str(TARGET_ROOT),
                "combined_registry": COMBINED_REGISTRY,
                "components": final_components,
                "combined": combined_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
