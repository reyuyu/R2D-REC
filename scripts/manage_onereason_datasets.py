#!/usr/bin/env python3
"""Create and register immutable, full-training OneReason dataset versions.

Examples:
  python scripts/manage_onereason_datasets.py init-raw-all
  python scripts/manage_onereason_datasets.py create-thought-prompts \
      --version v1_thought_prompt_all --parent raw_all
  python scripts/manage_onereason_datasets.py create-recommendation-cot-complete \
      --version v2_recommendation_cot_complete_all --parent v1_thought_prompt_all
  python scripts/manage_onereason_datasets.py register-version \
      --version v3_material_clean --parent v2_recommendation_cot_complete_all \
      --patch onereason_material_cot=/data/clean/onereason_material_cot.jsonl \
      --patch onereason_material_nocot=/data/clean/onereason_material_nocot.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data" / "onereason_dataset_versions.json"
DEFAULT_DATASET_INFO = REPO / "data" / "dataset_info.json"
DEFAULT_SOURCE_DIR = Path("/data/lf_data")
DEFAULT_VERSION_ROOT = Path("/data/lf_data_versions/alltrain")
LOGICAL_DATASETS = (
    "onereason_material_cot",
    "onereason_material_nocot",
    "onereason_user_action_nocot",
    "onereason_user_chain_cot",
    "onereason_user_chain_nocot",
    "onereason_recommendation_cot",
    "onereason_world_cot",
    "onereason_world_nocot",
)
VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
REQUIRED_RECOMMENDATION_SECTIONS = ("【兴趣归纳】", "【行为模式】", "【预测总结】")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_stats(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    records = 0
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            digest.update(raw_line)
            try:
                json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {error}") from error
            records += 1
    return {"file_name": str(path), "records": records, "sha256": digest.hexdigest()}


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {"schema_version": 1, "logical_datasets": list(LOGICAL_DATASETS), "versions": {}}


def resolve_entry(manifest: dict[str, Any], version: str, logical_dataset: str, seen: set[str] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    if version in seen:
        raise ValueError(f"Version parent cycle: {sorted(seen | {version})}")
    version_entry = manifest["versions"].get(version)
    if version_entry is None:
        raise KeyError(f"Unknown version {version!r}")
    override = version_entry.get("overrides", {}).get(logical_dataset)
    if override is not None:
        return override
    parent = version_entry.get("parent")
    if parent is None:
        raise KeyError(f"Version {version!r} does not resolve {logical_dataset!r}")
    return resolve_entry(manifest, parent, logical_dataset, seen | {version})


def register_override(registry: dict[str, Any], base_name: str, target: Path, version: str, stats: dict[str, Any]) -> dict[str, Any]:
    if base_name not in registry:
        raise KeyError(f"Base dataset {base_name!r} is missing from dataset_info.json")
    registry_name = f"{base_name}_{version}"
    existing = registry.get(registry_name)
    if existing is not None and existing.get("file_name") != str(target):
        raise FileExistsError(f"Registry name {registry_name!r} already points to a different file")
    entry = copy.deepcopy(registry[base_name])
    entry["file_name"] = str(target)
    registry[registry_name] = entry
    return {"registry_name": registry_name, **stats}


def ensure_version_name(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"Invalid version name {version!r}; use letters, digits, _ and - only")


def finalize_version(
    manifest_path: Path,
    dataset_info_path: Path,
    version: str,
    parent: str | None,
    description: str,
    overrides: dict[str, dict[str, Any]],
) -> None:
    ensure_version_name(version)
    manifest = load_manifest(manifest_path)
    if version in manifest["versions"]:
        raise FileExistsError(f"Version {version!r} already exists")
    if parent is not None and parent not in manifest["versions"]:
        raise KeyError(f"Parent version {parent!r} does not exist")
    manifest["versions"][version] = {
        "parent": parent,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": description,
        "overrides": overrides,
    }
    atomic_json_write(manifest_path, manifest)
    print(f"Registered version {version} with {len(overrides)} override(s) in {manifest_path}")


def init_raw_all(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    if "raw_all" in manifest["versions"]:
        raise FileExistsError("raw_all already exists")
    registry = read_json(args.dataset_info)
    overrides: dict[str, dict[str, Any]] = {}
    for base_name in LOGICAL_DATASETS:
        source = args.source_dir / f"{base_name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        stats = file_stats(source)
        if base_name not in registry:
            raise KeyError(f"{base_name} missing from dataset_info.json")
        if registry[base_name].get("file_name") != str(source):
            raise ValueError(f"Registry entry {base_name} must point to immutable source {source}")
        overrides[base_name] = {"registry_name": base_name, **stats}
    finalize_version(args.manifest, args.dataset_info, "raw_all", None, "Immutable full source datasets; no validation split.", overrides)


def transform_thought_prompt(source: Path, target: Path, marker: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    records = appended = already_present = 0
    with source.open("rb") as input_file, target.open("wb") as output_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            sample = json.loads(raw_line)
            prompt = sample.get("input")
            if not isinstance(prompt, str):
                raise TypeError(f"{source}:{line_number} input is not a string")
            if prompt.rstrip().endswith(marker):
                already_present += 1
            else:
                sample["input"] = prompt.rstrip() + "\n" + marker
                appended += 1
            encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            output_file.write(encoded)
            digest.update(encoded)
            records += 1
    return {"file_name": str(target), "records": records, "sha256": digest.hexdigest(), "appended": appended, "already_present": already_present}


def create_thought_prompts(args: argparse.Namespace) -> None:
    ensure_version_name(args.version)
    manifest = load_manifest(args.manifest)
    registry = read_json(args.dataset_info)
    destination = args.version_root / args.version
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    overrides: dict[str, dict[str, Any]] = {}
    updated_registry = copy.deepcopy(registry)
    try:
        for base_name in LOGICAL_DATASETS:
            source_entry = resolve_entry(manifest, args.parent, base_name)
            source = Path(source_entry["file_name"])
            marker = "/think" if "_cot" in base_name else "/no_think"
            target = temporary / f"{base_name}.jsonl"
            stats = transform_thought_prompt(source, target, marker)
            final_target = destination / target.name
            stats["file_name"] = str(final_target)
            overrides[base_name] = register_override(updated_registry, base_name, final_target, args.version, stats)
        temporary.replace(destination)
        args.dataset_info.write_text(json.dumps(updated_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finalize_version(args.manifest, args.dataset_info, args.version, args.parent, args.description, overrides)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def filter_recommendation_cot_complete(source: Path, target: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    kept = rejected = 0
    with source.open("rb") as input_file, target.open("wb") as output_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            sample = json.loads(raw_line)
            output = sample.get("output")
            if not isinstance(output, str):
                raise TypeError(f"{source}:{line_number} output is not a string")
            if not all(section in output for section in REQUIRED_RECOMMENDATION_SECTIONS):
                rejected += 1
                continue
            encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            output_file.write(encoded)
            digest.update(encoded)
            kept += 1
    return {"file_name": str(target), "records": kept, "sha256": digest.hexdigest(), "rejected_missing_required_sections": rejected}


def create_recommendation_cot_complete(args: argparse.Namespace) -> None:
    ensure_version_name(args.version)
    manifest = load_manifest(args.manifest)
    registry = read_json(args.dataset_info)
    destination = args.version_root / args.version
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = Path(resolve_entry(manifest, args.parent, "onereason_recommendation_cot")["file_name"])
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        target = temporary / "onereason_recommendation_cot.jsonl"
        stats = filter_recommendation_cot_complete(source, target)
        final_target = destination / target.name
        stats["file_name"] = str(final_target)
        updated_registry = copy.deepcopy(registry)
        overrides = {
            "onereason_recommendation_cot": register_override(
                updated_registry, "onereason_recommendation_cot", final_target, args.version, stats
            )
        }
        temporary.replace(destination)
        args.dataset_info.write_text(json.dumps(updated_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finalize_version(args.manifest, args.dataset_info, args.version, args.parent, args.description, overrides)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_patches(raw_patches: list[str]) -> dict[str, Path]:
    patches: dict[str, Path] = {}
    for raw_patch in raw_patches:
        if "=" not in raw_patch:
            raise ValueError(f"Patch must have LOGICAL_DATASET=PATH form: {raw_patch!r}")
        base_name, raw_path = raw_patch.split("=", 1)
        if base_name not in LOGICAL_DATASETS:
            raise ValueError(f"Unknown logical dataset in patch: {base_name!r}")
        source = Path(raw_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if base_name in patches:
            raise ValueError(f"Duplicate patch for {base_name!r}")
        patches[base_name] = source
    if not patches:
        raise ValueError("At least one --patch is required")
    return patches


def register_version(args: argparse.Namespace) -> None:
    ensure_version_name(args.version)
    manifest = load_manifest(args.manifest)
    registry = read_json(args.dataset_info)
    patches = parse_patches(args.patch)
    destination = args.version_root / args.version
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    updated_registry = copy.deepcopy(registry)
    overrides: dict[str, dict[str, Any]] = {}
    try:
        for base_name, source in patches.items():
            target = temporary / f"{base_name}.jsonl"
            shutil.copyfile(source, target)
            stats = file_stats(target)
            final_target = destination / target.name
            stats["file_name"] = str(final_target)
            overrides[base_name] = register_override(updated_registry, base_name, final_target, args.version, stats)
        temporary.replace(destination)
        args.dataset_info.write_text(json.dumps(updated_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finalize_version(args.manifest, args.dataset_info, args.version, args.parent, args.description, overrides)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    registry = read_json(args.dataset_info)
    failures: list[str] = []
    verified_paths: dict[Path, dict[str, Any]] = {}
    for version in manifest.get("versions", {}):
        for base_name in LOGICAL_DATASETS:
            try:
                entry = resolve_entry(manifest, version, base_name)
                registry_name = entry["registry_name"]
                path = Path(entry["file_name"])
                if registry_name not in registry:
                    failures.append(f"{version}:{base_name} registry entry {registry_name} is missing")
                elif registry[registry_name].get("file_name") != str(path):
                    failures.append(f"{version}:{base_name} registry path differs from manifest")
                elif not path.is_file():
                    failures.append(f"{version}:{base_name} file is missing: {path}")
                elif args.verify_hashes:
                    stats = verified_paths.setdefault(path, file_stats(path))
                    if stats["records"] != entry.get("records") or stats["sha256"] != entry.get("sha256"):
                        failures.append(f"{version}:{base_name} count or sha256 differs from manifest")
            except (KeyError, ValueError) as error:
                failures.append(f"{version}:{base_name} unresolved: {error}")
    if failures:
        raise SystemExit("Dataset version audit failed:\n" + "\n".join(failures))
    mode = " with content hashes" if args.verify_hashes else ""
    print(f"Dataset version audit passed for {len(manifest.get('versions', {}))} version(s){mode}.")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--version-root", type=Path, default=DEFAULT_VERSION_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-raw-all")
    add_common_arguments(init)
    init.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    init.set_defaults(func=init_raw_all)

    thought = subparsers.add_parser("create-thought-prompts")
    add_common_arguments(thought)
    thought.add_argument("--version", required=True)
    thought.add_argument("--parent", required=True)
    thought.add_argument("--description", default="Append /think or /no_think to every full-training prompt.")
    thought.set_defaults(func=create_thought_prompts)

    recommendation = subparsers.add_parser("create-recommendation-cot-complete")
    add_common_arguments(recommendation)
    recommendation.add_argument("--version", required=True)
    recommendation.add_argument("--parent", required=True)
    recommendation.add_argument("--description", default="Keep recommendation CoT samples containing all three required sections (【兴趣归纳】, 【行为模式】, 【预测总结】).")
    recommendation.set_defaults(func=create_recommendation_cot_complete)

    register = subparsers.add_parser("register-version")
    add_common_arguments(register)
    register.add_argument("--version", required=True)
    register.add_argument("--parent", required=True)
    register.add_argument("--patch", action="append", default=[])
    register.add_argument("--description", required=True)
    register.set_defaults(func=register_version)

    audit_parser = subparsers.add_parser("audit")
    add_common_arguments(audit_parser)
    audit_parser.add_argument("--verify-hashes", action="store_true", help="Recompute JSONL count and SHA-256 once per file.")
    audit_parser.set_defaults(func=audit)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
