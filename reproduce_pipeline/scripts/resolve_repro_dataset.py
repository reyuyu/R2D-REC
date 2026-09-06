#!/usr/bin/env python3
"""Resolve and verify a registered reproduction dataset without fixed data paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REGISTRY_FILE = "registry.json"
AUTO_ROOTS = (Path("/root/reproduce_datasets"), Path("/data/reproduce_datasets"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_root(explicit: str | None = None) -> Path:
    requested = explicit or os.environ.get("REPRO_DATA_ROOT")
    if requested:
        root = Path(requested).expanduser().resolve()
        if not (root / REGISTRY_FILE).is_file():
            raise RuntimeError(f"dataset registry not found under REPRO_DATA_ROOT: {root}")
        return root
    matches = [root.resolve() for root in AUTO_ROOTS if (root / REGISTRY_FILE).is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            "could not select one reproduce_datasets registry; set REPRO_DATA_ROOT explicitly"
        )
    return matches[0]


def load_registry(root: Path) -> dict[str, Any]:
    value = json.loads((root / REGISTRY_FILE).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("datasets"), dict):
        raise RuntimeError("unsupported reproduce dataset registry schema")
    return value


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def verify_manifest_directory(path: Path, manifest_path: Path, entry: dict[str, Any]) -> tuple[str, int, int]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"dataset contract manifest is missing: {manifest_path}")
    expected_manifest_sha = str(entry.get("sha256", ""))
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            "registered dataset manifest SHA256 mismatch: "
            f"expected={expected_manifest_sha} actual={actual_manifest_sha}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("dataset contract manifest has no files mapping")
    actual_names = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    }
    expected_names = set(files)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"registered raw directory file set mismatch: missing={missing} extra={extra}")
    rows = 0
    for relative, expected in sorted(files.items()):
        if not isinstance(expected, dict):
            raise RuntimeError(f"invalid manifest record: {relative}")
        candidate = path / relative
        if candidate.stat().st_size != int(expected.get("bytes", -1)):
            raise RuntimeError(f"registered raw file size mismatch: {relative}")
        if sha256(candidate) != str(expected.get("sha256", "")):
            raise RuntimeError(f"registered raw file SHA256 mismatch: {relative}")
        rows += int(expected.get("rows", -1))
    if len(files) != int(manifest.get("file_count", -1)):
        raise RuntimeError("dataset contract manifest file_count is inconsistent")
    if rows != int(manifest.get("total_rows", -1)):
        raise RuntimeError("dataset contract manifest total_rows is inconsistent")
    return actual_manifest_sha, len(files), rows


def resolve_dataset(root: Path, key: str, manifest_path: Path | None = None) -> dict[str, Any]:
    registry = load_registry(root)
    entry = registry["datasets"].get(key)
    if not isinstance(entry, dict):
        raise RuntimeError(f"dataset key is not registered: {key}")
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"registered dataset path must be relative and contained: {key}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"registered dataset escapes REPRO_DATA_ROOT: {key}") from error
    kind = str(entry.get("kind", "jsonl_file"))
    expected_sha = str(entry.get("sha256", ""))
    if kind == "jsonl_file":
        if not path.is_file():
            raise RuntimeError(f"registered dataset file is missing: {key}")
        actual_sha = sha256(path)
        actual_rows = count_rows(path)
        file_count = 1
    elif kind == "raw_parquet_directory":
        if not path.is_dir():
            raise RuntimeError(f"registered raw dataset directory is missing: {key}")
        if manifest_path is None:
            raise RuntimeError(f"raw directory dataset requires --manifest: {key}")
        actual_sha, file_count, actual_rows = verify_manifest_directory(path, manifest_path, entry)
        if int(entry.get("files", -1)) != file_count:
            raise RuntimeError(
                f"registered dataset file-count mismatch for {key}: "
                f"expected={entry.get('files')} actual={file_count}"
            )
    else:
        raise RuntimeError(f"unsupported registered dataset kind for {key}: {kind}")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"registered dataset SHA256 mismatch for {key}: expected={expected_sha} actual={actual_sha}"
        )
    expected_rows = int(entry.get("rows", -1))
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"registered dataset row mismatch for {key}: expected={expected_rows} actual={actual_rows}"
        )
    return {
        "key": key,
        "path": str(path),
        "sha256": actual_sha,
        "rows": actual_rows,
        "split": str(entry.get("split", "train")),
        "kind": kind,
        "files": file_count,
        "registry": str(root / REGISTRY_FILE),
        "repro_data_root": str(root),
        "registered_dataset_used_by_trainer": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--key", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    args = parser.parse_args()
    record = resolve_dataset(discover_root(args.root), args.key, args.manifest)
    if args.format == "lines":
        for name in ("key", "path", "sha256", "rows", "split", "repro_data_root", "kind", "files"):
            print(record[name])
    else:
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
