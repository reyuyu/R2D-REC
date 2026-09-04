#!/usr/bin/env python3
"""Verify the portable data package without loading full datasets into memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else root / "data"
    manifest = json.loads((root / "PARQUET_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    rows = 0
    for relative, expected in manifest["files"].items():
        path = data_root / relative
        metadata = pq.read_metadata(path)
        rows += metadata.num_rows
        if metadata.num_rows != expected["rows"]:
            failures.append(f"row mismatch: {relative}")
        if path.stat().st_size != expected["parquet_bytes"]:
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != expected["parquet_sha256"]:
            failures.append(f"sha256 mismatch: {relative}")
    package_manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for relative in (
        "recommendation/hcr_group_metadata.json",
        "recommendation/rec_group_catalog.json",
    ):
        candidate = data_root / relative
        expected = package_manifest[f"data/{relative}"]
        if not candidate.is_file():
            failures.append(f"missing training metadata: {relative}")
        elif candidate.stat().st_size != expected["bytes"] or sha256(candidate) != expected["sha256"]:
            failures.append(f"training metadata mismatch: {relative}")
    result = {"status": "valid" if not failures else "invalid", "parquet_files": len(manifest["files"]), "rows": rows, "failures": failures}
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
