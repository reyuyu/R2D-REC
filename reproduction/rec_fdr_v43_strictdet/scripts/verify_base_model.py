#!/usr/bin/env python3
"""Verify that a local base model is byte-identical to the reference model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model_root", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))["files"]
    failures = []
    for relative, record in expected.items():
        path = args.model_root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {relative}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"sha256 mismatch: {relative}")
    print(json.dumps({"status": "valid" if not failures else "invalid", "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
