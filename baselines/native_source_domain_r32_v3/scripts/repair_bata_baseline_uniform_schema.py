#!/usr/bin/env python3
"""Remove the non-training archive_source column and atomically refresh the manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_COLUMNS = {
    "instruction", "input", "output", "history", "system",
    "data_source", "source_segment", "aux_metadata_json",
}
EXPECTED_RECORDS = 222_001
EXPECTED_REMOVALS = 189_153


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    dataset = dataset_dir / "onereason_bata_baseline.jsonl"
    temporary = dataset.with_suffix(".jsonl.schema-repair.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary file: {temporary}")

    digest = hashlib.sha256()
    records = 0
    removals = 0
    with dataset.open(encoding="utf-8") as source, temporary.open("x", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if "archive_source" in row:
                del row["archive_source"]
                removals += 1
            if set(row) != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected columns at row {line_number}: {sorted(row)}")
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            output.write(encoded)
            digest.update(encoded.encode("utf-8"))
            records += 1
        output.flush()
        os.fsync(output.fileno())

    if records != EXPECTED_RECORDS or removals != EXPECTED_REMOVALS:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Schema repair mismatch: records={records}, removals={removals}")
    os.replace(temporary, dataset)

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = digest.hexdigest()
    manifest["row_columns"] = sorted(EXPECTED_COLUMNS)
    manifest["archive_source_provenance"] = "preserved in archive_source_counts manifest field; not a training row column"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"records": records, "removed_archive_source": removals, "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
