#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("/data/lf_data_splits")
TARGET_DIR = Path("/data/lf_data_versions/v1_thought_prompt")
VERSION = "v1_thought_prompt"
SUFFIX = "_train98"

def marker_for(filename: str) -> str:
    if "_cot_" in filename:
        return "/think"
    if "_nocot_" in filename:
        return "/no_think"
    raise ValueError(f"Cannot infer thought mode from {filename}")

def create_version() -> dict[str, dict[str, object]]:
    if TARGET_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing version directory: {TARGET_DIR}")
    temporary = TARGET_DIR.with_name(f"{TARGET_DIR.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    manifest: dict[str, dict[str, object]] = {}
    try:
        for source in sorted(SOURCE_DIR.glob(f"onereason_*{SUFFIX}.jsonl")):
            marker = marker_for(source.name)
            target = temporary / source.name
            source_hash, target_hash = hashlib.sha256(), hashlib.sha256()
            total = appended = already_present = 0
            with source.open("rb") as input_file, target.open("wb") as output_file:
                for raw_line in input_file:
                    source_hash.update(raw_line)
                    sample = json.loads(raw_line)
                    prompt = sample.get("input")
                    if not isinstance(prompt, str):
                        raise TypeError(f"{source}:{total + 1} input is not a string")
                    if prompt.rstrip().endswith(marker):
                        already_present += 1
                    else:
                        sample["input"] = prompt.rstrip() + "\\n" + marker
                        appended += 1
                    encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\\n").encode("utf-8")
                    output_file.write(encoded)
                    target_hash.update(encoded)
                    total += 1
            manifest[source.stem] = {
                "source_path": str(source),
                "source_sha256": source_hash.hexdigest(),
                "target_file": target.name,
                "target_sha256": target_hash.hexdigest(),
                "records": total,
                "marker": marker,
                "appended": appended,
                "already_present": already_present,
            }
        (temporary / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "transform": "Append /think to *_cot inputs and /no_think to *_nocot inputs when absent.",
                    "datasets": manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
        temporary.replace(TARGET_DIR)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest

def register() -> None:
    dataset_info = REPO / "data" / "dataset_info.json"
    registry = json.loads(dataset_info.read_text(encoding="utf-8"))
    raw_names = sorted(name for name in registry if name.startswith("onereason_") and name.endswith(SUFFIX))
    for raw_name in raw_names:
        versioned_name = raw_name[: -len(SUFFIX)] + f"_{VERSION}{SUFFIX}"
        if versioned_name in registry:
            raise FileExistsError(f"Dataset registry already contains {versioned_name}")
        entry = dict(registry[raw_name])
        entry["file_name"] = str(TARGET_DIR / f"{raw_name}.jsonl")
        registry[versioned_name] = entry
    dataset_info.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

if __name__ == "__main__":
    created = create_version()
    register()
    print(f"Registered {len(created)} datasets for {VERSION}.")
