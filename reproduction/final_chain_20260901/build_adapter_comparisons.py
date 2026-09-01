#!/usr/bin/env python3
"""Build cached CPU-only same-step LoRA comparisons for the dashboard."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from compare_adapters import compare_adapter_files
from repro_quality import inspect_checkpoint, load_json


def checkpoint_path(selected: Path, kind: str, step: int) -> Path:
    name = f"checkpoint-{step}" if kind == "trainer" else f"prompt-step-{step:04d}"
    return selected.parent / name


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_available(root: Path, reference_path: Path, *, stage_filter: str | None = None) -> dict[str, int]:
    reference = load_json(reference_path)
    output_root = root / "evidence" / "adapter_comparisons"
    built = skipped = pending = 0
    for stage in reference["stages"]:
        if stage_filter and stage["id"] != stage_filter:
            continue
        kind = stage["checkpoint_kind"]
        reproduced_selected = root / stage["selected_checkpoint_relative"]
        historical_selected = Path(stage["historical_checkpoint_path"])
        for step in stage["expected_checkpoints"]:
            output = output_root / f"{stage['id']}-{int(step)}.json"
            if output.is_file():
                skipped += 1
                continue
            reproduced = checkpoint_path(reproduced_selected, kind, int(step))
            historical = checkpoint_path(historical_selected, kind, int(step))
            if inspect_checkpoint(reproduced, kind, int(step))["status"] != "complete" or not (historical / "adapter_model.safetensors").is_file():
                pending += 1
                continue
            result = compare_adapter_files(
                historical / "adapter_model.safetensors",
                reproduced / "adapter_model.safetensors",
            )
            result.update({
                "comparison_scope": "lora_adapter_only",
                "stage_id": stage["id"],
                "step": int(step),
                "historical_checkpoint": str(historical),
                "reproduced_checkpoint": str(reproduced),
            })
            atomic_write_json(output, result)
            built += 1
            print(json.dumps({"built": str(output), "status": result["status"]}, ensure_ascii=False), flush=True)
    return {"built": built, "skipped": skipped, "pending": pending}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("historical_reference.json"))
    parser.add_argument("--stage")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    while True:
        counts = build_available(args.root.resolve(), args.reference.resolve(), stage_filter=args.stage)
        print(json.dumps({"status": "PASS", **counts}), flush=True)
        if not args.watch or counts["pending"] == 0:
            break
        time.sleep(max(args.poll_seconds, 10))


if __name__ == "__main__":
    main()
