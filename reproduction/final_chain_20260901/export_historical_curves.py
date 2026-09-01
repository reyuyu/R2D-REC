#!/usr/bin/env python3
"""Export compact, immutable historical training curves for the dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro_quality import load_json, load_metric_rows_from_path, sampled_curve, sha256_file


def export_curves(reference_path: Path) -> dict[str, object]:
    reference = load_json(reference_path)
    stages: dict[str, object] = {}
    for stage in reference["stages"]:
        source = Path(stage["historical_metrics_path"])
        rows = load_metric_rows_from_path(source.resolve(strict=True))
        stages[stage["id"]] = {
            "source_sha256": sha256_file(source),
            "raw_row_count": len(rows),
            "curve": sampled_curve(rows, stage),
        }
    return {"schema_version": 1, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("historical_reference.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("historical_curves.json"))
    args = parser.parse_args()
    payload = export_curves(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "stages": list(payload["stages"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
