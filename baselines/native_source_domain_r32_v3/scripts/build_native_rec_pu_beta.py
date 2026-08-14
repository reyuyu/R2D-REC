#!/usr/bin/env python3
"""Create a model-schema adapter for BETA without modifying BETA itself."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROUTES = (
    ("onereason_material_cot.jsonl", "material_sample", "material_cot"),
    ("onereason_material_nocot.jsonl", "material_sample", "material_nocot"),
    ("onereason_user_action_nocot.jsonl", "understand_user", "user_action"),
    ("onereason_user_chain_cot.jsonl", "understand_user", "user_chain_cot"),
    ("onereason_user_chain_nocot.jsonl", "understand_user", "user_chain_nocot"),
    ("onereason_recommendation_cot.jsonl", "recommend", "recommendation_cot"),
    ("onereason_recommendation_nocot.jsonl", "recommend", "recommendation_nocot"),
)
REC_FIELDS = (
    "recommendation_group_id",
    "recommendation_group_size",
    "recommendation_all_gold_sids",
    "recommendation_current_gold_sid",
)


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error


def adapt(row: dict, source: str, source_segment: str) -> dict:
    metadata = ""
    if source == "recommend":
        missing = [field for field in REC_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Recommendation row is missing BETA metadata: {missing}")
        metadata = json.dumps({field: row[field] for field in REC_FIELDS}, ensure_ascii=False, separators=(",", ":"))
    return {
        "instruction": row["instruction"],
        "input": row.get("input", ""),
        "output": row["output"],
        "history": row.get("history", []),
        "data_source": source,
        "source_segment": source_segment,
        "aux_metadata_json": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_file = output_dir / "onereason_native_rec_pu_beta.jsonl"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {output_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    digest = hashlib.sha256()
    with output_file.open("x", encoding="utf-8") as stream:
        for filename, source, segment in ROUTES:
            for row in rows(args.beta_dir / filename):
                encoded = json.dumps(adapt(row, source, segment), ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(encoded)
                digest.update(encoded.encode("utf-8"))
                counts[segment] += 1
    info = {
        "onereason_native_rec_pu_beta": {
            "file_name": output_file.name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
        }
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"source": str(args.beta_dir), "rows": dict(counts), "total": sum(counts.values()), "sha256": digest.hexdigest()}
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
