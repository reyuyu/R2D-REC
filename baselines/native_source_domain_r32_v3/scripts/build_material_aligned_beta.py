#!/usr/bin/env python3
"""Build a BETA dataset that preserves the uploaded material objective exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


MATERIAL_SOURCES = {
    "material_sample",
    "sid_bucket_canonical_no_think",
    "sid_bucket_reverse",
}
EXPECTED_MATERIAL_COUNTS = {
    "material_sample": 100_000,
    "sid_bucket_canonical_no_think": 11_298,
    "sid_bucket_reverse": 29_586,
}
EXPECTED_DOMAIN_COUNTS = {
    "video": 30_092,
    "prod": 29_180,
    "ad": 22_768,
    "living": 17_960,
}
DOMAIN_TARGETS = {"video": 0.22, "prod": 0.14, "ad": 0.16, "living": 0.10}
DOMAIN_WEIGHTS = {
    "video": 1.1791795483099141,
    "prod": 0.7738397930531297,
    "ad": 1.133452723686895,
    "living": 0.8980530210503628,
}
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
OTHER_ROUTES = (
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


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def material_projection(row: dict[str, Any]) -> bytes:
    projection = {key: row[key] for key in ("system", "prompt", "response", "data_source")}
    return (json.dumps(projection, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def material_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("system", "prompt", "response", "data_source") if key not in row]
    if missing:
        raise ValueError(f"Material row is missing fields: {missing}")
    if row["data_source"] not in MATERIAL_SOURCES:
        raise ValueError(f"Unexpected material data_source: {row['data_source']!r}")
    # Alpaca with an explicit system column produces the same system/user/assistant
    # roles as the original ShareGPT row while retaining trainer-side metadata.
    return {
        "instruction": row["prompt"],
        "input": "",
        "output": row["response"],
        "history": [],
        "system": row["system"],
        "data_source": row["data_source"],
        "source_segment": row["data_source"],
        "aux_metadata_json": "",
    }


def other_row(row: dict[str, Any], source: str, segment: str) -> dict[str, Any]:
    metadata = ""
    if source == "recommend":
        missing = [field for field in REC_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Recommendation row is missing metadata: {missing}")
        metadata = json.dumps(
            {field: row[field] for field in REC_FIELDS},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return {
        "instruction": row["instruction"],
        "input": row.get("input", ""),
        "output": row["output"],
        "history": row.get("history", []),
        "system": row.get("system", ""),
        "data_source": source,
        "source_segment": segment,
        "aux_metadata_json": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-source", type=Path, required=True)
    parser.add_argument("--beta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_file = output_dir / "onereason_beta_material_aligned.jsonl"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    source_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    dataset_digest = hashlib.sha256()
    source_projection_digest = hashlib.sha256()
    output_projection_digest = hashlib.sha256()

    with output_file.open("x", encoding="utf-8") as writer:
        for row in rows(args.material_source):
            converted = material_row(row)
            projection = material_projection(row)
            source_projection_digest.update(projection)
            output_projection_digest.update(
                material_projection(
                    {
                        "system": converted["system"],
                        "prompt": converted["instruction"],
                        "response": converted["output"],
                        "data_source": converted["data_source"],
                    }
                )
            )
            source = converted["data_source"]
            source_counts[source] += 1
            segment_counts[converted["source_segment"]] += 1
            if source == "material_sample":
                text = converted["instruction"] + "\n" + converted["output"]
                match = DOMAIN_RE.search(text)
                if match is None:
                    raise ValueError("material_sample row has no material domain marker")
                domain_counts[match.group(1)] += 1
            payload = json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n"
            writer.write(payload)
            dataset_digest.update(payload.encode("utf-8"))

        for filename, source, segment in OTHER_ROUTES:
            for row in rows(args.beta_dir / filename):
                converted = other_row(row, source, segment)
                source_counts[source] += 1
                segment_counts[segment] += 1
                payload = json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n"
                writer.write(payload)
                dataset_digest.update(payload.encode("utf-8"))

    actual_material = {key: source_counts[key] for key in EXPECTED_MATERIAL_COUNTS}
    if actual_material != EXPECTED_MATERIAL_COUNTS:
        raise ValueError(f"Material source count mismatch: {actual_material}")
    if dict(domain_counts) != EXPECTED_DOMAIN_COUNTS:
        raise ValueError(f"Material domain count mismatch: {dict(domain_counts)}")
    if source_projection_digest.digest() != output_projection_digest.digest():
        raise AssertionError("Material system/prompt/response/data_source projection changed")

    dataset_info = {
        "onereason_beta_material_aligned": {
            "file_name": output_file.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
                "system": "system",
            },
        }
    }
    manifest = {
        "name": "beta_material_aligned_v1",
        "material_source": str(args.material_source.resolve()),
        "material_source_sha256": hashlib.sha256(args.material_source.read_bytes()).hexdigest(),
        "material_projection_sha256": source_projection_digest.hexdigest(),
        "material_projection_preserved": True,
        "records": sum(source_counts.values()),
        "sha256": dataset_digest.hexdigest(),
        "source_counts": dict(source_counts),
        "source_segment_counts": dict(segment_counts),
        "material_domain_counts": dict(domain_counts),
        "material_domain_targets": DOMAIN_TARGETS,
        "material_domain_weights": DOMAIN_WEIGHTS,
        "material_row_order": "identical to material source upload",
        "non_material_source": str(args.beta_dir.resolve()),
        "world_included": False,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# BETA material-aligned v1\n\n"
        "Material rows preserve the uploaded system, prompt, response, data_source, and row order. "
        "Non-material rows use the active BETA user and recommendation components.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
