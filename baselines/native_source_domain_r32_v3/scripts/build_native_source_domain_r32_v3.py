"""Build the own-data adapter for the isolated Native Source-Domain R32 baseline."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/data/lf_data_versions/alltrain")
MATERIAL = ROOT / "rec_material_bucket_reverse_v1"
V1 = ROOT / "v1_thought_prompt_all"
V3 = ROOT / "v3_recommendation_multi_positive"
MATERIAL_BUCKET_ROWS_PER_ROUTE = 11_298
MATERIAL_REVERSE_ROWS_PER_ROUTE = 11_298
DOMAIN_TARGETS = {"video": 0.22, "prod": 0.14, "ad": 0.16, "living": 0.10}
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def with_source(row: dict, source: str, source_segment: str) -> dict:
    # The reference LLaMA-Factory JSON loader requires one homogeneous schema
    # across the complete file. Recommendation V3 contains diagnostic metadata
    # that is useful to the macro trainer but irrelevant to this native baseline.
    # Keep a homogeneous schema while retaining V3 multi-positive information
    # in a single extensible JSON field for future auxiliary losses.
    recommendation_metadata = ""
    if source == "recommend":
        recommendation_metadata = json.dumps(
            {
                "version": "recommendation_v3_multi_positive",
                "group_id": row.get("recommendation_group_id"),
                "group_size": row.get("recommendation_group_size"),
                "all_gold_sids": row.get("recommendation_all_gold_sids", []),
                "current_gold_sid": row.get("recommendation_current_gold_sid"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return {
        "instruction": row["instruction"],
        "input": row.get("input", ""),
        "output": row["output"],
        "history": row.get("history", []),
        "data_source": source,
        "source_segment": source_segment,
        "aux_metadata_json": recommendation_metadata,
    }


def material_rows(path: Path, route: str):
    count = 0
    for index, row in enumerate(rows(path)):
        count += 1
        if index < MATERIAL_BUCKET_ROWS_PER_ROUTE:
            # The reference baseline weights only the no-think canonical route
            # as a full-response bucket. Keep the own-data CoT counterpart as
            # a domain-weighted material sample instead of silently dropping it.
            source = "sid_bucket_canonical_no_think" if route == "material_nocot" else "material_sample"
            yield with_source(row, source, f"{route}:canonical")
        elif index < MATERIAL_BUCKET_ROWS_PER_ROUTE + MATERIAL_REVERSE_ROWS_PER_ROUTE:
            yield with_source(row, "sid_bucket_reverse", f"{route}:reverse")
        else:
            yield with_source(row, "material_sample", f"{route}:remainder")
    expected = MATERIAL_BUCKET_ROWS_PER_ROUTE + MATERIAL_REVERSE_ROWS_PER_ROUTE + 50_000
    if count != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {count}.")


def source_paths():
    return {
        "material_cot": MATERIAL / "onereason_material_cot.jsonl",
        "material_nocot": MATERIAL / "onereason_material_nocot.jsonl",
        "user_action": V1 / "onereason_user_action_nocot.jsonl",
        "user_chain_cot": V1 / "onereason_user_chain_cot.jsonl",
        "user_chain_nocot": V1 / "onereason_user_chain_nocot.jsonl",
        "recommendation_cot": V3 / "onereason_recommendation_cot_v3_multi_positive.jsonl",
        "recommendation_nocot": V3 / "onereason_recommendation_nocot_v3_multi_positive.jsonl",
    }


def classify_domain(row: dict) -> str:
    match = DOMAIN_RE.search(f"{row.get('instruction', '')}\n{row.get('input', '')}\n{row.get('output', '')}")
    if match is None:
        raise ValueError("material_sample row has no video/prod/ad/living domain token")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_file = output_dir / "onereason_native_source_domain_r32_v3.jsonl"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {output_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_count = Counter()
    domain_count = Counter()
    digest = hashlib.sha256()
    records = 0
    paths = source_paths()
    ordered_groups = (
        ("material_cot", "material_nocot"),
        ("user_action", "user_chain_cot", "user_chain_nocot"),
        ("recommendation_cot", "recommendation_nocot"),
    )

    with output_file.open("x", encoding="utf-8") as stream:
        # First source buckets are deliberately contiguous, matching the reference packing contract.
        for route in ("material_cot", "material_nocot"):
            for row in material_rows(paths[route], route):
                if row["data_source"] != "sid_bucket_canonical_no_think":
                    continue
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                source_count[row["data_source"]] += 1
                records += 1
        for route in ("material_cot", "material_nocot"):
            for row in material_rows(paths[route], route):
                if row["data_source"] != "sid_bucket_reverse":
                    continue
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                source_count[row["data_source"]] += 1
                records += 1
        for route in ("material_cot", "material_nocot"):
            for row in material_rows(paths[route], route):
                if row["data_source"] != "material_sample":
                    continue
                domain_count[classify_domain(row)] += 1
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                source_count[row["data_source"]] += 1
                records += 1
        for route in ("user_action", "user_chain_cot", "user_chain_nocot"):
            for row in rows(paths[route]):
                item = with_source(row, "understand_user", route)
                line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                source_count[item["data_source"]] += 1
                records += 1
        for route in ("recommendation_cot", "recommendation_nocot"):
            for row in rows(paths[route]):
                item = with_source(row, "recommend", route)
                line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                source_count[item["data_source"]] += 1
                records += 1

    if set(domain_count) != set(DOMAIN_TARGETS):
        raise ValueError(f"Material domain coverage mismatch: {dict(domain_count)}")
    total_material = sum(domain_count.values())
    target_sum = sum(DOMAIN_TARGETS.values())
    domain_weights = {
        domain: (DOMAIN_TARGETS[domain] / domain_count[domain]) * total_material / target_sum
        for domain in DOMAIN_TARGETS
    }
    dataset_info = {
        "onereason_native_source_domain_r32_v3": {
            "file_name": output_file.name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
        }
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": "native_source_domain_r32_v3",
        "parent_dataset_version": "rec_material_bucket_reverse_v1",
        "world_included": False,
        "records": records,
        "sha256": digest.hexdigest(),
        "source_counts": dict(source_count),
        "material_domain_counts": dict(domain_count),
        "material_domain_targets": DOMAIN_TARGETS,
        "material_domain_weights": domain_weights,
        "source_order": ["sid_bucket_canonical_no_think", "sid_bucket_reverse", "material_sample", "understand_user", "recommend"],
        "schema": [
            "instruction",
            "input",
            "output",
            "history",
            "data_source",
            "source_segment",
            "aux_metadata_json",
        ],
        "input_files": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
