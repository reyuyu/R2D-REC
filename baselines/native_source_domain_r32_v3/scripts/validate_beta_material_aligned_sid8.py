#!/usr/bin/env python3
"""Hard preflight for the immutable BETA material three-route experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import yaml


EXPECTED_MANIFEST = Path("/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json")
EXPECTED_DATASET = "onereason_beta_material_aligned"
EXPECTED_COUNTS = {
    "material_sample": 100_000,
    "sid_bucket_canonical_no_think": 11_298,
    "sid_bucket_reverse": 29_586,
}
EXPECTED_DOMAINS = {"ad": 22_768, "prod": 29_180, "living": 17_960, "video": 30_092}
EXPECTED_DOMAIN_WEIGHTS = {
    "video": 1.1791795483099141,
    "prod": 0.7738397930531297,
    "ad": 1.133452723686895,
    "living": 0.8980530210503628,
}
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
MATERIAL_SOURCES = tuple(EXPECTED_COUNTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection(row: dict) -> bytes:
    payload = {key: row[key] for key in ("system", "prompt", "response", "data_source")}
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def audit_dataset(manifest: dict, dataset: Path) -> None:
    if manifest.get("material_projection_preserved") is not True:
        raise AssertionError("material_projection_preserved is not true")
    if manifest.get("source_counts") is None:
        raise AssertionError("manifest has no source_counts")
    material_source = Path(manifest["material_source"])
    if sha256(material_source) != manifest["material_source_sha256"]:
        raise AssertionError("material source SHA256 differs from manifest")

    source_digest = hashlib.sha256()
    with material_source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_digest.update(projection(row))
    if source_digest.hexdigest() != manifest["material_projection_sha256"]:
        raise AssertionError("material source projection digest differs from manifest")

    counts: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    output_projection = hashlib.sha256()
    material_index = 0
    records = 0
    with material_source.open(encoding="utf-8") as source_handle, dataset.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            records += 1
            source = row.get("data_source")
            counts[source] += 1
            if source in EXPECTED_COUNTS:
                expected_line = next(source_handle, None)
                if expected_line is None:
                    raise AssertionError("combined dataset has too many material rows")
                converted = {
                    "system": row["system"],
                    "prompt": row["instruction"],
                    "response": row["output"],
                    "data_source": source,
                }
                if projection(converted) != projection(json.loads(expected_line)):
                    raise AssertionError(f"material projection changed at combined row {line_number}")
                output_projection.update(projection(converted))
                material_index += 1
                if source == "material_sample":
                    match = DOMAIN_RE.search(row["instruction"] + "\n" + row["output"])
                    if match is None:
                        raise AssertionError(f"material_sample has no domain at row {line_number}")
                    domains[match.group(1)] += 1
        if next(source_handle, None) is not None:
            raise AssertionError("combined dataset has too few material rows")

    if records != manifest["records"]:
        raise AssertionError(f"record count={records}, expected={manifest['records']}")
    if sha256(dataset) != manifest["sha256"]:
        raise AssertionError("combined dataset SHA256 differs from manifest")
    material_counts = {source: counts[source] for source in EXPECTED_COUNTS}
    if material_counts != EXPECTED_COUNTS or material_counts != {
        source: manifest["source_counts"].get(source) for source in EXPECTED_COUNTS
    }:
        raise AssertionError(f"material route counts mismatch: {material_counts}")
    if dict(domains) != EXPECTED_DOMAINS or dict(domains) != manifest["material_domain_counts"]:
        raise AssertionError(f"material domain counts mismatch: {dict(domains)}")
    if material_index != sum(EXPECTED_COUNTS.values()):
        raise AssertionError(f"material projection row count={material_index}")
    if output_projection.hexdigest() != manifest["material_projection_sha256"]:
        raise AssertionError("combined material projection digest differs from manifest")


def audit_registry(dataset: Path) -> None:
    registry_path = Path("/app/LLaMA-Factory/data/dataset_info.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get(EXPECTED_DATASET)
    if entry is None:
        raise AssertionError(f"dataset registry lacks {EXPECTED_DATASET}")
    if Path(entry["file_name"]).resolve() != dataset.resolve():
        raise AssertionError("dataset registry path does not match formal manifest dataset")


def audit_runtime_loss_route(launcher: Path, config: Path, manifest: Path) -> None:
    original_argv = sys.argv[:]
    original_manifest = os.environ.get("MATERIAL_DOMAIN_MANIFEST")
    original_weight = os.environ.get("GLOBAL_ITEM_WEIGHT")
    try:
        sys.argv = [str(launcher), str(config)]
        os.environ["MATERIAL_DOMAIN_MANIFEST"] = str(manifest)
        os.environ["GLOBAL_ITEM_WEIGHT"] = "8"
        spec = importlib.util.spec_from_file_location("material_aligned_training_route", launcher)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot load native training launcher")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        labels = [-100, 101, 202]
        item_ids = {101}
        expected = {
            "material_sample": [0.0, 8.0, 1.0],
            "sid_bucket_canonical_no_think": [0.0, 4.0, 4.0],
            "sid_bucket_reverse": [0.0, 8.0, 1.0],
        }
        actual = {source: module._build_loss_weights(labels, source, item_ids) for source in expected}
        if actual != expected:
            raise AssertionError(f"actual material loss routes mismatch: {actual}")

        for domain, weight in EXPECTED_DOMAIN_WEIGHTS.items():
            prompt = [{"content": f"<|{domain}_begin|>"}]
            if module._material_domain_weight("material_sample", prompt, []) != weight:
                raise AssertionError(f"domain weight mismatch for {domain}")
        for source in ("sid_bucket_canonical_no_think", "sid_bucket_reverse"):
            if module._material_domain_weight(source, [{"content": "<|prod_begin|>"}], []) != 1.0:
                raise AssertionError(f"{source} must not receive a domain multiplier")
    finally:
        sys.argv = original_argv
        if original_manifest is None:
            os.environ.pop("MATERIAL_DOMAIN_MANIFEST", None)
        else:
            os.environ["MATERIAL_DOMAIN_MANIFEST"] = original_manifest
        if original_weight is None:
            os.environ.pop("GLOBAL_ITEM_WEIGHT", None)
        else:
            os.environ["GLOBAL_ITEM_WEIGHT"] = original_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EXPECTED_MANIFEST)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = Path(args.manifest.parent / "onereason_beta_material_aligned.jsonl")
    if config.get("dataset") != EXPECTED_DATASET or Path(config.get("dataset_dir", "")).resolve() != args.manifest.parent.resolve():
        raise AssertionError("config does not select the formal material-aligned registry and directory")
    if manifest.get("name") != "beta_material_aligned_v1" or manifest.get("world_included") is not False:
        raise AssertionError("formal manifest identity/world contract mismatch")
    if manifest.get("material_domain_weights") != EXPECTED_DOMAIN_WEIGHTS:
        raise AssertionError("manifest domain weights differ from locked contract")

    audit_dataset(manifest, dataset)
    audit_registry(dataset)
    audit_runtime_loss_route(args.launcher, args.config, args.manifest)
    print(
        "PASS material_counts=" + json.dumps(EXPECTED_COUNTS, sort_keys=True)
        + " domains=" + json.dumps(EXPECTED_DOMAINS, sort_keys=True)
        + " routes=material(1,SID8,domain),canonical(response4),reverse(1,SID8)"
    )


if __name__ == "__main__":
    main()
