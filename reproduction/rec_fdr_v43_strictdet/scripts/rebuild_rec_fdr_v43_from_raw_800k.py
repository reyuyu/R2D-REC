#!/usr/bin/env python3
"""Rebuild the V4.3 S1 training data from the exact raw competition Parquet."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif path.stat().st_size != expected["bytes"]:
            failures.append(f"size mismatch: {relative}")
        elif pq.read_metadata(path).num_rows != expected["rows"]:
            failures.append(f"row mismatch: {relative}")
        elif sha256(path) != expected["sha256"]:
            failures.append(f"sha256 mismatch: {relative}")
    extras = sorted(str(path.relative_to(root)) for path in root.rglob("*.parquet") if str(path.relative_to(root)) not in manifest["files"])
    failures.extend(f"unexpected: {value}" for value in extras)
    if failures:
        raise RuntimeError("raw input verification failed:\n" + "\n".join(failures[:50]))
    print(json.dumps({"event": "raw_input_verified", "files": len(manifest["files"]), "rows": manifest["total_rows"]}))


def normalize(raw_root: Path, processed: Path, module: object) -> None:
    processed.mkdir(parents=True)
    sources = sorted(raw_root.rglob("*.parquet"))
    total = 0
    for index, source in enumerate(sources, 1):
        destination = processed / module.output_name(raw_root, source)
        total += module.convert_file(raw_root, source, destination)
        if index % 100 == 0 or index == len(sources):
            print(json.dumps({"event": "normalized", "completed": index, "total": len(sources)}), flush=True)
    if total != 792797:
        raise RuntimeError(f"normalized row count changed: {total}")


def classify(processed: Path, classified: Path, module: object) -> None:
    classified.mkdir(parents=True)
    groups = module.GROUPS
    temp_paths = []
    with ExitStack() as stack:
        handles = {}
        for group in groups:
            final = classified / f"{group}.jsonl"
            temp = final.with_suffix(".jsonl.tmp")
            temp_paths.append((temp, final))
            handles[group] = stack.enter_context(temp.open("w", encoding="utf-8"))
        counts = {group: 0 for group in groups}
        for source in sorted(processed.glob("*.parquet")):
            table = pq.read_table(source, columns=["messages", "source_dataset"])
            for row in table.to_pylist():
                system, prompt, response = module.split_dialog(row["messages"])
                for group, mode in module.classify(row["source_dataset"], prompt, response):
                    record = {
                        "system": system,
                        "prompt": module.with_control(prompt, mode),
                        "response": module.think_response(response) if mode == "think" else module.no_think_response(response),
                    }
                    handles[group].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[group] += 1
    for temp, final in temp_paths:
        os.replace(temp, final)
    expected = {
        "material_think_sid_to_semantic": 178364,
        "material_think_semantic_to_sid": 177466,
        "material_no_think_sid_to_semantic": 177281,
        "material_no_think_semantic_to_sid": 178569,
        "user_think": 8339,
        "user_no_think": 24509,
        "recommendation_think": 48269,
        "recommendation_no_think": 48269,
    }
    if counts != expected:
        raise RuntimeError(f"classified counts changed: {counts}")
    print(json.dumps({"event": "classified", "counts": counts}, ensure_ascii=False))


def parquet(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = pajson.open_json(source, read_options=pajson.ReadOptions(block_size=64 * 1024 * 1024))
    writer = None
    rows = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(target, batch.schema, compression="zstd", compression_level=9, use_dictionary=True)
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer:
            writer.close()
    return rows


def make_training_data(package_root: Path, v3: Path, v43: Path, output: Path) -> None:
    expected = json.loads((package_root / "PARQUET_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    base_names = sorted(Path(name).stem for name in expected if name.startswith("base/"))
    rec_names = sorted(Path(name).stem for name in expected if name.startswith("recommendation/"))
    generated = [(v3 / f"{name}.jsonl", output / "base" / f"{name}.parquet", f"base/{name}.parquet") for name in base_names]
    generated += [(v43 / f"{name}.jsonl", output / "recommendation" / f"{name}.parquet", f"recommendation/{name}.parquet") for name in rec_names]
    failures = []
    for source, target, key in generated:
        item = expected[key]
        source_hash = sha256(source)
        if source.stat().st_size != item["source_jsonl_bytes"] or source_hash != item["source_jsonl_sha256"]:
            failures.append(f"generated JSONL mismatch: {source.name}")
            continue
        rows = parquet(source, target)
        if rows != item["rows"]:
            failures.append(f"generated row mismatch: {source.name}")
    if failures:
        raise RuntimeError("final dataset differs from reference:\n" + "\n".join(failures))
    dataset_order = (
        "material_no_think_semantic_to_sid_train", "material_no_think_semantic_to_sid_val",
        "material_no_think_sid_to_semantic_train", "material_no_think_sid_to_semantic_val",
        "material_think_semantic_to_sid_train", "material_think_semantic_to_sid_val",
        "material_think_sid_to_semantic_train", "material_think_sid_to_semantic_val",
        "user_clean_train", "user_clean_val", "rec_r1_train", "rec_full_val", "rec_r1_val",
        "rec_r2_val", "rec_full_k1_train", "rec_full_k2_train", "rec_full_k3_train",
        "rec_r2_k1_train", "rec_r2_k2_train", "rec_r2_k3_train",
        "rec_r2_k1_pack_seed_0_train", "rec_r2_k1_pack_seed_1_train",
        "rec_r2_k1_pack_seed_2_train", "rec_r2_k1_pack_seed_3_train",
    )
    by_stem = {Path(name).stem: name for name in expected}
    if set(dataset_order) != set(by_stem):
        raise RuntimeError("dataset_info contract no longer covers the generated Parquet set")
    dataset_info = {
        name: {
            "file_name": by_stem[name],
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        for name in dataset_order
    }
    dataset_info_path = output / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("hcr_group_metadata.json", "rec_group_catalog.json"):
        shutil.copy2(v43 / name, output / "recommendation" / name)
    package_files = json.loads(
        (package_root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )["files"]
    expected_dataset_info = package_files["data/dataset_info.json"]
    if (
        dataset_info_path.stat().st_size != expected_dataset_info["bytes"]
        or sha256(dataset_info_path) != expected_dataset_info["sha256"]
    ):
        raise RuntimeError("generated dataset_info.json differs from reference")
    print(json.dumps({"event": "training_data_rebuilt", "output": str(output), "files": len(generated)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent.parent
    scripts = package_root / "scripts/data_generation"
    sys.path.insert(0, str(scripts))
    raw_root = args.raw_root.resolve()
    base_model = args.base_model.resolve()
    work = args.work_root.resolve()
    if work.exists():
        raise FileExistsError(f"refusing to overwrite work root: {work}")
    work.mkdir(parents=True)
    verify_raw(raw_root, package_root / "RAW_800K_MANIFEST.json")

    prep = importlib.import_module("prepare_dataset")
    exporter = importlib.import_module("export_classified_jsonl")
    processed = work / "processed"
    classified = work / "classified"
    normalize(raw_root, processed, prep)
    classify(processed, classified, exporter)

    train_root = work / "train"
    (train_root / "data").mkdir(parents=True)
    (train_root / "reports").mkdir(parents=True)
    model_alias = work / "OneReason-8B-pretrain-competition"
    model_alias.symlink_to(base_model, target_is_directory=True)

    balanced = importlib.import_module("build_balanced_lora_dataset")
    budget = importlib.import_module("build_material_budget_datasets")
    balanced.SOURCE = classified
    budget.TRAIN_ROOT = train_root
    budget.balanced.SOURCE = classified
    budget.build(140000, False)

    expa = importlib.import_module("build_material140k_expA_strict_three_task_dataset")
    expa.TRAIN_ROOT = train_root
    expa.SOURCE = train_root / "data/material140k_userrec_full_v1"
    expa.OUTPUT = train_root / "data/material140k_expA_strict_three_task_v1"
    expa.REPORT_DIR = train_root / "reports/material140k_expA_strict_three_task_v1"
    expa.main()

    v3 = importlib.import_module("build_material140k_userclean_rec_curriculum_v3_dataset")
    v3.ROOT = train_root
    v3.EXP_A = expa.OUTPUT
    v3.RAW = expa.SOURCE / "recommendation_think.jsonl"
    v3.OUT = train_root / "data/material140k_userclean_rec_curriculum_fullft_v3"
    v3.REPORT_DIR = train_root / "reports/material140k_userclean_rec_curriculum_fullft_v3"
    v3.main()

    candidate = importlib.import_module("build_rec_candidate_curriculum_v2_dataset")
    candidate.ROOT = train_root
    candidate.SOURCE = v3.OUT
    candidate.RAW_FULL = v3.RAW
    candidate.OUT = train_root / "data/rec_candidate_curriculum_v2"
    candidate.REPORT = train_root / "reports/rec_candidate_curriculum_v2"
    candidate.main()

    v4 = importlib.import_module("build_rec_fdr_v4_dataset")
    v4.ROOT = train_root
    v4.SOURCE = candidate.OUT
    v4.OUT = train_root / "data/rec_fdr_curriculum_v4"
    v4.REPORT = train_root / "reports/rec_fdr_curriculum_v4"
    v4.main()

    v42 = importlib.import_module("build_rec_fdr_v42_datasets")
    v42.ROOT = train_root
    v42.SOURCE = v4.OUT
    v42.OUT_A = train_root / "data/rec_fdr_v42_interestonly"
    v42.build_a()

    v43 = importlib.import_module("build_rec_fdr_v43_hcr_dataset")
    v43_out = train_root / "data/rec_fdr_v43_hcr_frozen_v42a"
    v43.build(v42.OUT_A, v43_out, None, "preserve")
    generated_hcr = v43_out / "hcr_group_metadata.json"
    hcr = json.loads(generated_hcr.read_text(encoding="utf-8"))
    hcr["source_dataset"] = "/data/LLm-8B/code/train/data/rec_fdr_v42_interestonly"
    generated_hcr.write_text(
        json.dumps(hcr, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    package_files = json.loads(
        (package_root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )["files"]
    for name in ("hcr_group_metadata.json", "rec_group_catalog.json"):
        generated = v43_out / name
        expected = package_files[f"data/recommendation/{name}"]
        if generated.stat().st_size != expected["bytes"] or sha256(generated) != expected["sha256"]:
            raise RuntimeError(f"generated training metadata differs from reference: {name}")
    make_training_data(package_root, v3.OUT, v43_out, work / "rebuilt_data")


if __name__ == "__main__":
    main()
