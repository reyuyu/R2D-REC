#!/usr/bin/env python3
"""Create the V4 FDR dataset view without mutating the completed V2 dataset.

Text rows are hard-linked from V2 (therefore byte-identical); V4 adds a
split-aware SID catalog and an R2-K1 feasibility report.  Training code must
use only ``train_domain_inventory`` for FDR negative construction.
"""
from __future__ import annotations

import json
import os
import re
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/LLm-8B/code/train")
SOURCE = ROOT / "data/rec_candidate_curriculum_v2"
OUT = ROOT / "data/rec_fdr_curriculum_v4"
REPORT = ROOT / "reports/rec_fdr_curriculum_v4"
FILES = (
    "rec_full_k1_train.jsonl", "rec_full_k2_train.jsonl", "rec_full_k3_train.jsonl",
    "rec_r1_train.jsonl", "rec_r2_k1_train.jsonl", "rec_r2_k2_train.jsonl",
    "rec_r2_k3_train.jsonl", "rec_full_val.jsonl", "rec_r1_val.jsonl",
    "rec_r2_val.jsonl",
)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        yield from (json.loads(line) for line in handle if line.strip())


def sid_tuple(sid: str) -> tuple[int, int, int]:
    start = sid.index("<s_a_") + 5
    a_end = sid.index(">", start)
    start_b = sid.index("<s_b_") + 5
    b_end = sid.index(">", start_b)
    start_c = sid.index("<s_c_") + 5
    c_end = sid.index(">", start_c)
    return int(sid[start:a_end]), int(sid[start_b:b_end]), int(sid[start_c:c_end])


def collect(split: str) -> tuple[dict[str, dict], dict[str, list[str]]]:
    names = ("rec_full_k1_train.jsonl", "rec_full_k2_train.jsonl", "rec_full_k3_train.jsonl",
             "rec_r2_k1_train.jsonl", "rec_r2_k2_train.jsonl", "rec_r2_k3_train.jsonl") if split == "train" else ("rec_full_val.jsonl", "rec_r2_val.jsonl")
    groups: dict[str, dict] = {}
    domains: defaultdict[str, set[str]] = defaultdict(set)
    for name in names:
        for row in rows(SOURCE / name):
            # A prompt group may contain future SIDs from several domains. FDR
            # negatives are same-domain, so the stable training identity is the
            # prompt/COT group *plus the row's target domain; its positive set
            # remains the complete group future set for false-negative safety.
            group = str(row["rec_group_key"]) + "|" + str(row["target_domain"])
            positives = sorted(row["positive_future_sids"])
            domain = str(row["target_domain"])
            old = groups.setdefault(group, {"group_key": group, "source_group_key": row["rec_group_key"], "prompt_group_id": str(row["prompt_group_id"]), "positive_future_sids": positives, "domain": domain, "k": len(positives), "entropy_bucket": row["future_entropy_bucket"]})
            if old["positive_future_sids"] != positives or old["domain"] != domain or old["prompt_group_id"] != str(row["prompt_group_id"]):
                raise ValueError(f"inconsistent group metadata: {group}")
            domains[domain].update(positives)
    return groups, {domain: sorted(values) for domain, values in domains.items()}


def r2_audit() -> dict[str, dict[str, float | int]]:
    report = {}
    for bucket in ("k1", "k2", "k3"):
        data = list(rows(SOURCE / f"rec_r2_{bucket}_train.jsonl"))
        token_count = sum(len(row["prompt"]) + len(row["response"]) for row in data)
        groups = {row["prompt_group_id"] for row in data}
        sids = {row["final_target_sid"] for row in data}
        report[bucket] = {"raw_rows": len(data), "unique_prompt_groups": len(groups), "unique_positive_sids": len(sids), "raw_character_count": token_count}
    return report


def main() -> None:
    temporary = OUT.with_name(OUT.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    train_groups, train_inventory = collect("train")
    val_groups, val_inventory = collect("validation")
    overlap = set(train_groups) & set(val_groups)
    if overlap:
        raise ValueError(f"train/validation recommendation group overlap: {len(overlap)}")
    keys = sorted(set(train_groups) | set(val_groups))
    index = {key: value + 1 for value, key in enumerate(keys)}
    for group in list(train_groups.values()) + list(val_groups.values()):
        group["group_index"] = index[group["group_key"]]
    # Preserve text byte-for-byte while assigning one unique index to each
    # full/R2 group. This replaces the colliding V2 local group-index field.
    for name in FILES:
        source = SOURCE / name
        target = temporary / name
        if not name.endswith(".jsonl") or not name.startswith(("rec_full", "rec_r2")):
            os.link(source, target)
            continue
        with source.open(encoding="utf-8") as input_handle, target.open("w", encoding="utf-8") as output_handle:
            for line in input_handle:
                row = json.loads(line)
                replacement = str(index[str(row["rec_group_key"]) + "|" + str(row["target_domain"])])
                rewritten, count = re.subn(r'("rec_group_index"\s*:\s*)\d+', r'\g<1>' + replacement, line, count=1)
                if count != 1:
                    raise ValueError(f"rec_group_index missing from {name}")
                output_handle.write(rewritten)
    # Framework fallback for dynamic repacking: four byte-preserving,
    # independently shuffled R2-K1 raw-row shards. The trainer merges and packs
    # these once, then enforces exposure by the raw group IDs inside each pack.
    canonical_k1 = temporary / "rec_r2_k1_train.jsonl"
    canonical_lines = canonical_k1.read_text(encoding="utf-8").splitlines(keepends=True)
    shard_names = []
    for shard in range(4):
        shuffled = list(canonical_lines)
        random.Random(19260817 + shard).shuffle(shuffled)
        shard_name = f"rec_r2_k1_pack_seed_{shard}_train"
        (temporary / f"{shard_name}.jsonl").write_text("".join(shuffled), encoding="utf-8")
        shard_names.append(shard_name)
    dataset_info = json.loads((SOURCE / "dataset_info.json").read_text(encoding="utf-8"))
    for shard_name in shard_names:
        dataset_info[shard_name] = {"file_name": f"{shard_name}.jsonl", "columns": {"prompt": "prompt", "response": "response", "system": "system"}}
    (temporary / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalog = {"schema_version": 2, "train_groups": list(train_groups.values()), "validation_groups": list(val_groups.values()), "train_domain_inventory": train_inventory, "validation_domain_inventory": val_inventory, "negative_catalog_policy": "FDR training negatives are train-only; validation audit uses validation-only catalog."}
    (temporary / "rec_group_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    audit = r2_audit()
    report = {"schema_version": 1, "source": str(SOURCE), "text_rows": "V2 text fields unchanged; only globally unique rec_group_index is rewritten", "train_validation_group_overlap": 0, "r2_bucket_feasibility": audit, "r2_k1_pack_shards": shard_names, "r2_k1_exposure_cap": {"reuse_unit": "prompt_group_id", "max_total_exposure_factor": 4.0, "includes_first_exposure": True, "reservation": {"stage_a": .10, "stage_b": .15, "stage_c1": .30, "stage_c2": .45}}}
    (temporary / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if OUT.exists():
        shutil.rmtree(OUT)
    temporary.rename(OUT)
    REPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "build_report.json", REPORT / "build_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
