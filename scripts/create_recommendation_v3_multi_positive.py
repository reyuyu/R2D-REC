#!/usr/bin/env python3
"""Create the metadata-only Recommendation V3 Multi-Positive data version.

The source rows remain byte-for-byte equivalent in all existing JSON fields.
Only recommendation_* provenance fields are added.  Grouping follows the
canonical prompt used by the V2 dual generator plus the target SID domain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SID_RE = re.compile(r"(?P<sid><\|(?P<domain>prod|video|living|ad)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)")
DOMAIN_NAMES = {"prod": "商品", "video": "视频", "living": "直播", "ad": "广告"}
DOMAIN_ORDER = ("商品", "视频", "直播", "广告")
META_FIELDS = (
    "recommendation_group_id",
    "recommendation_group_size",
    "recommendation_all_gold_sids",
    "recommendation_current_gold_sid",
)


def strip_prompt_mode(value: str) -> str:
    return re.sub(r"\s*/(?:think|no_think)\s*$", "", value).rstrip()


def canonical_prompt(obj: dict[str, Any]) -> str:
    return json.dumps(
        [obj.get("instruction", ""), strip_prompt_mode(obj.get("input", "")), obj.get("history", [])],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def group_id(prompt_key: str, domain: str) -> str:
    return hashlib.sha256((prompt_key + "\0" + domain).encode("utf-8")).hexdigest()


def parse_single_gold(obj: dict[str, Any]) -> tuple[str, str] | None:
    output = obj.get("output", "")
    close = output.find("</think>")
    answer = output[close + len("</think>") :] if close >= 0 else output
    matches = list(SID_RE.finditer(answer))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.group("sid"), DOMAIN_NAMES[match.group("domain")]


def qtile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def row_text_without_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in META_FIELDS}


def read_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"source_files": [], "raw_samples": 0, "parse_failure_rows": 0, "multi_gold_rows": 0}
    for path in paths:
        file_stats = {"path": str(path), "records": 0}
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                row = json.loads(line)
                stats["raw_samples"] += 1
                file_stats["records"] += 1
                parsed = parse_single_gold(row)
                if parsed is None:
                    stats["parse_failure_rows"] += 1
                    continue
                rows.append({"row": row, "sid": parsed[0], "domain": parsed[1], "source": str(path), "line": line_number})
        stats["source_files"].append(file_stats)
    return rows, stats


def build_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "unique_prompt_domain_groups": 0,
        "duplicate_gold_removed": 0,
        "cot_inconsistent_groups": 0,
        "inconsistent_examples": [],
        "unrecognized_domain_rows": 0,
    }
    for item in rows:
        row = item["row"]
        key = (canonical_prompt(row), item["domain"])
        group = groups.setdefault(
            key,
            {
                "prompt_key": key[0],
                "domain": key[1],
                "group_id": group_id(key[0], key[1]),
                "rows_by_sid": {},
                "ordered_sids": [],
                "cot_outputs": [],
                "all_rows": [],
            },
        )
        group["all_rows"].append(item)
        if item["sid"] in group["rows_by_sid"]:
            stats["duplicate_gold_removed"] += 1
            continue
        group["rows_by_sid"][item["sid"]] = item
        group["ordered_sids"].append(item["sid"])
        if "/think" in row.get("input", "") and "</think>" in row.get("output", ""):
            group["cot_outputs"].append(row.get("output", "").split("</think>", 1)[0])

    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        group["gold_sids"] = list(group["ordered_sids"])
        group["group_size"] = len(group["gold_sids"])
        if len(set(group["cot_outputs"])) > 1:
            stats["cot_inconsistent_groups"] += 1
            if len(stats["inconsistent_examples"]) < 20:
                stats["inconsistent_examples"].append(
                    {
                        "group_id": group["group_id"],
                        "domain": group["domain"],
                        "gold_count": group["group_size"],
                        "cot_variants": len(set(group["cot_outputs"])),
                        "lines": [item["line"] for item in group["all_rows"][:8]],
                    }
                )
        output.append(group)
    output.sort(key=lambda group: (DOMAIN_ORDER.index(group["domain"]), group["group_id"]))
    stats["unique_prompt_domain_groups"] = len(output)
    stats["deduplicated_gold_total"] = sum(group["group_size"] for group in output)
    stats["single_gold_groups"] = sum(group["group_size"] == 1 for group in output)
    stats["multi_gold_groups"] = sum(group["group_size"] > 1 for group in output)
    return output, stats


def augment_groups(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cot_records: list[dict[str, Any]] = []
    nocot_records: list[dict[str, Any]] = []
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_sizes: list[int] = []
    prefix_counts: Counter[str] = Counter()
    history_overlaps: list[int] = []
    for group in groups:
        all_sids = list(group["gold_sids"])
        group_sizes.append(len(all_sids))
        for sid in all_sids:
            row = copy.deepcopy(group["rows_by_sid"][sid]["row"])
            row.update(
                recommendation_group_id=group["group_id"],
                recommendation_group_size=len(all_sids),
                recommendation_all_gold_sids=all_sids,
                recommendation_current_gold_sid=sid,
            )
            mode = "cot" if "/think" in row.get("input", "") else "nocot"
            domain_counts[group["domain"]][mode] += 1
            if mode == "cot":
                cot_records.append(row)
            else:
                nocot_records.append(row)
            prefix_counts[sid.split("><s_a_", 1)[0] + ">"] += 1
            input_text = row.get("input", "")
            history_overlaps.append(sum(1 for candidate in all_sids if candidate in input_text))
    stats = {
        "cot_samples": len(cot_records),
        "nocot_samples": len(nocot_records),
        "cot_nocot_ratio": f"{len(cot_records)}:{len(nocot_records)}",
        "domain_counts": {domain: dict(domain_counts[domain]) for domain in DOMAIN_ORDER},
        "group_size_stats": {
            "mean": statistics.fmean(group_sizes) if group_sizes else 0.0,
            "p50": qtile(group_sizes, 0.50),
            "p90": qtile(group_sizes, 0.90),
            "p95": qtile(group_sizes, 0.95),
            "max": max(group_sizes, default=0),
        },
        "prefix_branch_stats": {"unique_prefixes": len(prefix_counts), "counts": dict(prefix_counts)},
        "history_overlap_stats": {
            "mean_gold_sids_found_in_prompt": statistics.fmean(history_overlaps) if history_overlaps else 0.0,
            "max_gold_sids_found_in_prompt": max(history_overlaps, default=0),
        },
    }
    return cot_records, nocot_records, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("wb") as file:
        for row in records:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            file.write(encoded)
            digest.update(encoded)
    return {"path": str(path), "records": len(records), "sha256": digest.hexdigest()}


def generate(args: argparse.Namespace) -> dict[str, Any]:
    rows, source_stats = read_rows([args.cot_source, args.nocot_source])
    parent_source = getattr(args, "parent_source", Path("/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl"))
    if parent_source.exists():
        source_stats["parent_v2_raw_source"] = str(parent_source)
        source_stats["parent_v2_raw_samples"] = sum(1 for _ in parent_source.open(encoding="utf-8"))
    else:
        source_stats["parent_v2_raw_source"] = str(parent_source)
        source_stats["parent_v2_raw_samples"] = None
    groups, group_stats = build_groups(rows)
    cot_records, nocot_records, output_stats = augment_groups(groups)
    total = len(cot_records) + len(nocot_records)
    if total != group_stats["deduplicated_gold_total"]:
        raise RuntimeError("V3 completeness check failed")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    temporary = args.output_root.with_name(f".{args.output_root.name}.tmp")
    temporary.mkdir(parents=True)
    try:
        cot_path = temporary / "onereason_recommendation_cot_v3_multi_positive.jsonl"
        nocot_path = temporary / "onereason_recommendation_nocot_v3_multi_positive.jsonl"
        files = {"cot": write_jsonl(cot_path, cot_records), "nocot": write_jsonl(nocot_path, nocot_records)}
        for file_info in files.values():
            file_info["path"] = str(args.output_root / Path(file_info["path"]).name)
        report = {
            "version": args.version,
            "parent": args.parent,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source_stats,
            "grouping_key": "canonical instruction + input/history without /think or /no_think + target domain",
            "metadata_fields": list(META_FIELDS),
            "statistics": {**source_stats, **group_stats, **output_stats, "output_total": total, "completeness_check": True, "files": files},
        }
        (temporary / "V3_MANIFEST.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    update_registries(args, files)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def update_registries(args: argparse.Namespace, files: dict[str, Any]) -> None:
    registry = json.loads(args.dataset_info.read_text(encoding="utf-8"))
    base_cot = registry["onereason_recommendation_cot_v2_dual"]
    base_nocot = registry["onereason_recommendation_nocot_v2_dual"]
    names = {"cot": "onereason_recommendation_cot_v3_multi_positive", "nocot": "onereason_recommendation_nocot_v3_multi_positive"}
    for mode, base in (("cot", base_cot), ("nocot", base_nocot)):
        if names[mode] in registry:
            raise FileExistsError(f"Registry entry already exists: {names[mode]}")
        entry = copy.deepcopy(base)
        entry["file_name"] = files[mode]["path"]
        registry[names[mode]] = entry
    args.dataset_info.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.version_manifest.exists():
        manifest = json.loads(args.version_manifest.read_text(encoding="utf-8"))
        versions = manifest.setdefault("versions", {})
        if args.version in versions:
            raise FileExistsError(f"Version already exists: {args.version}")
        versions[args.version] = {
            "parent": args.parent,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "description": "Recommendation V3 metadata-only multi-positive grouping derived from V2 dual data.",
            "overrides": {
                "onereason_recommendation_cot": {"registry_name": names["cot"], "file_name": files["cot"]["path"], "records": files["cot"]["records"], "sha256": files["cot"]["sha256"]},
                "onereason_recommendation_nocot": {"registry_name": names["nocot"], "file_name": files["nocot"]["path"], "records": files["nocot"]["records"], "sha256": files["nocot"]["sha256"]},
            },
        }
        args.version_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Recommendation V3 Multi-Positive metadata data.")
    parser.add_argument("--cot-source", type=Path, default=Path("/data/lf_data_versions/alltrain/v2_recommendation_dual/onereason_recommendation_cot_v2_dual.jsonl"))
    parser.add_argument("--nocot-source", type=Path, default=Path("/data/lf_data_versions/alltrain/v2_recommendation_dual/onereason_recommendation_nocot_v2_dual.jsonl"))
    parser.add_argument("--parent-source", type=Path, default=Path("/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/lf_data_versions/alltrain/v3_recommendation_multi_positive"))
    parser.add_argument("--dataset-info", type=Path, default=Path("/app/LLaMA-Factory/data/dataset_info.json"))
    parser.add_argument("--version-manifest", type=Path, default=Path("/app/LLaMA-Factory/data/onereason_dataset_versions.json"))
    parser.add_argument("--version", default="v3_recommendation_multi_positive")
    parser.add_argument("--parent", default="v2_recommendation_dual")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
