#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SID_RE = re.compile(r"(?P<sid><\|(?P<domain>prod|video|living|ad)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)")
DOMAIN_NAMES = {
    "prod": "商品",
    "video": "视频",
    "living": "直播",
    "ad": "广告",
}
DOMAIN_ORDER = ("商品", "视频", "直播", "广告")


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def strip_prompt_mode(value: str) -> str:
    return re.sub(r"\s*/(?:think|no_think)\s*$", "", value).rstrip()


def canonical_prompt(obj: dict[str, Any]) -> str:
    return json.dumps(
        [obj.get("instruction", ""), strip_prompt_mode(obj.get("input", "")), obj.get("history", [])],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_answer(output: str) -> dict[str, Any] | None:
    close = output.find("</think>")
    if "<think>" in output and close < 0:
        return None
    answer = output[close + len("</think>") :] if close >= 0 else output
    matches = list(SID_RE.finditer(answer))
    if not matches:
        return None
    return {
        "cot_prefix": output[: close + len("</think>")] if close >= 0 else "",
        "answer_prefix": answer[: matches[0].start()],
        "matches": matches,
    }


def collect_groups(source: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "raw_samples": 0,
        "parse_failure_rows": 0,
        "unknown_domain_gold": 0,
        "gold_occurrences": 0,
        "duplicate_gold_removed": 0,
        "cot_inconsistent_groups": 0,
        "answer_prefix_inconsistent_groups": 0,
        "skipped_groups": 0,
        "skipped_gold": 0,
        "inconsistent_examples": [],
    }
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            obj = json.loads(line)
            stats["raw_samples"] += 1
            parsed = parse_answer(obj.get("output", ""))
            if parsed is None:
                stats["parse_failure_rows"] += 1
                continue
            for match in parsed["matches"]:
                stats["gold_occurrences"] += 1
                domain = DOMAIN_NAMES[match.group("domain")]
                key = (canonical_prompt(obj), domain)
                group = groups.setdefault(
                    key,
                    {
                        "prompt_key": key[0],
                        "domain": domain,
                        "records": [],
                        "by_sid": {},
                    },
                )
                sid = match.group("sid")
                record = {
                    "line": line_number,
                    "obj": obj,
                    "sid": sid,
                    "cot_prefix": parsed["cot_prefix"],
                    "answer_prefix": parsed["answer_prefix"],
                }
                group["records"].append(record)
                group["by_sid"].setdefault(sid, record)

    valid_groups = {}
    for key, group in groups.items():
        records = group["records"]
        group["gold_sids"] = list(group["by_sid"])
        stats["duplicate_gold_removed"] += len(records) - len(group["gold_sids"])
        cot_values = {record["cot_prefix"] for record in records}
        answer_prefix_values = {record["answer_prefix"] for record in records}
        if len(cot_values) != 1 or None in cot_values:
            stats["cot_inconsistent_groups"] += 1
            stats["skipped_groups"] += 1
            stats["skipped_gold"] += len(group["gold_sids"])
            if len(stats["inconsistent_examples"]) < 20:
                stats["inconsistent_examples"].append(
                    {
                        "domain": group["domain"],
                        "lines": sorted({record["line"] for record in records})[:8],
                        "gold_sids": group["gold_sids"][:8],
                        "cot_count": len(cot_values),
                    }
                )
            continue
        if len(answer_prefix_values) != 1:
            stats["answer_prefix_inconsistent_groups"] += 1
        group["cot_prefix"] = records[0]["cot_prefix"]
        group["answer_prefix"] = records[0]["answer_prefix"]
        valid_groups[key] = group
    return valid_groups, stats


def assign_paths(groups: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    assignments: dict[tuple[str, str], set[str]] = {}
    by_domain: dict[str, list[tuple[tuple[str, str], dict[str, Any]]]] = defaultdict(list)
    for key, group in groups.items():
        by_domain[group["domain"]].append((key, group))

    for domain, domain_groups in by_domain.items():
        total = sum(len(group["gold_sids"]) for _, group in domain_groups)
        even_cot = sum(len(group["gold_sids"]) // 2 for _, group in domain_groups)
        odd_groups = [(key, group) for key, group in domain_groups if len(group["gold_sids"]) % 2]
        target_cot = total // 2
        if total % 2 and stable_hash(domain) % 2:
            target_cot += 1
        extra_cot = target_cot - even_cot
        if not 0 <= extra_cot <= len(odd_groups):
            raise RuntimeError(f"Invalid balanced assignment for {domain}: {extra_cot} of {len(odd_groups)}")
        odd_groups.sort(key=lambda pair: stable_hash(pair[0][0] + "\0" + pair[0][1]))
        extra_to_cot = {key for key, _ in odd_groups[:extra_cot]}
        for key, group in domain_groups:
            gold_sids = group["gold_sids"]
            paths: set[str] = set()
            use_cot_for_first = len(gold_sids) % 2 == 0 or key in extra_to_cot
            for index, sid in enumerate(gold_sids):
                is_cot = index % 2 == 0 if use_cot_for_first else index % 2 == 1
                paths.add(f"{sid}\t{'cot' if is_cot else 'nocot'}")
            assignments[key] = paths
    return assignments


def make_record(group: dict[str, Any], sid: str, mode: str) -> dict[str, Any]:
    representative = copy.deepcopy(group["by_sid"][sid]["obj"])
    base_input = strip_prompt_mode(representative.get("input", ""))
    representative["input"] = base_input + "\n/think" if mode == "cot" else base_input + "\n/no_think"
    if mode == "cot":
        representative["output"] = (group["cot_prefix"] + group["answer_prefix"] + sid).rstrip()
    else:
        representative["output"] = sid
    return representative


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("wb") as file:
        for record in records:
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            file.write(encoded)
            digest.update(encoded)
    return {"path": str(path), "records": len(records), "sha256": digest.hexdigest()}


def generate(args: argparse.Namespace) -> None:
    groups, stats = collect_groups(args.source)
    assignments = assign_paths(groups)
    cot_records: list[dict[str, Any]] = []
    nocot_records: list[dict[str, Any]] = []
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    ordered_groups = sorted(groups.items(), key=lambda pair: (DOMAIN_ORDER.index(pair[1]["domain"]), stable_hash(pair[0][0] + "\0" + pair[0][1])))
    for key, group in ordered_groups:
        for sid in group["gold_sids"]:
            modes = assignments[key]
            mode = "cot" if f"{sid}\tcot" in modes else "nocot"
            record = make_record(group, sid, mode)
            (cot_records if mode == "cot" else nocot_records).append(record)
            domain_counts[group["domain"]][mode] += 1

    stats["unique_prompt_domain_groups"] = len(groups)
    stats["single_gold_groups"] = sum(len(group["gold_sids"]) == 1 for group in groups.values())
    stats["multi_gold_groups"] = sum(len(group["gold_sids"]) > 1 for group in groups.values())
    stats["deduplicated_gold_total"] = sum(len(group["gold_sids"]) for group in groups.values())
    stats["cot_samples"] = len(cot_records)
    stats["nocot_samples"] = len(nocot_records)
    stats["cot_nocot_ratio"] = f"{len(cot_records)}:{len(nocot_records)}"
    stats["domain_counts"] = {domain: dict(domain_counts[domain]) for domain in DOMAIN_ORDER if domain in domain_counts}
    stats["output_total"] = len(cot_records) + len(nocot_records)
    stats["completeness_check"] = stats["output_total"] == stats["deduplicated_gold_total"]
    if stats["output_total"] != stats["deduplicated_gold_total"]:
        raise RuntimeError("Output count does not equal deduplicated gold count")

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    temporary = args.output_root.with_name(f".{args.output_root.name}.tmp")
    temporary.mkdir(parents=True)
    try:
        cot_path = temporary / "onereason_recommendation_cot_v2_dual.jsonl"
        nocot_path = temporary / "onereason_recommendation_nocot_v2_dual.jsonl"
        cot_file = write_jsonl(cot_path, cot_records)
        nocot_file = write_jsonl(nocot_path, nocot_records)
        cot_file["path"] = str(args.output_root / cot_path.name)
        nocot_file["path"] = str(args.output_root / nocot_path.name)
        stats["files"] = {"cot": cot_file, "nocot": nocot_file}
        (temporary / "DUAL_MANIFEST.json").write_text(json.dumps({"version": args.version, "source": str(args.source), "grouping_key": "instruction + input/history without mode marker + target domain", "statistics": stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    registry = json.loads(args.dataset_info.read_text(encoding="utf-8"))
    base_entry = registry["onereason_recommendation_cot"]
    cot_name = "onereason_recommendation_cot_v2_dual"
    nocot_name = "onereason_recommendation_nocot_v2_dual"
    if cot_name in registry or nocot_name in registry:
        raise FileExistsError("Dual dataset registry entry already exists")
    cot_entry = copy.deepcopy(base_entry)
    nocot_entry = copy.deepcopy(base_entry)
    cot_entry["file_name"] = str(args.output_root / cot_path.name)
    nocot_entry["file_name"] = str(args.output_root / nocot_path.name)
    registry[cot_name] = cot_entry
    registry[nocot_name] = nocot_entry
    args.dataset_info.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.version_manifest.exists():
        version_manifest = json.loads(args.version_manifest.read_text(encoding="utf-8"))
        versions = version_manifest.setdefault("versions", {})
        if args.version in versions:
            raise FileExistsError(f"Version {args.version} already exists in {args.version_manifest}")
        versions[args.version] = {
            "parent": args.parent,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "description": "Recommendation CoT/No-think dual-route deduplication.",
            "overrides": {
                "onereason_recommendation_cot": {
                    "registry_name": cot_name,
                    "file_name": str(args.output_root / cot_path.name),
                    "records": len(cot_records),
                    "sha256": cot_file["sha256"],
                }
            },
            "extra_registry_entries": {
                "onereason_recommendation_nocot": {
                    "registry_name": nocot_name,
                    "file_name": str(args.output_root / nocot_path.name),
                    "records": len(nocot_records),
                    "sha256": nocot_file["sha256"],
                }
            },
        }
        args.version_manifest.write_text(json.dumps(version_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create V2 recommendation CoT/No-think dual-route data.")
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/lf_data_versions/alltrain/v2_recommendation_dual"))
    parser.add_argument("--dataset-info", type=Path, default=Path("/app/LLaMA-Factory/data/dataset_info.json"))
    parser.add_argument("--version-manifest", type=Path, default=Path("/app/LLaMA-Factory/data/onereason_dataset_versions.json"))
    parser.add_argument("--version", default="v2_recommendation_dual")
    parser.add_argument("--parent", default="v2_recommendation_cot_complete_all")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
