#!/usr/bin/env python3
"""Build the deterministic, diversity-preserving balanced LoRA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


SEED = 19260817
SOURCE = Path("/data/LLm-8B/code/train/data/classified_jsonl/llamafactory")
OUTPUT = Path("/data/LLm-8B/code/train/data/balanced_lora_v1")
REPORT = Path("/data/LLm-8B/code/train/reports/balanced_lora_v1.json")

MATERIAL_QUOTAS = {
    "material_no_think_semantic_to_sid": 150_000,
    "material_no_think_sid_to_semantic": 120_000,
    "material_think_semantic_to_sid": 100_000,
    "material_think_sid_to_semantic": 80_000,
}
FULL_DATASETS = (
    "user_think",
    "recommendation_think",
    "world_500",
    "world_819",
)
SID_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)
TYPE_ORDER = ("video", "prod", "ad", "living")


def stable_hash(*parts: object) -> int:
    payload = "\0".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def material_sid(name: str, row: dict[str, str]) -> tuple[str, int, int, int]:
    field = row["response"] if "semantic_to_sid" in name else row["prompt"]
    matches = SID_RE.findall(field)
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one SID, found {len(matches)}")
    kind, a, b, c = matches[0]
    return kind, int(a), int(b), int(c)


def equalized_type_quotas(available: Counter[str], total: int) -> dict[str, int]:
    quotas = {kind: 0 for kind in TYPE_ORDER}
    remaining = total
    active = list(TYPE_ORDER)
    while active:
        share = remaining // len(active)
        constrained = [kind for kind in active if available[kind] <= share]
        if constrained:
            for kind in constrained:
                quotas[kind] = available[kind]
                remaining -= quotas[kind]
                active.remove(kind)
            continue
        for index, kind in enumerate(active):
            quotas[kind] = share + (1 if index < remaining % len(active) else 0)
        remaining = 0
        break
    if remaining != 0:
        raise ValueError(f"could not allocate {total} rows across {dict(available)}")
    return quotas


def select_material(
    name: str,
    quota: int,
    already_selected_sids: set[tuple[str, int, int, int]],
) -> tuple[set[int], dict[str, object], set[tuple[str, int, int, int]]]:
    source = SOURCE / f"{name}.jsonl"
    candidates: list[tuple[int, tuple[str, int, int, int], int]] = []
    component_frequency: Counter[tuple[str, str, int]] = Counter()
    type_counts: Counter[str] = Counter()
    all_sids: set[tuple[str, int, int, int]] = set()

    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            sid = material_sid(name, row)
            kind, a, b, c = sid
            candidates.append((index, sid, stable_hash(name, sid, row["prompt"], row["response"])))
            type_counts[kind] += 1
            all_sids.add(sid)
            component_frequency[(kind, "a", a)] += 1
            component_frequency[(kind, "b", b)] += 1
            component_frequency[(kind, "c", c)] += 1

    type_quotas = equalized_type_quotas(type_counts, quota)
    by_type: dict[str, list[tuple[int, tuple[str, int, int, int], int]]] = {
        kind: [] for kind in TYPE_ORDER
    }
    for candidate in candidates:
        by_type[candidate[1][0]].append(candidate)

    selected_indices: set[int] = set()
    selected_sids: set[tuple[str, int, int, int]] = set()
    selected_types: Counter[str] = Counter()
    for kind in TYPE_ORDER:
        ranked = sorted(
            by_type[kind],
            key=lambda item: (
                item[1] not in already_selected_sids,
                -max(
                    component_frequency[(kind, "a", item[1][1])],
                    component_frequency[(kind, "b", item[1][2])],
                    component_frequency[(kind, "c", item[1][3])],
                ),
                -sum(
                    component_frequency[(kind, level, value)]
                    for level, value in zip(("a", "b", "c"), item[1][1:])
                ),
                -item[2],
            ),
            reverse=True,
        )
        unique_ranked = []
        duplicate_ranked = []
        seen_in_type: set[tuple[str, int, int, int]] = set()
        for candidate in ranked:
            if candidate[1] in seen_in_type:
                duplicate_ranked.append(candidate)
            else:
                unique_ranked.append(candidate)
                seen_in_type.add(candidate[1])
        chosen = (unique_ranked + duplicate_ranked)[: type_quotas[kind]]
        for index, sid, _ in chosen:
            selected_indices.add(index)
            selected_sids.add(sid)
            selected_types[kind] += 1

    destination = OUTPUT / f"{name}.jsonl"
    temp = destination.with_suffix(".jsonl.tmp")
    with source.open(encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
        written = 0
        for index, line in enumerate(src):
            if index in selected_indices:
                dst.write(line)
                written += 1
    if written != quota:
        temp.unlink(missing_ok=True)
        raise ValueError(f"{name}: wrote {written}, expected {quota}")
    os.replace(temp, destination)

    component_coverage = {}
    for level, offset in (("a", 1), ("b", 2), ("c", 3)):
        full = {(sid[0], sid[offset]) for sid in all_sids}
        kept = {(sid[0], sid[offset]) for sid in selected_sids}
        component_coverage[level] = {
            "kept": len(kept),
            "full": len(full),
            "ratio": len(kept) / len(full),
        }
    stats = {
        "source_rows": len(candidates),
        "selected_rows": written,
        "type_quotas": type_quotas,
        "selected_types": dict(selected_types),
        "unique_sid_source": len(all_sids),
        "unique_sid_selected": len(selected_sids),
        "new_sid_vs_previous_groups": len(selected_sids - already_selected_sids),
        "component_coverage": component_coverage,
    }
    return selected_indices, stats, selected_sids


def copy_full_dataset(name: str) -> int:
    source = SOURCE / f"{name}.jsonl"
    destination = OUTPUT / f"{name}.jsonl"
    temp = destination.with_suffix(".jsonl.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, destination)
    with destination.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def select_user_no_think(quota: int = 16_000) -> dict[str, object]:
    name = "user_no_think"
    source = SOURCE / f"{name}.jsonl"
    candidates: list[tuple[int, int, int, set[str]]] = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            answer = row["response"].split("</think>", 1)[-1]
            target_types = set(SID_RE.findall(answer))
            kinds = {hit[0] for hit in target_types}
            candidates.append((index, len(row["prompt"]), stable_hash(name, row["prompt"]), kinds))

    ordered_lengths = sorted(length for _, length, _, _ in candidates)
    boundaries = [ordered_lengths[(len(ordered_lengths) * q) // 5] for q in range(1, 5)]

    def bucket(length: int) -> int:
        return sum(length > boundary for boundary in boundaries)

    selected: set[int] = set()
    selected_buckets: Counter[int] = Counter()
    per_bucket = [quota // 5 + (1 if i < quota % 5 else 0) for i in range(5)]
    for bucket_id in range(5):
        pool = [item for item in candidates if bucket(item[1]) == bucket_id]
        pool.sort(key=lambda item: (len(item[3]), -item[2]), reverse=True)
        for index, _, _, _ in pool[: per_bucket[bucket_id]]:
            selected.add(index)
            selected_buckets[bucket_id] += 1

    destination = OUTPUT / f"{name}.jsonl"
    temp = destination.with_suffix(".jsonl.tmp")
    selected_types: Counter[str] = Counter()
    with source.open(encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
        written = 0
        for index, line in enumerate(src):
            if index in selected:
                dst.write(line)
                row = json.loads(line)
                answer = row["response"].split("</think>", 1)[-1]
                selected_types.update({hit[0] for hit in SID_RE.findall(answer)})
                written += 1
    if written != quota:
        temp.unlink(missing_ok=True)
        raise ValueError(f"{name}: wrote {written}, expected {quota}")
    os.replace(temp, destination)
    return {
        "source_rows": len(candidates),
        "selected_rows": written,
        "prompt_char_quintile_boundaries": boundaries,
        "selected_per_quintile": dict(selected_buckets),
        "rows_with_target_type": {kind: selected_types[kind] for kind in TYPE_ORDER},
    }


def select_recommendation_no_think(quota: int = 24_000) -> dict[str, object]:
    name = "recommendation_no_think"
    source = SOURCE / f"{name}.jsonl"
    candidates: dict[str, list[tuple[int, int]]] = {kind: [] for kind in TYPE_ORDER}
    counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            answer = row["response"].split("</think>", 1)[-1]
            hits = SID_RE.findall(answer)
            if len(hits) != 1:
                raise ValueError(f"{name}:{index + 1}: expected one target SID")
            kind = hits[0][0]
            counts[kind] += 1
            candidates[kind].append((index, stable_hash(name, row["prompt"], answer)))

    raw = {kind: quota * counts[kind] / sum(counts.values()) for kind in TYPE_ORDER}
    type_quotas = {kind: int(raw[kind]) for kind in TYPE_ORDER}
    for kind in sorted(TYPE_ORDER, key=lambda k: raw[k] - type_quotas[k], reverse=True):
        if sum(type_quotas.values()) == quota:
            break
        type_quotas[kind] += 1

    selected: set[int] = set()
    for kind in TYPE_ORDER:
        candidates[kind].sort(key=lambda item: item[1])
        selected.update(index for index, _ in candidates[kind][: type_quotas[kind]])

    destination = OUTPUT / f"{name}.jsonl"
    temp = destination.with_suffix(".jsonl.tmp")
    with source.open(encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
        written = 0
        for index, line in enumerate(src):
            if index in selected:
                dst.write(line)
                written += 1
    if written != quota:
        temp.unlink(missing_ok=True)
        raise ValueError(f"{name}: wrote {written}, expected {quota}")
    os.replace(temp, destination)
    return {
        "source_rows": sum(counts.values()),
        "selected_rows": written,
        "source_types": dict(counts),
        "selected_types": type_quotas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {OUTPUT}; use --overwrite")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "seed": SEED,
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "material": {},
        "datasets": {},
    }
    selected_material_sids: set[tuple[str, int, int, int]] = set()
    for name, quota in MATERIAL_QUOTAS.items():
        _, stats, selected_sids = select_material(name, quota, selected_material_sids)
        report["material"][name] = stats
        selected_material_sids.update(selected_sids)
        print(f"selected {name}: {quota}", flush=True)

    report["datasets"]["user_no_think"] = select_user_no_think()
    print("selected user_no_think: 16000", flush=True)
    report["datasets"]["recommendation_no_think"] = select_recommendation_no_think()
    print("selected recommendation_no_think: 24000", flush=True)
    for name in FULL_DATASETS:
        rows = copy_full_dataset(name)
        report["datasets"][name] = {"source_rows": rows, "selected_rows": rows}
        print(f"copied {name}: {rows}", flush=True)

    report["material_unique_sid_union"] = len(selected_material_sids)
    report["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(OUTPUT.glob("*.jsonl"))
    }
    report["total_rows"] = sum(
        sum(1 for _ in path.open(encoding="utf-8")) for path in OUTPUT.glob("*.jsonl")
    )
    temp_report = REPORT.with_suffix(".json.tmp")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_report, REPORT)
    print(json.dumps({"total_rows": report["total_rows"], "report": str(REPORT)}), flush=True)


if __name__ == "__main__":
    main()
