#!/usr/bin/env python3
"""Build leakage-safe Full/R1/R2 data for candidate-level recommendation V2.

The established V3 split and text are reused. Full and R2 receive explicit
multi-positive future-set metadata; R1 is copied unchanged. The output also
contains a compact integer-indexed group catalog used by the trainer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/data/LLm-8B/code/train")
SOURCE = ROOT / "data/material140k_userclean_rec_curriculum_fullft_v3"
RAW_FULL = ROOT / "data/material140k_userrec_full_v1/recommendation_think.jsonl"
OUT = ROOT / "data/rec_candidate_curriculum_v2"
REPORT = ROOT / "reports/rec_candidate_curriculum_v2"
SEED = 19260817
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)


def stable_digest(text: str) -> str:
    return hashlib.sha256(f"{SEED}\0{text}".encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def sid_parts(value: str) -> tuple[str, int, int, int]:
    match = SID_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid SID: {value}")
    return match.group("domain"), int(match.group("a")), int(match.group("b")), int(match.group("c"))


def final_sid(response: str) -> str:
    matches = list(SID_RE.finditer(response))
    if not matches:
        raise ValueError("response has no SID")
    return matches[-1].group(0)


def entropy_bucket(k: int) -> str:
    return "k1" if k == 1 else "k2" if k <= 4 else "k3"


def exact_val_prompt_keys() -> set[str]:
    report = json.loads((SOURCE / "build_report.json").read_text(encoding="utf-8"))
    expected = int(report["recommendation"]["validation_prompt_groups"])
    keys = sorted({json.loads(line)["prompt"].strip() for line in RAW_FULL.open(encoding="utf-8")})
    ordered = sorted(keys, key=lambda key: stable_digest(f"recommendation-prompt-groups\0{key}"))
    return set(ordered[:expected])


def enrich_full() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    rows = load_jsonl(RAW_FULL)
    positives: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        positives[row["prompt"].strip()].add(final_sid(row["response"]))
    val_keys = exact_val_prompt_keys()
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    catalog: dict[str, list[str]] = {}
    for row in rows:
        prompt_key = row["prompt"].strip()
        target = final_sid(row["response"])
        group_id = stable_digest("prompt-group\0" + prompt_key)[:24]
        group_key = "full:" + group_id
        future = sorted(positives[prompt_key])
        enriched = dict(row)
        enriched.update(
            {
                "rec_group_key": group_key,
                "prompt_group_id": group_id,
                "final_target_sid": target,
                "positive_future_sids": future,
                "group_unique_sid_count": len(future),
                "future_entropy_bucket": entropy_bucket(len(future)),
                "target_domain": sid_parts(target)[0],
            }
        )
        catalog[group_key] = future
        (val if prompt_key in val_keys else train).append(enriched)
    return train, val, catalog


def enrich_r2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    catalog: dict[str, list[str]] = {}
    for split in ("train", "val"):
        rows = load_jsonl(SOURCE / f"rec_r2_{split}.jsonl")
        groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        unique: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key2 = (row["prompt_group_id"], row["sanitized_cot"])
            groups[key2].add(row["final_target_sid"])
            unique.setdefault((*key2, row["final_target_sid"]), row)
        for (group_id, sanitized, target), row in unique.items():
            future = sorted(groups[(group_id, sanitized)])
            cot_hash = hashlib.sha256(sanitized.encode()).hexdigest()[:16]
            group_key = f"r2:{group_id}:{cot_hash}"
            enriched = dict(row)
            enriched.update(
                {
                    "rec_group_key": group_key,
                    "positive_future_sids": future,
                    "sample_weight": 1.0 / len(future),
                    "group_unique_sid_count": len(future),
                    "future_entropy_bucket": entropy_bucket(len(future)),
                    "target_domain": sid_parts(target)[0],
                }
            )
            catalog[group_key] = future
            output[split].append(enriched)
    return output["train"], output["val"], catalog


def dataset_info() -> dict[str, Any]:
    names = ["rec_r1_train", "rec_full_val", "rec_r1_val", "rec_r2_val"]
    names += [f"rec_{task}_{bucket}_train" for task in ("full", "r2") for bucket in ("k1", "k2", "k3")]
    return {
        name: {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        for name in names
    }


def main() -> None:
    temporary = OUT.with_name(OUT.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    full_train, full_val, full_catalog = enrich_full()
    r2_train, r2_val, r2_catalog = enrich_r2()
    r1_train = load_jsonl(SOURCE / "rec_r1_train.jsonl")
    r1_val = load_jsonl(SOURCE / "rec_r1_val.jsonl")
    pools = {
        "rec_r1_train": r1_train,
        "rec_r1_val": r1_val,
        "rec_full_val": full_val,
        "rec_r2_val": r2_val,
    }
    for task, rows in (("full", full_train), ("r2", r2_train)):
        for bucket in ("k1", "k2", "k3"):
            pools[f"rec_{task}_{bucket}_train"] = [
                row for row in rows if row["future_entropy_bucket"] == bucket
            ]
    counts = {name: dump_jsonl(temporary / f"{name}.jsonl", rows) for name, rows in pools.items()}
    catalog = {**full_catalog, **r2_catalog}
    group_keys = sorted(catalog)
    group_index = {key: index + 1 for index, key in enumerate(group_keys)}
    group_rows = []
    domain_inventory: defaultdict[str, set[str]] = defaultdict(set)
    for key in group_keys:
        future = catalog[key]
        for sid in future:
            domain_inventory[sid_parts(sid)[0]].add(sid)
        group_rows.append(
            {
                "group_index": group_index[key],
                "group_key": key,
                "positive_future_sids": future,
                "domain": sid_parts(future[0])[0],
                "k": len(future),
                "entropy_bucket": entropy_bucket(len(future)),
            }
        )
    for name in (
        "rec_full_k1_train", "rec_full_k2_train", "rec_full_k3_train", "rec_full_val",
        "rec_r2_k1_train", "rec_r2_k2_train", "rec_r2_k3_train", "rec_r2_val",
    ):
        path = temporary / f"{name}.jsonl"
        rows = load_jsonl(path)
        for row in rows:
            row["rec_group_index"] = group_index[row["rec_group_key"]]
        dump_jsonl(path, rows)
    (temporary / "rec_group_catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "groups": group_rows,
                "domain_inventory": {key: sorted(value) for key, value in domain_inventory.items()},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    (temporary / "dataset_info.json").write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "source": str(SOURCE),
        "raw_full_source": str(RAW_FULL),
        "seed": SEED,
        "counts": counts,
        "group_count": len(group_rows),
        "max_k": max(row["k"] for row in group_rows),
        "entropy_counts": dict(Counter(row["entropy_bucket"] for row in group_rows)),
        "full_text_unchanged": True,
        "r1_text_unchanged": True,
        "r2_exact_duplicates_removed": True,
        "target_leakage_policy": "reused leakage-safe V3 R1/R2 text",
    }
    (temporary / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if OUT.exists():
        shutil.rmtree(OUT)
    temporary.rename(OUT)
    REPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "build_report.json", REPORT / "build_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
