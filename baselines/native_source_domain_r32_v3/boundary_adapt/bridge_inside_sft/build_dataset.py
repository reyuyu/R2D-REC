"""Build the deterministic train/holdout split and transition-only rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .common import (
    ADAPTER_SHA256,
    BASE,
    DOMAIN_ORDER,
    FRESH_BATA,
    SOURCE_SHA256,
    SPLIT_SALT,
    construct_token_row,
    count_distribution,
    file_sha,
    source_group_id,
    stable_hash,
)


def load_unique_source(source: Path) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    duplicates = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            group = source_group_id(row)
            signature = stable_hash(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if group in unique:
                if signatures[group] != signature:
                    # Duplicate rows may differ only in serialized JSON spacing; compare objects.
                    if unique[group] != row:
                        raise RuntimeError(f"CONFLICTING_DUPLICATE_GROUP={group}")
                duplicates += 1
                continue
            unique[group], signatures[group] = row, signature
    return list(unique.values()), duplicates


def historical_groups(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {str(item["recommendation_group_id"]) for item in payload["items"]}
    if len(groups) != 12:
        raise RuntimeError(f"HISTORICAL_GROUP_COUNT={len(groups)}")
    return groups


def split_rows(rows: list[dict[str, Any]], historical: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["target_domain"]].append(row)
    holdout_ids = set()
    for domain in DOMAIN_ORDER:
        candidates = by_domain[domain]
        target = round(len(candidates) * .10)
        forced = [row for row in candidates if row["group_id"] in historical]
        remaining = sorted(
            (row for row in candidates if row["group_id"] not in historical),
            key=lambda row: (stable_hash(SPLIT_SALT + row["group_id"]), row["group_id"]),
        )
        selected = forced + remaining[:target - len(forced)]
        if len(selected) != target:
            raise RuntimeError(f"HOLDOUT_TARGET_FAIL domain={domain}")
        holdout_ids.update(row["group_id"] for row in selected)
    train = [row for row in rows if row["group_id"] not in holdout_ids]
    holdout = [row for row in rows if row["group_id"] in holdout_ids]
    return train, holdout


def primary_heldout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for domain in DOMAIN_ORDER:
        domain_rows = [row for row in rows if row["target_domain"] == domain]
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in domain_rows:
            k_bucket = "K1" if row["gold_count"] == 1 else "K2PLUS"
            buckets[(row["prompt_family"], k_bucket)].append(row)
        for values in buckets.values():
            values.sort(key=lambda row: (stable_hash("primary_heldout_256|" + row["group_id"]), row["group_id"]))
        domain_selected = []
        keys = sorted(buckets)
        while len(domain_selected) < 64:
            progressed = False
            for key in keys:
                if buckets[key] and len(domain_selected) < 64:
                    domain_selected.append(buckets[key].pop(0))
                    progressed = True
            if not progressed:
                raise RuntimeError(f"INSUFFICIENT_PRIMARY_HELDOUT domain={domain}")
        selected.extend(domain_selected)
    if len(selected) != 256 or len({row["group_id"] for row in selected}) != 256:
        raise RuntimeError("PRIMARY_HELDOUT_CARDINALITY_FAIL")
    return selected


def public_row(row: dict[str, Any], *, training: bool) -> dict[str, Any]:
    if training:
        return {key: row[key] for key in (
            "group_id", "target_domain", "prompt_family", "gold_count", "input_ids", "labels",
            "labeled_token_count", "input_ids_sha256", "source_row_sha256",
        )}
    return row


def build(source: Path, historical_manifest: Path, output: Path) -> dict[str, Any]:
    if file_sha(source) != SOURCE_SHA256:
        raise RuntimeError("TRAIN_SOURCE_SHA256_MISMATCH")
    if file_sha(FRESH_BATA / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("FRESH_BATA_SHA256_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    raw_rows, duplicates = load_unique_source(source)
    rows = [construct_token_row(tokenizer, row) for row in raw_rows]
    if len(rows) != 15943:
        raise RuntimeError(f"UNIQUE_GROUP_COUNT={len(rows)}")
    expected_domains = {"video": 9227, "prod": 2232, "ad": 2415, "living": 2069}
    actual_domains = Counter(row["target_domain"] for row in rows)
    if dict(actual_domains) != expected_domains:
        raise RuntimeError(f"DOMAIN_COUNTS_MISMATCH={actual_domains}")
    historical = historical_groups(historical_manifest)
    train, holdout = split_rows(rows, historical)
    primary = primary_heldout(holdout)
    train_ids, holdout_ids = {row["group_id"] for row in train}, {row["group_id"] for row in holdout}
    if train_ids & holdout_ids or historical & train_ids or not historical <= holdout_ids:
        raise RuntimeError("SPLIT_LEAKAGE_GATE_FAIL")
    if {row["group_id"] for row in primary} & train_ids:
        raise RuntimeError("PRIMARY_HELDOUT_TRAIN_OVERLAP")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "train_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps(public_row(row, training=True), separators=(",", ":")) + "\n")
    with (output / "heldout_256_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"items": [public_row(row, training=False) for row in primary]}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    labeled_ids = [label for row in train for label in row["labels"] if label != -100]
    bridge_lengths = [len(row["bridge_token_ids"]) for row in train]
    label_lengths = [row["labeled_token_count"] for row in train]
    system_audit = []
    for row in sorted(rows, key=lambda item: stable_hash("system_audit|" + item["group_id"]))[:64]:
        text = tokenizer.decode(row["prompt_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        checks = {
            "system": text.count("<|im_start|>system") == 1,
            "user": text.count("<|im_start|>user") == 1,
            "assistant": text.count("<|im_start|>assistant") == 1,
        }
        if not all(checks.values()):
            raise RuntimeError(f"SYSTEM_RENDER_AUDIT_FAIL group={row['group_id']}")
        system_audit.append({"group_id": row["group_id"], "checks": checks})

    split = {
        "split_salt": SPLIT_SALT,
        "source": str(source),
        "source_sha256": SOURCE_SHA256,
        "unique_groups": len(rows),
        "byte_identical_duplicates_dropped": duplicates,
        "train_groups": len(train),
        "holdout_groups": len(holdout),
        "train_per_domain": dict(Counter(row["target_domain"] for row in train)),
        "holdout_per_domain": dict(Counter(row["target_domain"] for row in holdout)),
        "train_holdout_intersection": 0,
        "historical_probes_in_train": len(historical & train_ids),
        "historical_probes_in_holdout": len(historical & holdout_ids),
        "historical_probe_groups": sorted(historical),
        "train_group_ids": sorted(train_ids),
        "holdout_group_ids": sorted(holdout_ids),
        "primary_heldout_groups": 256,
        "primary_heldout_train_overlap": 0,
        "primary_per_domain": dict(Counter(row["target_domain"] for row in primary)),
        "primary_prompt_family": dict(Counter((row["target_domain"], row["prompt_family"]) for row in primary)),
        "primary_gold_count_bucket": dict(Counter((row["target_domain"], "K1" if row["gold_count"] == 1 else "K2PLUS") for row in primary)),
        "system_preserve_pass": True,
        "system_audit": system_audit,
        "bridge_variant_count": len({row["bridge"] for row in rows}),
        "bridge_token_length_distribution": count_distribution(bridge_lengths),
        "labeled_token_policy": "BRIDGE_PLUS_CLOSE_ONLY",
        "labeled_token_length_distribution": count_distribution(label_lengths),
        "labeled_abc_token_count": 0,
        "labeled_domain_token_count": 0,
        "labeled_cot_body_token_count": 0,
        "total_labeled_tokens": len(labeled_ids),
        "close_token_id": tokenizer.encode("</think>", add_special_tokens=False)[0],
        "domain_token_atomic": True,
    }
    (output / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--historical-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(Path(args.source), Path(args.historical_manifest), Path(args.output))
    print(json.dumps({key: result[key] for key in (
        "train_groups", "holdout_groups", "train_per_domain", "holdout_per_domain",
        "historical_probes_in_train", "historical_probes_in_holdout", "primary_heldout_groups",
        "bridge_variant_count", "bridge_token_length_distribution", "labeled_token_length_distribution",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
