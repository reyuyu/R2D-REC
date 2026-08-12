#!/usr/bin/env python3
"""Build the clean bata_baseline dataset from the reproduction Parquet.

Every archive row is preserved except understand_user, which is replaced in
order by the active BETA understand_user rows. Recommendation multi-positive
metadata is derived from the exact user prompt and target domain, but is not
used by the ordinary CE baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


EXPECTED_ARCHIVE_COUNTS = {
    "material_sample": 100_000,
    "sid_bucket_canonical_no_think": 11_298,
    "sid_bucket_reverse": 29_586,
    "understand_user": 32_452,
    "recommend": 48_269,
}
EXPECTED_BETA_USER_COUNT = 32_848
EXPECTED_OUTPUT_COUNTS = {
    **EXPECTED_ARCHIVE_COUNTS,
    "understand_user": EXPECTED_BETA_USER_COUNT,
}
EXPECTED_MATERIAL_DOMAINS = {"video": 30_092, "prod": 29_180, "ad": 22_768, "living": 17_960}
MATERIAL_DOMAIN_WEIGHTS = {
    "video": 1.1791795483099141,
    "prod": 0.7738397930531297,
    "ad": 1.133452723686895,
    "living": 0.8980530210503628,
}
SID_RE = re.compile(r"<[|](video|prod|ad|living)_begin[|]><s_a_[0-9]+><s_b_[0-9]+><s_c_[0-9]+>")
THINK_SUFFIX_RE = re.compile(r"/(?:no_think|think)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def archive_rows(path: Path) -> Iterator[dict[str, Any]]:
    columns = ["messages", "source", "data_source"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
        yield from batch.to_pylist()


def recommendation_identity(row: dict[str, Any]) -> tuple[tuple[str, str], str]:
    messages = row["messages"]
    matches = list(SID_RE.finditer(messages[2]["content"]))
    if not matches:
        raise ValueError("Recommendation response contains no complete target SID")
    current = matches[-1].group(0)
    domain = matches[-1].group(1)
    history = THINK_SUFFIX_RE.sub("", messages[1]["content"]).rstrip()
    return (history, domain), current


def group_id(key: tuple[str, str]) -> str:
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def message_projection(messages: list[dict[str, str]], source: str, data_source: str) -> bytes:
    payload = {"messages": messages, "source": source, "data_source": data_source}
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def output_projection(row: dict[str, Any], archive_source: str) -> bytes:
    messages = [
        {"role": "system", "content": row.get("system", "")},
        {"role": "user", "content": row["instruction"] + row.get("input", "")},
        {"role": "assistant", "content": row["output"]},
    ]
    return message_projection(messages, archive_source, row["data_source"])


def convert_archive(
    row: dict[str, Any], recommendation_groups: dict[tuple[str, str], list[str]]
) -> dict[str, Any]:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Archive row is not exactly system/user/assistant")
    metadata = ""
    if row["data_source"] == "recommend":
        key, current = recommendation_identity(row)
        golds = recommendation_groups[key]
        metadata = json.dumps(
            {
                "recommendation_group_id": group_id(key),
                "recommendation_group_size": len(golds),
                "recommendation_all_gold_sids": golds,
                "recommendation_current_gold_sid": current,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        segment = "recommendation_nocot" if "one_third_no_think" in row["source"] else "recommendation_cot"
    else:
        segment = row["data_source"]
    return {
        "instruction": messages[1]["content"],
        "input": "",
        "output": messages[2]["content"],
        "history": [],
        "system": messages[0]["content"],
        "data_source": row["data_source"],
        "source_segment": segment,
        "aux_metadata_json": metadata,
    }


def beta_user_rows(path: Path) -> Iterator[dict[str, Any]]:
    for row in jsonl_rows(path):
        if row.get("data_source") != "understand_user":
            continue
        required = {"instruction", "output", "data_source", "source_segment"}
        if not required.issubset(row):
            raise ValueError(f"BETA user row is missing {sorted(required - set(row))}")
        yield row


def write_row(writer, digest: hashlib._Hash, row: dict[str, Any]) -> None:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    writer.write(payload)
    digest.update(payload.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-parquet", type=Path, required=True)
    parser.add_argument("--beta-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_file = output_dir / "onereason_bata_baseline.jsonl"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    archive_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    recommendation_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in archive_rows(args.archive_parquet):
        archive_counts[row["data_source"]] += 1
        source_counts[row["source"]] += 1
        if row["data_source"] == "recommend":
            key, current = recommendation_identity(row)
            if current not in recommendation_groups[key]:
                recommendation_groups[key].append(current)
    if dict(archive_counts) != EXPECTED_ARCHIVE_COUNTS:
        raise ValueError(f"Archive counts differ: {dict(archive_counts)}")

    beta_users = iter(beta_user_rows(args.beta_dataset))
    output_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    dataset_digest = hashlib.sha256()
    archive_non_user_source_digest = hashlib.sha256()
    archive_non_user_output_digest = hashlib.sha256()
    beta_user_digest = hashlib.sha256()
    output_user_digest = hashlib.sha256()
    replaced_users = 0
    appended_users = 0

    with output_file.open("x", encoding="utf-8") as writer:
        for archive_row in archive_rows(args.archive_parquet):
            if archive_row["data_source"] == "understand_user":
                try:
                    converted = next(beta_users)
                except StopIteration as error:
                    raise ValueError("BETA has fewer user rows than the archive") from error
                encoded = (json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                beta_user_digest.update(encoded)
                output_user_digest.update(encoded)
                replaced_users += 1
            else:
                converted = convert_archive(archive_row, recommendation_groups)
                source_projection = message_projection(
                    archive_row["messages"], archive_row["source"], archive_row["data_source"]
                )
                archive_non_user_source_digest.update(source_projection)
                converted_projection = output_projection(converted, archive_row["source"])
                archive_non_user_output_digest.update(converted_projection)
                if source_projection != converted_projection:
                    raise AssertionError("Archive message projection changed during conversion")
                if archive_row["data_source"] == "material_sample":
                    matches = SID_RE.findall(archive_row["messages"][1]["content"] + archive_row["messages"][2]["content"])
                    if not matches:
                        raise ValueError("material_sample has no domain marker")
                    domain_counts[matches[0]] += 1

            output_counts[converted["data_source"]] += 1
            segment_counts[converted["source_segment"]] += 1
            write_row(writer, dataset_digest, converted)

        for converted in beta_users:
            encoded = (json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            beta_user_digest.update(encoded)
            output_user_digest.update(encoded)
            output_counts[converted["data_source"]] += 1
            segment_counts[converted["source_segment"]] += 1
            appended_users += 1
            write_row(writer, dataset_digest, converted)

    if dict(output_counts) != EXPECTED_OUTPUT_COUNTS:
        raise ValueError(f"Output counts differ: {dict(output_counts)}")
    if replaced_users != EXPECTED_ARCHIVE_COUNTS["understand_user"] or appended_users != 396:
        raise ValueError(f"Unexpected user replacement: replaced={replaced_users}, appended={appended_users}")
    if dict(domain_counts) != EXPECTED_MATERIAL_DOMAINS:
        raise ValueError(f"Material domain counts differ: {dict(domain_counts)}")
    if archive_non_user_source_digest.digest() != archive_non_user_output_digest.digest():
        raise AssertionError("Archive non-user projection digest changed")

    gold_size_histogram = Counter(len(golds) for golds in recommendation_groups.values())
    recommendation_rows_by_group_kind = {"singleton": 0, "multi_positive": 0}
    for row in archive_rows(args.archive_parquet):
        if row["data_source"] != "recommend":
            continue
        key, _ = recommendation_identity(row)
        kind = "singleton" if len(recommendation_groups[key]) == 1 else "multi_positive"
        recommendation_rows_by_group_kind[kind] += 1

    dataset_info = {
        "onereason_bata_baseline": {
            "file_name": output_file.name,
            "formatting": "alpaca",
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
        "name": "bata_baseline_v1",
        "experiment_name": "pure bata_baseline",
        "records": sum(output_counts.values()),
        "sha256": dataset_digest.hexdigest(),
        "archive_parquet": str(args.archive_parquet.resolve()),
        "archive_parquet_sha256": sha256_file(args.archive_parquet),
        "archive_counts": dict(archive_counts),
        "archive_source_counts": dict(source_counts),
        "archive_non_user_projection_sha256": archive_non_user_source_digest.hexdigest(),
        "archive_non_user_projection_preserved": True,
        "intentional_difference": "understand_user is replaced by active BETA understand_user",
        "beta_dataset": str(args.beta_dataset.resolve()),
        "beta_dataset_sha256": sha256_file(args.beta_dataset),
        "beta_user_projection_sha256": beta_user_digest.hexdigest(),
        "beta_user_projection_preserved": beta_user_digest.digest() == output_user_digest.digest(),
        "user_rows_replaced_in_archive_positions": replaced_users,
        "user_rows_appended": appended_users,
        "source_counts": dict(output_counts),
        "source_segment_counts": dict(segment_counts),
        "material_domain_counts": dict(domain_counts),
        "material_domain_weights": MATERIAL_DOMAIN_WEIGHTS,
        "loss_routes": {
            "material_sample": "valid-token CE; SID/domain-marker tokens x8; sample domain multiplier",
            "sid_bucket_canonical_no_think": "all valid response tokens x4",
            "sid_bucket_reverse": "valid-token CE; SID/domain-marker tokens x8",
            "understand_user": "valid-token CE; SID/domain-marker tokens x8",
            "recommend": "ordinary one-hot valid-token CE; SID/domain-marker tokens x8",
        },
        "recommendation_metadata": {
            "group_key": "exact user prompt without terminal think marker + target domain",
            "groups": len(recommendation_groups),
            "gold_count_histogram": {str(k): v for k, v in sorted(gold_size_histogram.items())},
            "rows": recommendation_rows_by_group_kind,
            "maximum_unique_gold_count": max(gold_size_histogram),
            "training_usage": "interface retained; REC-PU disabled for bata_baseline",
        },
        "world_included": False,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Pure bata_baseline\n\n"
        "The reproduction archive is preserved exactly except for understand_user, which uses active BETA. "
        "Recommendation multi-positive metadata is retained as a dormant interface; formal baseline training uses ordinary SID8 CE.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
