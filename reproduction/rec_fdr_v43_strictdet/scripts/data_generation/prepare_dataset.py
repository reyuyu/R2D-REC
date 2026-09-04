#!/usr/bin/env python3
"""Convert the competition Parquet shards into LLaMA-Factory ShareGPT Parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("/data/LLm-8B/data/SecondRoundDataWithCaptionNoInternalPaths")
DEFAULT_OUTPUT = Path("/data/LLm-8B/code/train/data/processed")
DEFAULT_MARKER = Path("/data/LLm-8B/code/train/data/.processed_success.json")

MESSAGE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
        ]
    )
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("messages", MESSAGE_TYPE, nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
    ]
)
VALID_ROLES = {"system", "user", "assistant"}


def flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"unsupported content type: {type(content).__name__}")

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
            raise ValueError(f"unsupported content item: {item!r}")
        parts.append(item["text"])
    return "".join(parts)


def normalize_messages(raw: object) -> list[dict[str, str]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty JSON list")

    result: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"unsupported role: {role!r}")
        result.append({"role": role, "content": flatten_content(message.get("content"))})

    dialog = result[1:] if result[0]["role"] == "system" else result
    if len(dialog) < 2 or len(dialog) % 2 != 0:
        raise ValueError(f"invalid non-system message count: {len(dialog)}")
    for index, message in enumerate(dialog):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"expected role {expected!r}, got {message['role']!r}")
    return result


def output_name(input_root: Path, source: Path) -> str:
    relative = source.relative_to(input_root)
    return "__".join(relative.with_suffix("").parts) + ".parquet"


def convert_file(input_root: Path, source: Path, destination: Path) -> int:
    parquet = pq.ParquetFile(source)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temp, OUTPUT_SCHEMA, compression="zstd")
    total = 0
    try:
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["messages"])
            normalized = [normalize_messages(value) for value in table.column("messages").to_pylist()]
            count = len(normalized)
            output = pa.Table.from_arrays(
                [
                    pa.array(normalized, type=MESSAGE_TYPE),
                    pa.array([source.relative_to(input_root).parts[0]] * count, type=pa.string()),
                    pa.array([str(source.relative_to(input_root))] * count, type=pa.string()),
                ],
                schema=OUTPUT_SCHEMA,
            )
            writer.write_table(output)
            total += count
    except Exception:
        writer.close()
        temp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temp, destination)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = sorted(args.input.rglob("*.parquet"))
    if not sources:
        raise SystemExit(f"No Parquet files found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    converted = 0
    reused = 0
    for index, source in enumerate(sources, start=1):
        destination = args.output / output_name(args.input, source)
        expected_rows = pq.ParquetFile(source).metadata.num_rows
        if destination.exists() and not args.overwrite:
            actual_rows = pq.ParquetFile(destination).metadata.num_rows
            if actual_rows == expected_rows:
                total_rows += actual_rows
                reused += 1
                continue
        rows = convert_file(args.input, source, destination)
        total_rows += rows
        converted += 1
        print(f"[{index}/{len(sources)}] {source.relative_to(args.input)}: {rows} rows", flush=True)

    marker = {
        "input": str(args.input),
        "output": str(args.output),
        "source_files": len(sources),
        "converted_files": converted,
        "reused_files": reused,
        "rows": total_rows,
    }
    args.marker.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
