#!/usr/bin/env python3
"""Compose task-pool alpha_mini inputs into a train-ready alpha_mini_v1 dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "/data/models/onereason-8b-pretrain-competition"
SCHEMA = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}


def rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != SCHEMA:
                raise ValueError(f"Unexpected schema at {path}:{line_no}: {sorted(row)}")
            if row["data_source"] == "recommend":
                metadata = json.loads(row["aux_metadata_json"])
                required = {
                    "recommendation_group_id", "recommendation_group_size",
                    "recommendation_current_gold_sid", "recommendation_all_gold_sids",
                }
                if set(metadata) != required:
                    raise ValueError(f"Incomplete recommendation metadata at {path}:{line_no}")
            yield row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_stats(path: Path, tokenizer) -> dict:
    prompt_total = output_total = rows_count = max_tokens = 0
    for row in rows(path):
        prompt = "".join(str(row[key]) for key in ("system", "instruction", "input"))
        prompt_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        output_count = len(tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"])
        prompt_total += prompt_count
        output_total += output_count
        rows_count += 1
        max_tokens = max(max_tokens, prompt_count + output_count)
    return {
        "rows": rows_count,
        "prompt_tokens": prompt_total,
        "output_tokens": output_total,
        "total_tokens": prompt_total + output_total,
        "mean_tokens_per_row": round((prompt_total + output_total) / rows_count, 4),
        "max_tokens_per_row": max_tokens,
    }


def main() -> None:
    root = Path("/data/lf_data_versions/task_pools")
    inputs = [
        ("understand_user", root / "\u61c2\u7528\u6237" / "alpha_mini" / "understand_user_alpha_mini.jsonl"),
        ("recommendation", root / "\u61c2\u63a8\u8350" / "alpha_mini" / "recommendation_multipositive_video_top550_other_domains_all.jsonl"),
        ("material", root / "\u61c2\u7269\u6599" / "alpha_mini" / "material_alpha_mini.jsonl"),
    ]
    expected_rows = {"understand_user": 2000, "recommendation": 11192, "material": 36298}
    output_dir = Path("/data/lf_data_versions/alltrain/alpha_mini_v1")
    output_file = output_dir / "onereason_alpha_mini_v1.jsonl"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    for name, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    input_counts, data_sources, source_segments = Counter(), Counter(), Counter()
    with output_file.open("x", encoding="utf-8") as output:
        for input_name, path in inputs:
            count = 0
            for row in rows(path):
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                output.write(payload)
                digest.update(payload.encode("utf-8"))
                count += 1
                data_sources[row["data_source"]] += 1
                source_segments[row["source_segment"]] += 1
            if count != expected_rows[input_name]:
                raise ValueError(f"Unexpected {input_name} rows: {count}")
            input_counts[input_name] = count
    if sum(input_counts.values()) != 49490:
        raise AssertionError("Unexpected total row count")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    stats = token_stats(output_file, tokenizer)
    input_hashes = {name: sha256(path) for name, path in inputs}
    dataset_info = {
        "onereason_alpha_mini_v1": {
            "file_name": output_file.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction", "query": "input", "response": "output",
                "history": "history", "system": "system",
            },
        }
    }
    manifest = {
        "kind": "train_ready_composed_dataset",
        "name": "alpha_mini_v1",
        "dataset_key": "onereason_alpha_mini_v1",
        "input_task_pools": {name: str(path) for name, path in inputs},
        "input_sha256": input_hashes,
        "composition_order": [name for name, _ in inputs],
        "composition_rule": "byte-preserving row composition; no row mutation, deduplication, shuffle, or resampling",
        "schema": sorted(SCHEMA),
        "counts": {
            "by_input_task_pool": dict(input_counts),
            "by_data_source": dict(sorted(data_sources.items())),
            "by_source_segment": dict(sorted(source_segments.items())),
            "total_rows": sum(input_counts.values()),
        },
        "recommendation_metadata": {
            "preserved": True,
            "required_fields": [
                "recommendation_group_id", "recommendation_group_size",
                "recommendation_current_gold_sid", "recommendation_all_gold_sids",
            ],
        },
        "tokens": {
            "model": MODEL,
            "method": "AutoTokenizer add_special_tokens=False; prompt=system+instruction+input; total=prompt+output; excludes chat-template fixed overhead",
            **stats,
        },
        "file": output_file.name,
        "sha256": digest.hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f'''# alpha_mini_v1

训练就绪的 alpha_mini 组合数据集。

## 组成

| 任务池 | 行数 |
|---|---:|
| 懂用户 alpha_mini | {input_counts["understand_user"]:,} |
| 懂推荐 alpha_mini | {input_counts["recommendation"]:,} |
| 懂物料 alpha_mini | {input_counts["material"]:,} |
| 合计 | **{sum(input_counts.values()):,}** |

组合过程逐行保留原始训练字段和顺序：懂推荐的多正样本 `aux_metadata_json` 完整保留；不额外清洗、去重或重采样。运行时仍应按训练配置进行 shuffle / packing。

## 使用

- 训练 JSONL：`{output_file.name}`
- LLaMAFactory 数据集 key：`onereason_alpha_mini_v1`
- 字段映射与可追溯统计：`dataset_info.json`、`manifest.json`

总原始 token（不含 chat-template 固定开销）：{stats["total_tokens"]:,}。
'''
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
