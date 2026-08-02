#!/usr/bin/env python3
"""Exact pre-truncation length audit for the OneReason SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments


FILES = {
    "material_cot": "onereason_material_cot_train98.jsonl",
    "material_nocot": "onereason_material_nocot_train98.jsonl",
    "user_action": "onereason_user_action_nocot_train98.jsonl",
    "user_chain_cot": "onereason_user_chain_cot_train98.jsonl",
    "user_chain_nocot": "onereason_user_chain_nocot_train98.jsonl",
    "recommendation": "onereason_recommendation_cot_train98.jsonl",
    "world_cot": "onereason_world_cot_train98.jsonl",
    "world_nocot": "onereason_world_nocot_train98.jsonl",
}


def make_messages(row: dict) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in row.get("history") or []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.extend(({"role": "user", "content": str(turn[0])}, {"role": "assistant", "content": str(turn[1])}))
    query = "\n".join(str(value) for value in (row.get("instruction", ""), row.get("input", "")) if value)
    messages.extend(({"role": "user", "content": query}, {"role": "assistant", "content": str(row.get("output", ""))}))
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/lf_data_splits"))
    parser.add_argument("--model", type=Path, default=Path("/data/models/onereason-8b-pretrain-competition"))
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=args.template))
    eos_length = int(template.efficient_eos)
    result = {"thresholds": [8192, 16384], "datasets": {}, "total": {"samples": 0, "over_8192": 0, "over_16384": 0}}
    started = time.time()

    for name, filename in FILES.items():
        stats = {"samples": 0, "over_8192": 0, "over_16384": 0, "max_length": 0}
        with (args.data_dir / filename).open(encoding="utf-8") as reader:
            for line in reader:
                row = json.loads(line)
                prompt_ids, response_ids = template.encode_oneturn(tokenizer, make_messages(row))
                length = len(prompt_ids) + len(response_ids) + eos_length
                stats["samples"] += 1
                stats["over_8192"] += length > 8192
                stats["over_16384"] += length > 16384
                stats["max_length"] = max(stats["max_length"], length)
                if stats["samples"] % args.progress_every == 0:
                    print(f"{name}: {stats['samples']} scanned, >16k={stats['over_16384']}", flush=True)
        result["datasets"][name] = stats
        for key in result["total"]:
            if key != "samples":
                result["total"][key] += stats[key]
        result["total"]["samples"] += stats["samples"]
        print(json.dumps({name: stats}, ensure_ascii=False), flush=True)

    result["elapsed_seconds"] = round(time.time() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
