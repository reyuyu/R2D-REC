#!/usr/bin/env python3
"""Offline BFD pack-length benchmark for recC/recE candidates."""
from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.ipc as ipc

GROUPS = {
    "material/cot": "01297b747a727d3c",
    "material/nocot": "880cdb3a61bebc2f",
    "user_action/action_nocot": "a959f363a9f7b16a",
    "user_chain/cot": "5384be5e29f7c40b",
    "user_chain/nocot": "e3d43df2581f6077",
    "recommendation/cot": "8f50eb78057fee7f",
    "recommendation/nocot": "1a0cd6ef9586867e",
}
CANDIDATES = {
    "8k": {},
    "12k_user": {
        "user_action/action_nocot": 12288,
        "user_chain/cot": 12288,
        "user_chain/nocot": 12288,
    },
    "16k_user": {
        "user_action/action_nocot": 16384,
        "user_chain/cot": 16384,
        "user_chain/nocot": 16384,
    },
}


def lengths(cache_root: Path, prefix: str) -> list[int]:
    result: list[int] = []
    for path in sorted(cache_root.rglob(f"cache-{prefix}_*.arrow")):
        table = ipc.open_stream(str(path)).read_all()
        result.extend(pc.list_value_length(table["input_ids"]).to_pylist())
    if not result:
        raise FileNotFoundError(f"No Arrow cache found for prefix {prefix} under {cache_root}")
    return [int(value) for value in result]


def bfd(values: list[int], max_length: int, window: int, seed: int = 42) -> list[list[int]]:
    indexes = list(range(len(values)))
    rng = random.Random(seed)
    rng.shuffle(indexes)
    plans: list[list[int]] = []
    for start in range(0, len(indexes), window):
        current = indexes[start : start + window]
        current.sort(key=lambda index: (-min(values[index], max_length), index))
        packs: list[list[int]] = []
        used: list[int] = []
        for index in current:
            length = min(values[index], max_length)
            choices = [pos for pos, total in enumerate(used) if total + length <= max_length]
            if choices:
                pos = min(choices, key=lambda value: (max_length - used[value] - length, value))
                packs[pos].append(index)
                used[pos] += length
            else:
                packs.append([index])
                used.append(length)
        plans.extend(packs)
    rng.shuffle(plans)
    return plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("/root/.cache/huggingface/datasets/json"))
    parser.add_argument("--window", type=int, default=4096)
    args = parser.parse_args()
    all_lengths = {name: lengths(args.cache_root, prefix) for name, prefix in GROUPS.items()}

    for candidate, overrides in CANDIDATES.items():
        print(f"=== {candidate} ===")
        total_packs = 0
        for name, values in all_lengths.items():
            max_length = overrides.get(name, 8192)
            plans = bfd(values, max_length, args.window)
            token_counts = [sum(min(values[index], max_length) for index in pack) for pack in plans]
            segments = [len(pack) for pack in plans]
            utilization = [tokens / max_length for tokens in token_counts]
            total_packs += len(plans)
            print(
                name,
                f"max_pack_length={max_length}",
                f"pack_count={len(plans)}",
                f"mean_pack_tokens={statistics.mean(token_counts):.2f}",
                f"pack_utilization={statistics.mean(utilization):.4f}",
                f"single_sample_pack_ratio={sum(size == 1 for size in segments) / len(segments):.4f}",
                f"segments_per_pack={statistics.mean(segments):.3f}",
            )
        print(f"total_pack_count={total_packs}")
        print(f"estimated_macro_steps_per_pack_epoch={math.ceil(total_packs / 8)}")


if __name__ == "__main__":
    main()
