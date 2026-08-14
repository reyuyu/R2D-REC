#!/usr/bin/env python3
"""CPU/Gloo audit of the exact Accelerate BatchSamplerShard ownership contract."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import torch.distributed as dist
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard

BASELINE_ROOT = Path("/data/baselines/native_source_domain_r32_v3")
sys.path.insert(0, str(BASELINE_ROOT))
from pack_ratio_sampler import PackRatioSampler, TASK_NAMES  # noqa: E402


PACK_TASK_IDS = [0] * 3242 + [1] * 8072 + [2] * 9443 + [3] * 12859
RATIOS = (0.20, 0.45, 0.20, 0.15)


def main() -> None:
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"Expected four ranks, got {world}")
    sampler = PackRatioSampler(PACK_TASK_IDS, RATIOS, seed=20260806)
    sampler.set_epoch(0)
    batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
    shard = BatchSamplerShard(
        batch_sampler,
        num_processes=world,
        process_index=rank,
        split_batches=False,
        even_batches=True,
    )
    local = [int(batch[0]) for _, batch in zip(range(len(shard)), shard)]
    if len(local) != 8404:
        raise RuntimeError(f"rank {rank} length mismatch: {len(local)}")
    gathered: list[list[int] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        by_rank = [list(value or []) for value in gathered]
        merged = [by_rank[current_rank][offset] for offset in range(8404) for current_rank in range(world)]
        expected, _ = sampler.build_plan()
        if merged != expected:
            raise RuntimeError("BatchSamplerShard merge does not equal the PackRatio global plan.")
        counts = Counter(PACK_TASK_IDS[index] for index in merged)
        result = {
            "rank_lengths": [len(value) for value in by_rank],
            "merged_length": len(merged),
            "counts": {TASK_NAMES[key]: counts[key] for key in range(4)},
            "first_three_windows": [
                {TASK_NAMES[key]: Counter(PACK_TASK_IDS[index] for index in merged[start : start + 64])[key] for key in range(4)}
                for start in range(0, 192, 64)
            ],
        }
        print("PACK_RATIO_BATCHSAMPLER_SHARD=" + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
