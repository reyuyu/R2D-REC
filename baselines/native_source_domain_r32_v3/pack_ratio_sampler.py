"""Deterministic global packed-row sampling for the PackRatio ablation.

The sampler deliberately knows nothing about ranks or DataLoader workers.
Accelerate owns distributed sharding after this sampler yields one global plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Iterable, Sequence

from torch.utils.data import Sampler


TASK_NAMES = ("material", "recommendation", "user_action", "user_chain")
TASK_COUNT = len(TASK_NAMES)


@dataclass(frozen=True)
class PackRatioConfig:
    enabled: bool
    target_ratios: tuple[float, float, float, float]


def parse_target_ratios(raw: str) -> tuple[float, float, float, float]:
    """Parse and normalize the strict flat YAML representation."""

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("`multitask_pack_ratio_targets` must be a non-empty string.")

    values: dict[str, float] = {}
    for item in raw.split(","):
        key, separator, value = item.strip().partition("=")
        if not separator or not key or not value or key in values:
            raise ValueError("Invalid `multitask_pack_ratio_targets` entry: {!r}.".format(item))
        if key not in TASK_NAMES:
            raise ValueError("Unknown pack-ratio task: {!r}.".format(key))
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError("Invalid ratio for {!r}: {!r}.".format(key, value)) from error
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("Pack ratios must be finite and > 0.")
        values[key] = parsed

    if set(values) != set(TASK_NAMES):
        missing = sorted(set(TASK_NAMES) - set(values))
        extra = sorted(set(values) - set(TASK_NAMES))
        raise ValueError("Pack ratios must contain exactly four tasks; missing={}, extra={}.".format(missing, extra))

    total = sum(values.values())
    if not math.isfinite(total) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Pack ratios must sum to 1.0 (got {:.17g}).".format(total))

    return tuple(values[name] / total for name in TASK_NAMES)


def target_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int, int]:
    """Allocate exactly ``total`` slots using deterministic largest remainder."""

    if total < 0 or len(ratios) != TASK_COUNT:
        raise ValueError("Expected a non-negative total and four ratios.")

    raw = [total * float(ratio) for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(TASK_COUNT), key=lambda task_id: (-(raw[task_id] - counts[task_id]), task_id))
    for task_id in order[:remaining]:
        counts[task_id] += 1

    result = tuple(int(value) for value in counts)
    if sum(result) != total:
        raise AssertionError("Largest-remainder allocation did not preserve total length.")
    return result  # type: ignore[return-value]


def derive_pack_task_id(labels: Sequence[int], sample_task_ids: Sequence[int], ignore_index: int) -> int:
    """Return the primary task using the same causal shift as native SID8 loss."""

    if len(labels) != len(sample_task_ids):
        raise ValueError("Packed labels and sample_task_ids must have equal length.")

    counts = [0] * TASK_COUNT
    for label, task_id in zip(labels[1:], sample_task_ids[1:]):
        if label != ignore_index and 0 <= task_id < TASK_COUNT:
            counts[task_id] += 1

    # ``min`` is the required deterministic tie-break in task-id order.
    return min(range(TASK_COUNT), key=lambda task_id: (-counts[task_id], task_id))


def _cycle_seed(base_seed: int, task_id: int, cycle_id: int) -> int:
    """Stable task-isolated seed; never use Python's process-salted hash()."""

    return (
        (int(base_seed) & ((1 << 63) - 1))
        ^ ((task_id + 1) * 0x517CC1B727220A95)
        ^ ((cycle_id + 1) * 0x6C8E9CF570932BD5)
    ) & ((1 << 63) - 1)


def _cycle_permutation(pool: Sequence[int], base_seed: int, task_id: int, cycle_id: int) -> list[int]:
    result = list(pool)
    random.Random(_cycle_seed(base_seed, task_id, cycle_id)).shuffle(result)
    return result


def task_stream_window(
    pool: Sequence[int], *, base_seed: int, task_id: int, start: int, count: int
) -> list[int]:
    """Return a deterministic slice of an infinite sequence of shuffled pool cycles."""

    if not pool and count:
        raise ValueError("Cannot allocate positive quota to an empty task pool.")
    if start < 0 or count < 0:
        raise ValueError("Task stream start/count must be non-negative.")

    result: list[int] = []
    pool_size = len(pool)
    position = start
    while len(result) < count:
        cycle_id, offset = divmod(position, pool_size)
        cycle = _cycle_permutation(pool, base_seed, task_id, cycle_id)
        take = min(count - len(result), pool_size - offset)
        result.extend(cycle[offset : offset + take])
        position += take
    return result


class PackRatioSampler(Sampler[int]):
    """CPU-only global index sampler with exact task quotas per epoch."""

    def __init__(self, pack_task_ids: Iterable[int], target_ratios: Sequence[float], seed: int) -> None:
        self.pack_task_ids = tuple(int(task_id) for task_id in pack_task_ids)
        self.target_ratios = tuple(float(value) for value in target_ratios)
        if len(self.target_ratios) != TASK_COUNT or any(value <= 0.0 for value in self.target_ratios):
            raise ValueError("PackRatioSampler requires four positive normalized ratios.")
        normalized = sum(self.target_ratios)
        if not math.isclose(normalized, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("PackRatioSampler ratios must sum to 1.")
        if any(task_id < 0 or task_id >= TASK_COUNT for task_id in self.pack_task_ids):
            raise ValueError("pack_task_ids contains an unknown task id.")

        self.seed = int(seed)
        self.epoch = 0
        self.pools = tuple(
            tuple(index for index, task_id in enumerate(self.pack_task_ids) if task_id == expected)
            for expected in range(TASK_COUNT)
        )
        self.quotas = target_counts(len(self.pack_task_ids), self.target_ratios)
        for task_id, quota in enumerate(self.quotas):
            if quota and not self.pools[task_id]:
                raise ValueError("Target quota for {!r} is positive but its pool is empty.".format(TASK_NAMES[task_id]))

    def __len__(self) -> int:
        return len(self.pack_task_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def fingerprint(self) -> str:
        payload = {
            "dataset_length": len(self),
            "epoch": self.epoch,
            "pool_counts": [len(pool) for pool in self.pools],
            "ratios": self.target_ratios,
            "seed": self.seed,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

    def build_plan(self) -> tuple[list[int], list[int]]:
        """Return global dataset indices and their primary task IDs for one epoch."""

        streams = [
            task_stream_window(
                pool,
                base_seed=self.seed,
                task_id=task_id,
                start=self.epoch * self.quotas[task_id],
                count=self.quotas[task_id],
            )
            for task_id, pool in enumerate(self.pools)
        ]
        used = [0] * TASK_COUNT
        plan: list[int] = []
        plan_tasks: list[int] = []
        total = len(self)
        for slot in range(total):
            candidates = [task_id for task_id in range(TASK_COUNT) if used[task_id] < self.quotas[task_id]]
            chosen = min(
                candidates,
                key=lambda task_id: (
                    -(self.target_ratios[task_id] * (slot + 1) - used[task_id]),
                    task_id,
                ),
            )
            plan.append(streams[chosen][used[chosen]])
            plan_tasks.append(chosen)
            used[chosen] += 1

        if len(plan) != total or tuple(used) != self.quotas:
            raise AssertionError("Pack-ratio plan does not match its exact epoch quotas.")
        if Counter(plan_tasks) != Counter({task_id: self.quotas[task_id] for task_id in range(TASK_COUNT)}):
            raise AssertionError("Pack-ratio task accounting mismatch.")
        return plan, plan_tasks

    def __iter__(self):
        plan, _ = self.build_plan()
        return iter(plan)
