"""Optional task-aware SFT batching used by ``multitask_macro_training``.

The module deliberately keeps task scheduling and packing separate from loss
composition.  It is therefore safe to use as the data-side foundation for
future GradNorm/PCGrad work without introducing either algorithm here.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import torch
from torch.utils.data import Dataset

from ..extras.constants import IGNORE_INDEX
from .action_select import ActionSelectMetadataParser, offset_action_metadata
from .loader import get_dataset
from .onereason_dataset_versions import resolve_managed_dataset_registry_name

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..data.template import Template
    from ..hparams import DataArguments, ModelArguments
    from ..hparams.training_args import TrainingArguments


TASK_DATASETS: dict[str, dict[str, str]] = {
    "material": {"cot": "onereason_material_cot", "nocot": "onereason_material_nocot"},
    "user": {
        "action_nocot": "onereason_user_action_nocot",
        "chain_cot": "onereason_user_chain_cot",
        "chain_nocot": "onereason_user_chain_nocot",
    },
    "recommendation": {"cot": "onereason_recommendation_cot"},
    "world": {"cot": "onereason_world_cot", "nocot": "onereason_world_nocot"},
}
TASK_IDS = {"material": 0, "user": 1, "recommendation": 2, "world": 3}
SUBTASK_RATIOS = {
    "material": {"cot": 0.50, "nocot": 0.50},
    "user": {"action_nocot": 0.50, "chain_cot": 0.25, "chain_nocot": 0.25},
    "recommendation": {"cot": 1.00},
    "world": {"cot": 0.10, "nocot": 0.90},
}
LENGTH_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
PACKING_MODE_LENGTH_BUCKET_GREEDY = "length_bucket_greedy"
PACKING_MODE_SAME_SUBTASK_BFD = "same_subtask_bfd"

TASK_LAYOUT_LEGACY = "legacy"
TASK_LAYOUT_USER_SPLIT_NO_WORLD = "user_split_no_world"
TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL = "user_split_no_world_rec_dual"
TASK_LAYOUTS = (TASK_LAYOUT_LEGACY, TASK_LAYOUT_USER_SPLIT_NO_WORLD, TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL)

# Experiment E ablation layout: split user into user_action / user_chain and
# drop the world task entirely. The legacy maps above remain the module-level
# defaults so every existing caller and test keeps its current semantics when
# the new option is not enabled.
TASK_DATASETS_BY_LAYOUT: dict[str, dict[str, dict[str, str]]] = {
    TASK_LAYOUT_LEGACY: TASK_DATASETS,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD: {
        "material": {"cot": "onereason_material_cot", "nocot": "onereason_material_nocot"},
        "user_action": {"action_nocot": "onereason_user_action_nocot"},
        "user_chain": {"cot": "onereason_user_chain_cot", "nocot": "onereason_user_chain_nocot"},
        "recommendation": {"cot": "onereason_recommendation_cot"},
    },
    TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL: {
        "material": {"cot": "onereason_material_cot", "nocot": "onereason_material_nocot"},
        "user_action": {"action_nocot": "onereason_user_action_nocot"},
        "user_chain": {"cot": "onereason_user_chain_cot", "nocot": "onereason_user_chain_nocot"},
        "recommendation": {"cot": "onereason_recommendation_cot", "nocot": "onereason_recommendation_nocot"},
    },
}
TASK_IDS_BY_LAYOUT: dict[str, dict[str, int]] = {
    TASK_LAYOUT_LEGACY: TASK_IDS,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD: {"material": 0, "user_action": 1, "user_chain": 2, "recommendation": 3},
    TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL: {"material": 0, "user_action": 1, "user_chain": 2, "recommendation": 3},
}
SUBTASK_RATIOS_BY_LAYOUT: dict[str, dict[str, dict[str, float]]] = {
    TASK_LAYOUT_LEGACY: SUBTASK_RATIOS,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD: {
        "material": {"cot": 0.50, "nocot": 0.50},
        "user_action": {"action_nocot": 1.00},
        "user_chain": {"cot": 0.50, "nocot": 0.50},
        "recommendation": {"cot": 1.00},
    },
    TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL: {
        "material": {"cot": 0.50, "nocot": 0.50},
        "user_action": {"action_nocot": 1.00},
        "user_chain": {"cot": 0.50, "nocot": 0.50},
        "recommendation": {"cot": 0.50, "nocot": 0.50},
    },
}


def resolve_task_layout(layout: str) -> str:
    if layout not in TASK_LAYOUTS:
        raise ValueError(f"Unknown multitask_task_layout {layout!r}; expected one of {TASK_LAYOUTS}.")
    return layout


def get_task_ids(layout: str = TASK_LAYOUT_LEGACY) -> dict[str, int]:
    return dict(TASK_IDS_BY_LAYOUT[resolve_task_layout(layout)])


def get_task_datasets(layout: str = TASK_LAYOUT_LEGACY) -> dict[str, dict[str, str]]:
    return TASK_DATASETS_BY_LAYOUT[resolve_task_layout(layout)]


def get_subtask_ratios(layout: str = TASK_LAYOUT_LEGACY) -> dict[str, dict[str, float]]:
    return SUBTASK_RATIOS_BY_LAYOUT[resolve_task_layout(layout)]


def get_action_task_name(layout: str = TASK_LAYOUT_LEGACY) -> str:
    # Return the task that owns the Action Select subtask for a layout.
    task_ids = get_task_ids(layout)
    return "user_action" if "user_action" in task_ids else "user"


def normalize_subtask_pack_lengths(
    configured: Mapping[str, int] | None,
    layout: str,
    default_length: int,
) -> dict[str, int]:
    """Resolve canonical and human-friendly subtask pack-length keys."""
    valid = {
        f"{task}/{subtask}"
        for task, subtasks in get_task_datasets(layout).items()
        for subtask in subtasks
    }
    aliases: dict[str, str] = {}
    for key in valid:
        task, subtask = key.split("/", 1)
        aliases[f"{task}/{subtask}"] = key
        if task == "user_action" and subtask == "action_nocot":
            aliases["user/action"] = key
            aliases["user/action_nocot"] = key
        elif task == "user_chain":
            aliases[f"user/chain_{subtask}"] = key
    result = {key: int(default_length) for key in valid}
    for raw_key, raw_value in (configured or {}).items():
        key = aliases.get(str(raw_key))
        if key is None:
            raise ValueError(
                f"Unknown multitask_pack_length_by_subtask key {raw_key!r}; expected one of {sorted(valid)}"
            )
        value = int(raw_value)
        if value <= 0:
            raise ValueError("multitask_pack_length_by_subtask values must be positive.")
        result[key] = value
    return result


def resolve_multitask_dataset_name(base_dataset_name: str, data_args) -> str:
    """Resolve a logical subdataset to a versioned registry name without changing task identity."""
    layout = getattr(data_args, "multitask_task_layout", TASK_LAYOUT_LEGACY)
    valid_names = {name for subtasks in get_task_datasets(layout).values() for name in subtasks.values()}
    overrides = data_args.multitask_dataset_version_overrides
    unknown_overrides = set(overrides) - valid_names
    if unknown_overrides:
        raise ValueError(f"Unknown multitask dataset override(s): {sorted(unknown_overrides)}")

    version = overrides.get(base_dataset_name, data_args.multitask_dataset_version)
    if version == "raw":
        return base_dataset_name + data_args.multitask_train_dataset_suffix
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError(f"Invalid multitask dataset version: {version!r}")
    managed_name = resolve_managed_dataset_registry_name(
        base_dataset_name,
        version,
        getattr(data_args, "multitask_dataset_version_manifest", "data/onereason_dataset_versions.json"),
    )
    if managed_name is not None:
        return managed_name
    # Legacy versions predate the inheritance manifest and retain their old
    # alias convention so historical train98 experiments stay reproducible.
    return f"{base_dataset_name}_{version}{data_args.multitask_train_dataset_suffix}"


class TokenizedSubDataset(Dataset):
    """A tokenized dataset plus stable task/subtask provenance."""

    def __init__(
        self,
        dataset: Dataset,
        task_name: str,
        subtask_name: str,
        subtask_id: int,
        metadata_parser: ActionSelectMetadataParser | None = None,
        task_ids: Mapping[str, int] | None = None,
    ):
        self.dataset = dataset
        self.task_name = task_name
        self.task_id = (task_ids or TASK_IDS)[task_name]
        self.subtask_name = subtask_name
        self.subtask_id = subtask_id
        self.metadata_parser = metadata_parser
        self._metadata_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        labels = item["labels"]
        recommendation_metadata = item.pop("recommendation_metadata", None)
        if isinstance(recommendation_metadata, str):
            try:
                recommendation_metadata = json.loads(recommendation_metadata) if recommendation_metadata else None
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid recommendation metadata JSON at index {index}.") from exc
        elif isinstance(recommendation_metadata, list):
            # Packed processor caches may expose one JSON value per segment.
            # The multitask loader disables native packing, but accepting a
            # singleton list keeps old caches backward compatible.
            if len(recommendation_metadata) == 1:
                recommendation_metadata = recommendation_metadata[0]
                if isinstance(recommendation_metadata, str):
                    try:
                        recommendation_metadata = json.loads(recommendation_metadata) if recommendation_metadata else None
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid recommendation metadata JSON at index {index}.") from exc
        sample_metadata = dict(item.pop("sample_metadata", {}) or {})
        # Some pre-V3 tokenized caches kept the four fields in a flat metadata
        # struct. Prefer a valid direct record over a null-valued legacy struct.
        if not isinstance(recommendation_metadata, dict):
            flat = {key: sample_metadata.get(key) for key in (
                "recommendation_group_id", "recommendation_group_size",
                "recommendation_all_gold_sids", "recommendation_current_gold_sid",
            )}
            if flat["recommendation_group_id"] is not None and flat["recommendation_all_gold_sids"] is not None:
                recommendation_metadata = flat
        if self.metadata_parser is not None:
            if index not in self._metadata_cache:
                self._metadata_cache[index] = self.metadata_parser.parse(item["input_ids"], labels)
            sample_metadata.update(self._metadata_cache[index])
        if isinstance(recommendation_metadata, dict):
            sample_metadata["recommendation_multi_positive"] = recommendation_metadata
        item.update(
            task_name=self.task_name,
            task_id=self.task_id,
            subtask_name=self.subtask_name,
            subtask_id=self.subtask_id,
            sample_id=index,
            seq_len=len(item["input_ids"]),
            supervised_token_count=sum(token != IGNORE_INDEX for token in labels),
            sample_metadata=sample_metadata,
        )
        return item


class DeterministicMixtureScheduler:
    """A short, shuffled but reproducible subtask cycle."""

    def __init__(self, ratios: Mapping[str, float], seed: int):
        if not ratios or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-8):
            raise ValueError("Subtask ratios must be non-empty and sum to 1.")
        denominators = [Fraction(str(value)).limit_denominator(100).denominator for value in ratios.values()]
        period = math.lcm(*denominators)
        self.base_schedule = [name for name, value in ratios.items() for _ in range(round(value * period))]
        if not self.base_schedule:
            raise ValueError("Subtask schedule is empty.")
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.schedule: list[str] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.position = 0
        self.schedule = list(self.base_schedule)
        random.Random(self.seed + epoch).shuffle(self.schedule)

    def next(self) -> str:
        value = self.schedule[self.position]
        self.position = (self.position + 1) % len(self.schedule)
        return value

    def coverage_metrics(self) -> dict[str, Any]:
        task_total = {}
        task_consumed = {}
        subtask_total = {}
        subtask_consumed = {}
        for task, loader in self.task_loaders.items():
            task_total[task] = loader.total_pack_count
            task_consumed[task] = sum(sampler.cursor for sampler in loader.samplers.values())
            for subtask, sampler in loader.samplers.items():
                key = f"{task}/{subtask}"
                subtask_total[key] = sampler.global_plan_count
                subtask_consumed[key] = sampler.cursor
        stats = getattr(self, "_last_pack_stats", [])
        tokens = [value for value, _ in stats]
        segments = [value for _, value in stats]
        utilization = [value / max(1, 8192) for value in tokens]
        ordered = sorted(utilization)
        p10 = ordered[max(0, int(math.ceil(len(ordered) * 0.10)) - 1)] if ordered else 0.0
        return {
            "task_pack_count_total": task_total,
            "task_pack_consumed": task_consumed,
            "task_coverage_ratio": {name: task_consumed[name] / max(1, task_total[name]) for name in task_total},
            "subtask_pack_count_total": subtask_total,
            "subtask_pack_consumed": subtask_consumed,
            "subtask_coverage_ratio": {name: subtask_consumed[name] / max(1, subtask_total[name]) for name in subtask_total},
            "pack_tokens_mean": sum(tokens) / max(1, len(tokens)),
            "pack_utilization_mean": sum(utilization) / max(1, len(utilization)),
            "pack_utilization_p10": p10,
            "pack_segments_mean": sum(segments) / max(1, len(segments)),
            "current_macro_allocation": dict(self.current_global_allocation),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.set_epoch(int(state["epoch"]))
        self.position = int(state["position"]) % len(self.schedule)


class CoverageDeficitScheduler:
    """Coverage scheduler for task/subtask pack queues."""

    def __init__(self, queue_lengths: Mapping[str, int], seed: int, global_slots: int = 8):
        self.queue_lengths = {name: int(value) for name, value in queue_lengths.items()}
        if not self.queue_lengths or any(value <= 0 for value in self.queue_lengths.values()):
            raise ValueError("Coverage scheduler requires positive queue lengths.")
        self.seed, self.global_slots = int(seed), int(global_slots)
        self.epoch, self.slot_count = 0, 0
        self.consumed = {name: 0 for name in self.queue_lengths}

    @property
    def total(self) -> int:
        return sum(self.queue_lengths.values())

    def _deficit(self, name: str, consumed=None, slot_count=None) -> float:
        consumed = consumed or self.consumed
        slot_count = self.slot_count if slot_count is None else slot_count
        return slot_count * self.queue_lengths[name] / self.total - consumed[name]

    def _pick(self, consumed, slot_count):
        candidates = [name for name, count in self.queue_lengths.items() if consumed[name] < count]
        if not candidates:
            raise RuntimeError("Coverage epoch has no unconsumed queue.")
        order = list(self.queue_lengths)
        return max(candidates, key=lambda name: (self._deficit(name, consumed, slot_count), -order.index(name)))

    def next(self) -> str:
        name = self._pick(self.consumed, self.slot_count)
        self.consumed[name] += 1
        self.slot_count += 1
        return name

    def allocation_at(self, macro_step: int) -> dict[str, int]:
        if self.global_slots < len(self.queue_lengths):
            raise ValueError("global_slots must cover every queue.")
        virtual = dict(self.consumed)
        slots = self.slot_count
        allocation = {name: 1 for name in self.queue_lengths}
        for name in allocation:
            if virtual[name] >= self.queue_lengths[name]:
                raise RuntimeError(
                    f"Queue {name} completed epoch {self.epoch} before barrier; refusing repetition."
                )
            virtual[name] += 1
            slots += 1
        for _ in range(self.global_slots - len(allocation)):
            name = self._pick(virtual, slots)
            allocation[name] += 1
            virtual[name] += 1
            slots += 1
        return allocation

    def commit(self, allocation: Mapping[str, int]) -> None:
        if sum(allocation.values()) != self.global_slots:
            raise ValueError("Coverage allocation size mismatch.")
        for name, count in allocation.items():
            self.consumed[name] += int(count)
        self.slot_count += self.global_slots

    def all_complete(self) -> bool:
        return all(self.consumed[name] >= count for name, count in self.queue_lengths.items())

    def advance_epoch(self, queue_lengths=None) -> None:
        if not self.all_complete():
            raise RuntimeError("Cannot pass global epoch barrier early.")
        if queue_lengths is not None:
            self.queue_lengths = {name: int(value) for name, value in queue_lengths.items()}
        self.epoch += 1
        self.consumed = {name: 0 for name in self.queue_lengths}
        self.slot_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch, "queue_lengths": self.queue_lengths,
            "consumed": self.consumed, "slot_count": self.slot_count,
            "global_slots": self.global_slots,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.queue_lengths = {k: int(v) for k, v in state["queue_lengths"].items()}
        self.consumed = {k: int(v) for k, v in state["consumed"].items()}
        self.slot_count = int(state.get("slot_count", sum(self.consumed.values())))

    def coverage(self) -> dict[str, float]:
        return {name: self.consumed[name] / max(1, count) for name, count in self.queue_lengths.items()}


class LengthBucketPackSampler:
    """Deterministic same-subtask greedy/BFD pack plan with checkpoint cursor."""

    def __init__(
        self, dataset: TokenizedSubDataset, max_pack_length: int, seed: int,
        rank: int = 0, world_size: int = 1, max_segments: int | None = None,
        packing_mode: str = PACKING_MODE_LENGTH_BUCKET_GREEDY, bfd_window_size: int = 4096,
    ):
        if packing_mode not in {PACKING_MODE_LENGTH_BUCKET_GREEDY, PACKING_MODE_SAME_SUBTASK_BFD}:
            raise ValueError(f"Unknown packing mode {packing_mode!r}.")
        if bfd_window_size <= 0:
            raise ValueError("bfd_window_size must be positive.")
        self.dataset, self.max_pack_length = dataset, int(max_pack_length)
        self.seed, self.rank, self.world_size, self.max_segments = seed, rank, world_size, max_segments
        self.packing_mode, self.bfd_window_size = packing_mode, int(bfd_window_size)
        self.epoch, self.cursor, self.samples_seen, self.tokens_seen = 0, 0, 0, 0
        # Read only the Arrow input_ids column once. Repeated row materialization
        # during BFD would otherwise copy full labels/input lists for every
        # candidate lookup and make plan construction dominate startup.
        try:
            self._seq_lengths = [len(value) for value in dataset.dataset["input_ids"]]
        except Exception:
            self._seq_lengths = [len(dataset[index]["input_ids"]) for index in range(len(dataset))]
        self.plan, self.global_plan, self.global_plan_tokens = [], [], []
        self._build_plan()

    def _length(self, index: int) -> int:
        return min(int(self._seq_lengths[index]), self.max_pack_length)

    def _bucket(self, length: int) -> int:
        for bucket in LENGTH_BUCKETS:
            if length <= bucket:
                return bucket
        return LENGTH_BUCKETS[-1]

    def _build_length_bucket_plan(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            buckets[self._bucket(self.dataset[index]["seq_len"])].append(index)
        rng = random.Random(self.seed + self.epoch)
        result = []
        for _, indexes in sorted(buckets.items()):
            rng.shuffle(indexes)
            current, used = [], 0
            for index in indexes:
                length = self._length(index)
                exceeds = self.max_segments is not None and len(current) >= self.max_segments
                if current and (used + length > self.max_pack_length or exceeds):
                    result.append(current)
                    current, used = [], 0
                current.append(index)
                used += length
            if current:
                result.append(current)
        return result

    def _build_bfd_plan(self) -> list[list[int]]:
        # Windowed BFD bounds construction time but remains deterministic.
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)
        result = []
        for start in range(0, len(indices), self.bfd_window_size):
            window = indices[start:start + self.bfd_window_size]
            window.sort(key=lambda index: (-self._length(index), index))
            packs, used = [], []
            for index in window:
                length = self._length(index)
                candidates = [
                    p for p, pack in enumerate(packs)
                    if used[p] + length <= self.max_pack_length
                    and (self.max_segments is None or len(pack) < self.max_segments)
                ]
                if candidates:
                    p = min(candidates, key=lambda item: (self.max_pack_length - used[item] - length, item))
                    packs[p].append(index)
                    used[p] += length
                else:
                    packs.append([index])
                    used.append(length)
            result.extend(packs)
        rng.shuffle(result)
        return result

    def _build_plan(self) -> None:
        self.global_plan = (
            self._build_bfd_plan() if self.packing_mode == PACKING_MODE_SAME_SUBTASK_BFD
            else self._build_length_bucket_plan()
        )
        if not self.global_plan:
            raise RuntimeError(f"No usable packs for {self.dataset.task_name}/{self.dataset.subtask_name}.")
        self.global_plan_tokens = [sum(self._length(index) for index in pack) for pack in self.global_plan]
        padded = list(self.global_plan)
        padding = (-len(padded)) % self.world_size
        if padding:
            padded.extend(padded[:padding])
        self.plan = padded[self.rank::self.world_size]
        if not self.plan:
            raise RuntimeError("No rank-local packs.")
        self.cursor = 0

    @property
    def global_plan_count(self) -> int:
        return len(self.global_plan)

    @property
    def current_epoch_complete(self) -> bool:
        return self.cursor >= len(self.plan)

    def next_pack(self, allow_epoch_rollover: bool = True) -> list[int]:
        if self.cursor >= len(self.plan):
            if not allow_epoch_rollover:
                raise RuntimeError(
                    f"{self.dataset.task_name}/{self.dataset.subtask_name} exhausted before epoch barrier."
                )
            self.epoch += 1
            self._build_plan()
        pack = self.plan[self.cursor]
        self.cursor += 1
        self.samples_seen += len(pack)
        self.tokens_seen += sum(self._seq_lengths[index] for index in pack)
        return pack

    def set_epoch(self, epoch: int) -> None:
        self.epoch, self.cursor = int(epoch), 0
        self._build_plan()

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch, "cursor": self.cursor,
            "samples_seen": self.samples_seen, "tokens_seen": self.tokens_seen,
            "packing_mode": self.packing_mode, "bfd_window_size": self.bfd_window_size,
            "max_pack_length": self.max_pack_length,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_length = state.get("max_pack_length")
        if saved_length is not None and int(saved_length) != self.max_pack_length:
            raise ValueError(
                f"Pack length changed across checkpoint resume: {saved_length} -> {self.max_pack_length}."
            )
        self.epoch, self.cursor = int(state["epoch"]), 0
        self._build_plan()
        self.cursor = int(state["cursor"])
        self.samples_seen = int(state.get("samples_seen", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))


class TaskPackCollator:
    """Concatenates one same-subtask pack and retains segment-level provenance."""

    def __init__(self, base_collator=None):
        self.base_collator = base_collator

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty pack.")
        task_ids = {sample["task_id"] for sample in samples}
        subtask_ids = {sample["subtask_id"] for sample in samples}
        assert len(task_ids) == len(subtask_ids) == 1, "A v1 pack must contain exactly one subtask."
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        attention_ids: list[int] = []
        offsets, cu_seqlens = [], [0]
        segment_lengths: list[int] = []
        action_aux_metadata: list[dict[str, Any]] = []
        supervised = 0
        for segment_id, sample in enumerate(samples, start=1):
            start = len(input_ids)
            ids, segment_labels = list(sample["input_ids"]), list(sample["labels"])
            input_ids.extend(ids)
            labels.extend(segment_labels)
            position_ids.extend(range(len(ids)))
            attention_ids.extend([segment_id] * len(ids))
            end = len(input_ids)
            segment_lengths.append(end - start)
            offsets.append((start, end))
            cu_seqlens.append(end)
            supervised += sample["supervised_token_count"]
            metadata = sample["sample_metadata"]
            if metadata.get("action_select"):
                packed_metadata = offset_action_metadata(metadata, start)
                packed_metadata.update(segment_index=segment_id - 1, sample_id=sample["sample_id"])
                action_aux_metadata.append(packed_metadata)
        assert supervised == sum(token != IGNORE_INDEX for token in labels)
        raw_features = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_ids,
            "position_ids": position_ids,
        }
        if self.base_collator is None:
            model_features = {key: torch.tensor([value], dtype=torch.long) for key, value in raw_features.items()}
        else:
            # Reuse the native 4D/FA2 masking implementation. It is the sole
            # authority for attention isolation; metadata is reattached below
            # and never passed to model.forward.
            model_features = self.base_collator([raw_features])
        model_features.update({
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "segment_offsets": torch.tensor(offsets, dtype=torch.long),
            "task_name": samples[0]["task_name"],
            "task_id": samples[0]["task_id"],
            "subtask_name": samples[0]["subtask_name"],
            "subtask_id": samples[0]["subtask_id"],
            "sample_ids": [sample["sample_id"] for sample in samples],
            "segment_subtask_ids": [sample["subtask_id"] for sample in samples],
            "supervised_token_count": supervised,
            "packed_token_count": len(input_ids),
            "segment_lengths": segment_lengths,
            "num_segments": len(samples),
            "sample_metadata": [sample["sample_metadata"] for sample in samples],
            "action_aux_metadata": action_aux_metadata,
        })
        return model_features


class TaskDataLoader:
    """Infinite, one-top-level-task-at-a-time packed microbatch source."""

    def __init__(
        self, task_name: str, datasets: Mapping[str, TokenizedSubDataset], ratios: Mapping[str, float], collator=None, **kwargs
    ):
        self.task_name = task_name
        self.coverage_mode = bool(kwargs.pop("coverage_mode", False))
        default_length = int(kwargs.pop("max_pack_length"))
        configured_lengths = dict(kwargs.pop("max_pack_length_by_subtask", {}) or {})
        self.samplers = {
            name: LengthBucketPackSampler(
                dataset,
                max_pack_length=int(configured_lengths.get(name, default_length)),
                **kwargs,
            )
            for name, dataset in datasets.items()
        }
        self.scheduler = (
            CoverageDeficitScheduler(
                {name: sampler.global_plan_count for name, sampler in self.samplers.items()},
                kwargs["seed"], global_slots=1,
            )
            if self.coverage_mode else DeterministicMixtureScheduler(ratios, kwargs["seed"])
        )
        self.collator = collator or TaskPackCollator()
        self.datasets = datasets

    @property
    def total_pack_count(self) -> int:
        return sum(sampler.global_plan_count for sampler in self.samplers.values())

    @property
    def current_epoch_complete(self) -> bool:
        return all(sampler.current_epoch_complete for sampler in self.samplers.values())

    def advance_epoch(self) -> None:
        for sampler in self.samplers.values():
            sampler.set_epoch(sampler.epoch + 1)
        if self.coverage_mode:
            self.scheduler.advance_epoch({name: sampler.global_plan_count for name, sampler in self.samplers.items()})
        else:
            self.scheduler.set_epoch(self.scheduler.epoch + 1)

    def coverage_metrics(self) -> dict[str, Any]:
        totals = {name: sampler.global_plan_count for name, sampler in self.samplers.items()}
        consumed = {name: sampler.cursor for name, sampler in self.samplers.items()}
        return {"total": totals, "consumed": consumed, "ratio": {name: consumed[name] / max(1, totals[name]) for name in totals}}

    def __iter__(self) -> "TaskDataLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        subtask = self.scheduler.next()
        sampler = self.samplers[subtask]
        batch = self.collator(
            [self.datasets[subtask][index] for index in sampler.next_pack(allow_epoch_rollover=not self.coverage_mode)]
        )
        batch["pack_max_length"] = sampler.max_pack_length
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "samplers": {name: sampler.state_dict() for name, sampler in self.samplers.items()},
            "max_pack_length_by_subtask": {
                name: sampler.max_pack_length for name, sampler in self.samplers.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.scheduler.load_state_dict(state["scheduler"])
        for name, sampler_state in state["samplers"].items():
            self.samplers[name].load_state_dict(sampler_state)


def split_global_microbatch_allocation(
    allocation: Mapping[str, int], world_size: int
) -> tuple[list[dict[str, int]], dict[str, list[int]]]:
    """Split one ordered global macro-step into equal rank-local worklists.

    This is intentionally distinct from ordinary DDP data parallelism.  The
    returned rank allocations together contain exactly the requested global
    task allocation, so two ranks can collectively process eight microbatches
    instead of each independently processing eight.
    """
    total = sum(allocation.values())
    if world_size <= 0 or total % world_size:
        raise ValueError("The global microbatch count must divide evenly across DDP ranks.")

    task_sequence = [task for task, count in allocation.items() for _ in range(count)]
    local_count = total // world_size
    rank_allocations: list[dict[str, int]] = []
    active_ranks: dict[str, list[int]] = {task: [] for task in allocation}
    for rank in range(world_size):
        local_allocation: dict[str, int] = {}
        for task in task_sequence[rank * local_count : (rank + 1) * local_count]:
            local_allocation[task] = local_allocation.get(task, 0) + 1
        rank_allocations.append(local_allocation)
        for task in local_allocation:
            active_ranks[task].append(rank)
    return rank_allocations, active_ranks


class Balanced40SuperCycle:
    """Deterministic 40-macro-step task schedule for the coverage/GradNorm compromise.

    The counts are deliberately specified in *global* microbatches.  Across one
    cycle they yield material=115, user=123, recommendation=81 and world=1
    packs, while every individual macro-step still contains exactly eight packs.
    """

    CYCLE_LENGTH = 40
    _PATTERN: tuple[tuple[int, dict[str, int]], ...] = (
        (35, {"material": 3, "user": 3, "recommendation": 2, "world": 0}),
        (3, {"material": 2, "user": 4, "recommendation": 2, "world": 0}),
        (1, {"material": 2, "user": 3, "recommendation": 3, "world": 0}),
        (1, {"material": 2, "user": 3, "recommendation": 2, "world": 1}),
    )

    def __init__(self, seed: int, layout: str = TASK_LAYOUT_LEGACY):
        self.seed = seed
        self.layout = resolve_task_layout(layout)

    def allocation_at(self, macro_step: int) -> dict[str, int]:
        if self.layout == TASK_LAYOUT_USER_SPLIT_NO_WORLD:
            # Experiment E plan A: every macro-step carries exactly two packs
            # per top-level task, so a 40-step cycle totals 80/80/80/80 and
            # GradNorm always sees all four tasks in the same macro-step.
            return {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
        if self.layout == TASK_LAYOUT_USER_SPLIT_NO_WORLD_REC_DUAL:
            # rec_A: match the all-train pack volumes while keeping all four
            # GradNorm tasks present in every macro-step.
            return {"material": 4, "user_action": 1, "user_chain": 1, "recommendation": 2}
        cycle, position = divmod(macro_step, self.CYCLE_LENGTH)
        entries = [dict(allocation) for repeats, allocation in self._PATTERN for _ in range(repeats)]
        random.Random(self.seed + cycle).shuffle(entries)
        allocation = entries[position]
        if sum(allocation.values()) != 8:
            raise RuntimeError("Balanced40SuperCycle produced a non-eight-microbatch macro-step.")
        return allocation

    def state_dict(self, macro_step: int) -> dict[str, int | str]:
        return {"mode": "balanced_40", "macro_step": macro_step}


class MultiTaskMacroStepLoader:
    """Yields grouped task microbatches; one iteration is one optimizer update."""

    def __init__(
        self,
        task_loaders: Mapping[str, TaskDataLoader],
        allocation: Mapping[str, int],
        max_steps: int,
        global_allocation: Mapping[str, int] | None = None,
        supercycle: Balanced40SuperCycle | None = None,
        rank: int = 0,
        world_size: int = 1,
        synchronized_global_consumption: bool = False,
        task_ids: Mapping[str, int] | None = None,
        cost_aware_partition: bool = False,
        attention_cost_weight: float = 1.0,
    ):
        self.task_loaders, self.allocation, self.max_steps = dict(task_loaders), dict(allocation), max_steps
        self.global_allocation = dict(global_allocation or allocation)
        self.supercycle, self.rank, self.world_size = supercycle, rank, world_size
        self.synchronized_global_consumption = synchronized_global_consumption
        self.cost_aware_partition = bool(cost_aware_partition)
        self.attention_cost_weight = float(attention_cost_weight)
        if self.attention_cost_weight < 0:
            raise ValueError("attention_cost_weight must be non-negative.")
        self._last_partition_metrics: dict[str, Any] = {}
        self.task_ids = dict(task_ids) if task_ids is not None else dict(TASK_IDS)
        self.macro_step = 0
        if sum(self.global_allocation.values()) != 8:
            raise ValueError("multitask_microbatch_allocation must sum to 8.")
        if self.synchronized_global_consumption:
            if self.world_size <= 0 or 8 % self.world_size:
                raise ValueError("Global macro slots must divide evenly over DDP ranks.")
            if set(self.task_loaders) != set(self.task_ids):
                raise ValueError("Synchronized global consumption requires loaders for all four tasks.")
        self.current_global_allocation = dict(self.global_allocation)
        self.current_local_allocation = dict(self.allocation)

    def __len__(self) -> int:
        return self.max_steps

    def __iter__(self) -> "MultiTaskMacroStepLoader":
        return self

    def _pack_cost(self, microbatch: Mapping[str, Any]) -> float:
        tokens = float(microbatch.get("packed_token_count", 0))
        lengths = microbatch.get("segment_lengths", [])
        max_length = float(microbatch.get("pack_max_length", 8192) or 8192)
        return tokens + self.attention_cost_weight * sum(float(length) ** 2 for length in lengths) / max_length

    def _partition_global_microbatches(self, global_microbatches: list[tuple[str, dict[str, Any]]]):
        if self.world_size != 2 or len(global_microbatches) != 8:
            raise RuntimeError("Cost-aware partition currently requires exactly 8 global packs and 2 ranks.")
        costs = [self._pack_cost(microbatch) for _, microbatch in global_microbatches]
        tokens = [int(microbatch.get("packed_token_count", 0)) for _, microbatch in global_microbatches]
        baseline = tuple(range(0, 8, 2))
        candidates = []
        for mask in range(1 << 8):
            if mask.bit_count() != 4:
                continue
            indices = tuple(index for index in range(8) if mask & (1 << index))
            cost0 = sum(costs[index] for index in indices)
            cost1 = sum(costs) - cost0
            candidates.append((abs(cost0 - cost1), max(cost0, cost1), indices, cost0, cost1))
        _, _, selected, cost0, cost1 = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        selected_set = set(selected)
        rank1_indices = tuple(index for index in range(8) if index not in selected_set)
        rank_tokens = (sum(tokens[index] for index in selected), sum(tokens[index] for index in rank1_indices))
        avg_cost = (cost0 + cost1) / 2.0
        self._last_partition_metrics = {
            "rank0_pack_cost": cost0,
            "rank1_pack_cost": cost1,
            "rank_cost_gap_ratio": abs(cost0 - cost1) / max(avg_cost, 1e-12),
            "rank0_pack_tokens": rank_tokens[0],
            "rank1_pack_tokens": rank_tokens[1],
            "rank_token_gap_ratio": abs(rank_tokens[0] - rank_tokens[1]) / max(sum(rank_tokens) / 2.0, 1.0),
            "partition_changed": int(selected != baseline),
        }
        return (selected, rank1_indices)

    def _assign_global_microbatches(self, global_microbatches):
        # A single-process coverage loader must retain the complete global macro
        # step. Cost-aware 4+4 partitioning is only meaningful for two ranks.
        if self.world_size == 1:
            assignments = (tuple(range(len(global_microbatches))),)
            costs = [self._pack_cost(microbatch) for _, microbatch in global_microbatches]
            tokens = [int(microbatch.get("packed_token_count", 0)) for _, microbatch in global_microbatches]
            self._last_partition_metrics = {
                "rank0_pack_cost": sum(costs),
                "rank1_pack_cost": 0.0,
                "rank_cost_gap_ratio": 2.0 if costs else 0.0,
                "rank0_pack_tokens": sum(tokens),
                "rank1_pack_tokens": 0,
                "rank_token_gap_ratio": 2.0 if tokens else 0.0,
                "partition_changed": 0,
            }
        elif self.cost_aware_partition:
            assignments = self._partition_global_microbatches(global_microbatches)
        else:
            assignments = (tuple(range(0, 8, 2)), tuple(range(1, 8, 2)))
            costs = [self._pack_cost(microbatch) for _, microbatch in global_microbatches]
            tokens = [int(microbatch.get("packed_token_count", 0)) for _, microbatch in global_microbatches]
            rank0, rank1 = assignments
            cost0, cost1 = sum(costs[index] for index in rank0), sum(costs[index] for index in rank1)
            token0, token1 = sum(tokens[index] for index in rank0), sum(tokens[index] for index in rank1)
            self._last_partition_metrics = {
                "rank0_pack_cost": cost0,
                "rank1_pack_cost": cost1,
                "rank_cost_gap_ratio": abs(cost0 - cost1) / max((cost0 + cost1) / 2.0, 1e-12),
                "rank0_pack_tokens": token0,
                "rank1_pack_tokens": token1,
                "rank_token_gap_ratio": abs(token0 - token1) / max((token0 + token1) / 2.0, 1.0),
                "partition_changed": 0,
            }
        return {
            rank: [global_microbatches[index] for index in indices]
            for rank, indices in enumerate(assignments)
        }

    def _record_pack_stats(self, global_microbatches: list[tuple[str, dict[str, Any]]]) -> None:
        self._last_pack_stats = [
            (
                int(microbatch.get("packed_token_count", 0)),
                int(microbatch.get("num_segments", 0)),
                int(microbatch.get("pack_max_length", 8192) or 8192),
                int(microbatch.get("supervised_token_count", 0)),
            )
            for _, microbatch in global_microbatches
        ]
        by_task: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for task, microbatch in global_microbatches:
            by_task[task].append(
                (
                    int(microbatch.get("packed_token_count", 0)),
                    int(microbatch.get("num_segments", 0)),
                    int(microbatch.get("supervised_token_count", 0)),
                )
            )
        self._last_task_pack_stats = dict(by_task)

    def __next__(self) -> dict[str, list[dict[str, Any]]]:
        if self.macro_step >= self.max_steps:
            raise StopIteration
        coverage_mode = isinstance(self.supercycle, CoverageDeficitScheduler)
        if coverage_mode:
            if all(loader.current_epoch_complete for loader in self.task_loaders.values()):
                for loader in self.task_loaders.values():
                    loader.advance_epoch()
                self.supercycle.advance_epoch({
                    task: loader.total_pack_count for task, loader in self.task_loaders.items()
                })
            self.current_global_allocation = self.supercycle.allocation_at(self.macro_step)
            task_slots = [
                task for task in self.task_ids for _ in range(self.current_global_allocation[task])
            ]
            global_microbatches = [(task, next(self.task_loaders[task])) for task in task_slots]
            self.supercycle.commit(self.current_global_allocation)
            assigned = self._assign_global_microbatches(global_microbatches)
            result: dict[str, list[dict[str, Any]]] = {}
            for task, microbatch in assigned[self.rank]:
                result.setdefault(task, []).append(microbatch)
            self.current_local_allocation = {
                task: len(microbatches) for task, microbatches in result.items()
            }
            self._record_pack_stats(global_microbatches)
        elif self.supercycle is None:
            self.current_global_allocation = dict(self.global_allocation)
            result = {
                task: [next(self.task_loaders[task]) for _ in range(count)]
                for task, count in self.allocation.items()
            }
            self.current_local_allocation = dict(self.allocation)
            self._record_pack_stats([(task, microbatch) for task, microbatches in result.items() for microbatch in microbatches])
        else:
            self.current_global_allocation = self.supercycle.allocation_at(self.macro_step)
            task_slots = [
                task for task in self.task_ids for _ in range(self.current_global_allocation[task])
            ]
            global_microbatches = [(task, next(self.task_loaders[task])) for task in task_slots]
            assigned = self._assign_global_microbatches(global_microbatches)
            result = {}
            for task, microbatch in assigned[self.rank]:
                result.setdefault(task, []).append(microbatch)
            self.current_local_allocation = {
                task: len(microbatches) for task, microbatches in result.items()
            }
            self._record_pack_stats(global_microbatches)
        expected_local = 8 // self.world_size if self.synchronized_global_consumption else sum(self.allocation.values())
        if self.synchronized_global_consumption and sum(self.current_local_allocation.values()) != expected_local:
            raise RuntimeError("A DDP rank did not receive the expected number of super-cycle microbatches.")
        self.macro_step += 1
        return result

    def coverage_metrics(self) -> dict[str, Any]:
        task_total = {}
        task_consumed = {}
        subtask_total = {}
        subtask_consumed = {}
        for task, loader in self.task_loaders.items():
            task_total[task] = loader.total_pack_count
            task_consumed[task] = sum(sampler.cursor for sampler in loader.samplers.values())
            for subtask, sampler in loader.samplers.items():
                key = f"{task}/{subtask}"
                subtask_total[key] = sampler.global_plan_count
                subtask_consumed[key] = sampler.cursor
        stats = getattr(self, "_last_pack_stats", [])
        tokens = [row[0] for row in stats]
        segments = [row[1] for row in stats]
        max_lengths = [row[2] for row in stats]
        supervised_tokens = [row[3] for row in stats]
        utilization = [value / max(1, limit) for value, limit in zip(tokens, max_lengths)]
        ordered = sorted(utilization)
        p10 = ordered[max(0, int(math.ceil(len(ordered) * 0.10)) - 1)] if ordered else 0.0
        task_stats = getattr(self, "_last_task_pack_stats", {})
        task_pack_tokens_mean = {
            task: sum(item[0] for item in values) / max(1, len(values))
            for task, values in task_stats.items()
        }
        task_samples_per_pack = {
            task: sum(item[1] for item in values) / max(1, len(values))
            for task, values in task_stats.items()
        }
        return {
            "task_pack_count_total": task_total,
            "task_pack_consumed": task_consumed,
            "task_coverage_ratio": {name: task_consumed[name] / max(1, task_total[name]) for name in task_total},
            "subtask_pack_count_total": subtask_total,
            "subtask_pack_consumed": subtask_consumed,
            "subtask_coverage_ratio": {name: subtask_consumed[name] / max(1, subtask_total[name]) for name in subtask_total},
            "pack_tokens_mean": sum(tokens) / max(1, len(tokens)),
            "pack_utilization_mean": sum(utilization) / max(1, len(utilization)),
            "pack_utilization_p10": p10,
            "pack_segments_mean": sum(segments) / max(1, len(segments)),
            "global_tokens_per_macro": sum(tokens),
            "supervised_tokens_per_macro": sum(supervised_tokens),
            "task_pack_tokens_mean": task_pack_tokens_mean,
            "task_samples_per_pack": task_samples_per_pack,
            "current_macro_allocation": dict(self.current_global_allocation),
            **self._last_partition_metrics,
        }

    def state_dict(self) -> dict[str, Any]:
        state = {
            "macro_step": self.macro_step,
            "task_loaders": {task: loader.state_dict() for task, loader in self.task_loaders.items()},
            "cost_aware_partition": self.cost_aware_partition,
            "attention_cost_weight": self.attention_cost_weight,
        }
        if self.supercycle is not None:
            if isinstance(self.supercycle, Balanced40SuperCycle):
                state["supercycle"] = self.supercycle.state_dict(self.macro_step)
            else:
                state["supercycle"] = self.supercycle.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.macro_step = int(state["macro_step"])
        for task, loader_state in state["task_loaders"].items():
            self.task_loaders[task].load_state_dict(loader_state)
        if self.supercycle is not None and "supercycle" in state and hasattr(self.supercycle, "load_state_dict"):
            self.supercycle.load_state_dict(state["supercycle"])


def build_multitask_datasets(template, model_args, data_args, training_args, tokenizer, processor=None):
    suffix = data_args.multitask_train_dataset_suffix
    layout = resolve_task_layout(getattr(data_args, "multitask_task_layout", TASK_LAYOUT_LEGACY))
    task_datasets = get_task_datasets(layout)
    task_ids = get_task_ids(layout)
    # dataset remains the canonical list of registered logical datasets. Version
    # selection happens below so a one-subtask ablation does not need to rewrite
    # the whole task/sampler declaration.
    expected = {name + suffix for subtasks in task_datasets.values() for name in subtasks.values()}
    provided = set(data_args.dataset or [])
    missing = expected - provided
    if missing:
        raise ValueError(
            f"multitask_macro_training requires all registered datasets for layout {layout!r}; missing: {sorted(missing)}"
        )
    groups: dict[str, dict[str, TokenizedSubDataset]] = {}
    action_metadata_parser = None
    if data_args.user_action_aux_enabled:
        action_metadata_parser = ActionSelectMetadataParser(tokenizer)
    subtask_id = 0
    for task_name, subtasks in task_datasets.items():
        groups[task_name] = {}
        for subtask_name, base_dataset_name in subtasks.items():
            dataset_name = resolve_multitask_dataset_name(base_dataset_name, data_args)
            task_args = copy.deepcopy(data_args)
            task_args.dataset, task_args.eval_dataset, task_args.val_size = [dataset_name], None, 0.0
            task_args.packing, task_args.neat_packing, task_args.tokenized_path = False, False, None
            module = get_dataset(template, model_args, task_args, training_args, "sft", tokenizer, processor)
            metadata_parser = action_metadata_parser if subtask_name == "action_nocot" else None
            groups[task_name][subtask_name] = TokenizedSubDataset(
                module["train_dataset"],
                task_name,
                subtask_name,
                subtask_id,
                metadata_parser,
                task_ids=task_ids,
            )
            subtask_id += 1
    return groups
