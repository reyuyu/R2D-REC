"""Optional task-aware SFT batching used by ``multitask_macro_training``.

The module deliberately keeps task scheduling and packing separate from loss
composition.  It is therefore safe to use as the data-side foundation for
future GradNorm/PCGrad work without introducing either algorithm here.
"""

from __future__ import annotations

import copy
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

TASK_LAYOUT_LEGACY = "legacy"
TASK_LAYOUT_USER_SPLIT_NO_WORLD = "user_split_no_world"
TASK_LAYOUTS = (TASK_LAYOUT_LEGACY, TASK_LAYOUT_USER_SPLIT_NO_WORLD)

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
}
TASK_IDS_BY_LAYOUT: dict[str, dict[str, int]] = {
    TASK_LAYOUT_LEGACY: TASK_IDS,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD: {"material": 0, "user_action": 1, "user_chain": 2, "recommendation": 3},
}
SUBTASK_RATIOS_BY_LAYOUT: dict[str, dict[str, dict[str, float]]] = {
    TASK_LAYOUT_LEGACY: SUBTASK_RATIOS,
    TASK_LAYOUT_USER_SPLIT_NO_WORLD: {
        "material": {"cot": 0.50, "nocot": 0.50},
        "user_action": {"action_nocot": 1.00},
        "user_chain": {"cot": 0.50, "nocot": 0.50},
        "recommendation": {"cot": 1.00},
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
        if self.metadata_parser is None:
            sample_metadata = {}
        else:
            if index not in self._metadata_cache:
                self._metadata_cache[index] = self.metadata_parser.parse(item["input_ids"], labels)
            sample_metadata = self._metadata_cache[index]
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

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.set_epoch(int(state["epoch"]))
        self.position = int(state["position"]) % len(self.schedule)


class LengthBucketPackSampler:
    """Creates same-subtask greedy packs and exposes resumable rank-local plans."""

    def __init__(
        self,
        dataset: TokenizedSubDataset,
        max_pack_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        max_segments: int | None = None,
    ):
        self.dataset, self.max_pack_length = dataset, max_pack_length
        self.seed, self.rank, self.world_size, self.max_segments = seed, rank, world_size, max_segments
        self.epoch, self.cursor, self.samples_seen, self.tokens_seen = 0, 0, 0, 0
        self.plan: list[list[int]] = []
        self._build_plan()

    def _bucket(self, length: int) -> int:
        for bucket in LENGTH_BUCKETS:
            if length <= bucket:
                return bucket
        return LENGTH_BUCKETS[-1]

    def _build_plan(self) -> None:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            buckets[self._bucket(self.dataset[index]["seq_len"])].append(index)
        rng = random.Random(self.seed + self.epoch)
        global_plan: list[list[int]] = []
        for _, indexes in sorted(buckets.items()):
            rng.shuffle(indexes)
            current: list[int] = []
            used = 0
            for index in indexes:
                length = self.dataset[index]["seq_len"]
                if length > self.max_pack_length:
                    # Native tokenization already applies cutoff; retain the sample as a single segment.
                    length = self.max_pack_length
                exceeds_segments = self.max_segments is not None and len(current) >= self.max_segments
                if current and (used + length > self.max_pack_length or exceeds_segments):
                    global_plan.append(current)
                    current, used = [], 0
                current.append(index)
                used += length
            if current:
                global_plan.append(current)
        # Keep the same number of packs on every rank.  A short subtask may
        # form fewer packs than the DDP world size; pad deterministically just
        # like DistributedSampler instead of leaving a rank with no work.
        if global_plan:
            padding = (-len(global_plan)) % self.world_size
            if padding:
                global_plan.extend(global_plan[:padding])
        self.plan = global_plan[self.rank :: self.world_size]
        if not self.plan:
            raise RuntimeError(f"No usable packs for {self.dataset.task_name}/{self.dataset.subtask_name}.")
        self.cursor = 0

    def next_pack(self) -> list[int]:
        if self.cursor >= len(self.plan):
            self.epoch += 1
            self._build_plan()
        pack = self.plan[self.cursor]
        self.cursor += 1
        self.samples_seen += len(pack)
        self.tokens_seen += sum(self.dataset[index]["seq_len"] for index in pack)
        return pack

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build_plan()

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state["epoch"])
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
        self.scheduler = DeterministicMixtureScheduler(ratios, kwargs["seed"])
        self.samplers = {name: LengthBucketPackSampler(dataset, **kwargs) for name, dataset in datasets.items()}
        self.collator = collator or TaskPackCollator()
        self.datasets = datasets

    def __iter__(self) -> "TaskDataLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        subtask = self.scheduler.next()
        sampler = self.samplers[subtask]
        return self.collator([self.datasets[subtask][index] for index in sampler.next_pack()])

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict(), "samplers": {name: sampler.state_dict() for name, sampler in self.samplers.items()}}

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
    ):
        self.task_loaders, self.allocation, self.max_steps = dict(task_loaders), dict(allocation), max_steps
        self.global_allocation = dict(global_allocation or allocation)
        self.supercycle, self.rank, self.world_size = supercycle, rank, world_size
        self.synchronized_global_consumption = synchronized_global_consumption
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

    def __next__(self) -> dict[str, list[dict[str, Any]]]:
        if self.macro_step >= self.max_steps:
            raise StopIteration
        if self.supercycle is None:
            self.current_global_allocation = dict(self.global_allocation)
            result = {task: [next(self.task_loaders[task]) for _ in range(count)] for task, count in self.allocation.items()}
            self.current_local_allocation = dict(self.allocation)
        else:
            self.current_global_allocation = self.supercycle.allocation_at(self.macro_step)
            task_slots = [
                task for task in self.task_ids for _ in range(self.current_global_allocation[task])
            ]
            # Every rank advances the same global task/pack queue.  It only
            # forwards its interleaved slots, so two ranks consume all eight
            # distinct packs while each performs exactly four backward calls.
            global_microbatches = [(task, next(self.task_loaders[task])) for task in task_slots]
            result: dict[str, list[dict[str, Any]]] = {}
            for slot_index, (task, microbatch) in enumerate(global_microbatches):
                if slot_index % self.world_size == self.rank:
                    result.setdefault(task, []).append(microbatch)
            self.current_local_allocation = {task: len(microbatches) for task, microbatches in result.items()}
            expected_local = 8 // self.world_size
            if sum(self.current_local_allocation.values()) != expected_local:
                raise RuntimeError("A DDP rank did not receive the expected number of super-cycle microbatches.")
        self.macro_step += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        state = {
            "macro_step": self.macro_step,
            "task_loaders": {task: loader.state_dict() for task, loader in self.task_loaders.items()},
        }
        if self.supercycle is not None:
            state["supercycle"] = self.supercycle.state_dict(self.macro_step)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.macro_step = int(state["macro_step"])
        for task, loader_state in state["task_loaders"].items():
            self.task_loaders[task].load_state_dict(loader_state)


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
