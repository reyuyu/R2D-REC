"""Read-only Gold CoT join layer with a fail-closed provenance gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping


class DataProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetTopology:
    original_think_groups: int
    fixed_probe_groups: int
    post_probe_groups: int
    sampler_dropped_groups: int
    training_groups: int
    fresh_rollouts: int
    optimizer_steps: int


def planned_topology(group_count: int = 1549, probe_count: int = 4, group_size: int = 4, num_iterations: int = 2) -> DatasetTopology:
    post_probe = group_count - probe_count
    dropped = post_probe % group_size
    training = post_probe - dropped
    fresh_rollouts = training // group_size
    return DatasetTopology(
        original_think_groups=group_count,
        fixed_probe_groups=probe_count,
        post_probe_groups=post_probe,
        sampler_dropped_groups=dropped,
        training_groups=training,
        fresh_rollouts=fresh_rollouts,
        optimizer_steps=fresh_rollouts * num_iterations,
    )


def build_think_composite_dataset(
    records: Iterable[Mapping],
    gold_by_group: Mapping[str, str],
    *,
    provenance_ready: bool,
) -> list[dict]:
    """Join Gold after loading; never mutate or augment the generation prompt."""
    if not provenance_ready:
        raise DataProvenanceError("Gold CoT provenance is incomplete; formal dataset construction is blocked")
    output = []
    for source in records:
        if source.get("route") != "think":
            continue
        group_id = source["recommendation_group_id"]
        if group_id not in gold_by_group:
            raise DataProvenanceError("missing Gold CoT for group " + group_id)
        record = copy.deepcopy(dict(source))
        original_prompt = record["prompt"]
        record["gold_cot"] = gold_by_group[group_id]
        if record["prompt"] != original_prompt:
            raise AssertionError("Gold CoT changed the model input prompt")
        output.append(record)
    return output
