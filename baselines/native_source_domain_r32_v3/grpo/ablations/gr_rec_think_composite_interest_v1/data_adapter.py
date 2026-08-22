"""Read-only Gold CoT join layer with a fail-closed provenance gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_interest_units



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


def planned_topology(group_count: int = 1549, probe_count: int = 12, group_size: int = 4, num_iterations: int = 2) -> DatasetTopology:
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
    eligible_group_ids: Iterable[str],
) -> list[dict]:
    """Build only the explicitly audited, provenance-safe and parser-valid cohort."""
    eligible = set(eligible_group_ids)
    output = []
    for source in records:
        if source.get("route") != "think":
            continue
        group_id = source["recommendation_group_id"]
        if group_id not in eligible:
            continue
        if group_id not in gold_by_group:
            raise DataProvenanceError("eligible group is missing Gold CoT: " + group_id)
        record = copy.deepcopy(dict(source))
        original_prompt = record["prompt"]
        gold_cot = gold_by_group[group_id]
        parsed = extract_interest_units(gold_cot, original_prompt)
        if not parsed.parser_success or not parsed.units:
            raise DataProvenanceError("eligible group has parser-invalid Gold CoT: " + group_id)
        record["gold_cot"] = gold_cot
        if record["prompt"] != original_prompt:
            raise AssertionError("Gold CoT changed the model input prompt")
        output.append(record)
    if {record["recommendation_group_id"] for record in output} != eligible:
        missing = eligible.difference(record["recommendation_group_id"] for record in output)
        raise DataProvenanceError("eligible groups are absent from source records: " + ",".join(sorted(missing)))
    return output
