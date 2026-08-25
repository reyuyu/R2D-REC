"""Frozen evaluation interface declarations; no model execution in Phase 1.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationInterface:
    name: str
    expected_groups: int
    group_ids: tuple[str, ...]
    optimizer_allowed: bool = False
    backward_allowed: bool = False
    checkpoint_selection_allowed: bool = True


def evaluate_probe20(probe_group_ids: Sequence[str], dev_group_ids: Sequence[str]) -> EvaluationInterface:
    probe, dev = tuple(probe_group_ids), set(dev_group_ids)
    if len(probe) != 20 or len(set(probe)) != 20 or not set(probe) <= dev:
        raise ValueError("Probe20 must be 20 unique groups and a Dev subset")
    return EvaluationInterface("probe20", 20, probe)


def evaluate_dev512(dev_group_ids: Sequence[str]) -> EvaluationInterface:
    dev = tuple(dev_group_ids)
    if len(dev) != 512 or len(set(dev)) != 512:
        raise ValueError("Dev512 must contain 512 unique groups")
    return EvaluationInterface("dev512", 512, dev)


def final_checkpoint_selection_interface() -> EvaluationInterface:
    return EvaluationInterface("final2048", 2048, (), checkpoint_selection_allowed=False)
