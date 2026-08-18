"""CPU-only monitor aggregation for DSR-Simple."""
from __future__ import annotations

import os
import statistics
from collections import Counter
from typing import Sequence

from gr_rec_dsr_v1.dsr_monitor import summarize_nothink_records

from .simple_objectives import (
    SIMPLE_BRANCHES,
    choose_simple_think_aux_scores,
    simple_group_advantages,
)


TRAINING_SIGNAL_FIELDS = (
    "raw_interest_n",
    "s_n",
    "unique_valid_target_a",
    "d_a",
    "simple_s_aux",
    "simple_a_aux",
    "simple_branch",
)
DIAGNOSTIC_ONLY_FIELDS = (
    "grounded_n",
    "grounding_coverage",
    "fake_or_ungrounded_sid_count",
    "d_cot",
    "s_cot",
    "s_prefix",
    "entropy_s_explore",
    "beam_invalid",
    "exact",
    "ab",
    "a",
)


def decorate_simple_monitor(writer, runner: str):
    original = writer.write_manifest

    def write_manifest(manifest):
        return original({
            **manifest,
            "runner": runner,
            "experiment": "GR_REC_DSR_Simple_Ablation_v1",
            "parent": "BATA",
            "baseline": "GR_REC_v1",
            "reference_ablation": "GR_REC_DSR_Ablation_v1",
            "dsr_simple": {
                "think_lambda": float(os.environ.get("DSR_SIMPLE_THINK_LAMBDA", "0.10")),
                "nothink_scale": float(os.environ.get("DSR_SIMPLE_NOTHINK_SCALE", "1.0")),
                "training_signal_fields": list(TRAINING_SIGNAL_FIELDS),
                "diagnostic_only_fields": list(DIAGNOSTIC_ONLY_FIELDS),
                "extra_forward": False,
                "sampling_changed": False,
                "whole_cot_credit_assignment": True,
            },
        })

    writer.write_manifest = write_manifest
    return writer


def _groups(records: Sequence[dict], group_size: int) -> list[list[dict]]:
    if len(records) % group_size:
        raise ValueError(f"record count {len(records)} is not divisible by G={group_size}")
    return [list(records[start:start + group_size]) for start in range(0, len(records), group_size)]


def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _rate(values):
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def summarize_simple_think_records(
    records: Sequence[dict],
) -> tuple[dict, list[float], list[float]]:
    """Aggregate existing records without tokenizer, parser, model, or Beam work."""
    grouped = _groups(records, 4)
    scores: list[float] = []
    branches: list[str] = []
    group_aux_stds: list[float] = []
    primary_zero: list[bool] = []
    all_zero: list[bool] = []
    rescue_active: list[bool] = []
    for group in grouped:
        group_scores, branch = choose_simple_think_aux_scores(group)
        rewards = [float(item["primary_reward"]) for item in group]
        aux_std = statistics.pstdev(group_scores)
        scores.extend(group_scores)
        branches.append(branch)
        group_aux_stds.append(aux_std)
        primary_zero.append(statistics.pstdev(rewards) == 0.0)
        all_zero.append(all(value == 0.0 for value in rewards))
        rescue_active.append(branch != "primary_only" and aux_std > 0.0)
    advantages = simple_group_advantages(scores).tolist()
    for item, score, advantage in zip(records, scores, advantages):
        item["simple_s_aux"] = score
        item["simple_a_aux"] = advantage
    for group, branch in zip(grouped, branches):
        for item in group:
            item["simple_branch"] = branch

    diagnostics = [item.get("diagnostic_only") or {} for item in records]
    coverage = [item.get("grounding_coverage") for item in diagnostics]
    coverage = [value for value in coverage if value is not None]
    branch_counts = Counter(branches)
    payload = {
        "group_count": len(grouped),
        "training_signal": {
            "diagnostic_only": False,
            "raw_interest_n_mean": _mean(int(item["raw_interest_n"]) for item in records),
            "s_n_mean": _mean(float(item["s_n"]) for item in records),
            "unique_valid_target_a_mean": _mean(
                int(item["unique_valid_target_a"]) for item in records
            ),
            "d_a_mean": _mean(float(item["d_a"]) for item in records),
            "simple_s_aux_mean": _mean(scores),
            "simple_aux_std_mean": _mean(group_aux_stds),
            "simple_a_aux_abs_mean": _mean(abs(value) for value in advantages),
            "simple_rescue_active_rate": _rate(rescue_active),
            "branches": {name: branch_counts[name] for name in SIMPLE_BRANCHES},
        },
        "primary_zero_std_rate": _rate(primary_zero),
        "all_zero_rate": _rate(all_zero),
        "raw_interest_n_mean": _mean(int(item["raw_interest_n"]) for item in records),
        "s_n_mean": _mean(float(item["s_n"]) for item in records),
        "unique_valid_target_a_mean": _mean(
            int(item["unique_valid_target_a"]) for item in records
        ),
        "d_a_mean": _mean(float(item["d_a"]) for item in records),
        "simple_aux_std_mean": _mean(group_aux_stds),
        "simple_a_aux_abs_mean": _mean(abs(value) for value in advantages),
        "simple_rescue_active_rate": _rate(rescue_active),
        "simple_branches": {name: branch_counts[name] for name in SIMPLE_BRANCHES},
        "simple_aux_scores": scores,
        "simple_aux_advantages": advantages,
        "diagnostic_only": {
            "diagnostic_only": True,
            "fields": list(DIAGNOSTIC_ONLY_FIELDS),
            "grounded_n_mean": _mean(int(item.get("grounded_n", 0)) for item in diagnostics),
            "grounding_coverage_mean": _mean(coverage) if coverage else None,
            "fake_or_ungrounded_sid_count": sum(
                int(item.get("fake_or_ungrounded_sid_count", 0)) for item in diagnostics
            ),
            "d_cot_mean": _mean(float(item.get("d_cot", 0.0)) for item in diagnostics),
            "s_prefix_mean": _mean(float(item.get("s_prefix", 0.0)) for item in diagnostics),
            "beam_invalid_count": sum(int(item.get("beam_invalid", 0)) for item in records),
            "exact_mean": _mean(float(item.get("exact", 0.0)) for item in diagnostics),
            "ab_mean": _mean(float(item.get("ab", 0.0)) for item in diagnostics),
            "a_mean": _mean(float(item.get("a", 0.0)) for item in diagnostics),
        },
    }
    return payload, scores, advantages


__all__ = [
    "decorate_simple_monitor",
    "summarize_nothink_records",
    "summarize_simple_think_records",
]
