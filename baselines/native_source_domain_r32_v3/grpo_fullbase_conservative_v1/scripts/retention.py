"""Fixed recommendation retention probes with complete RNG restoration."""
from __future__ import annotations

import json
import random
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

try:
    from transformers import TrainerCallback
except ModuleNotFoundError:  # Pure contract tests do not require Transformers.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from checkpointing import write_json_atomic
from modeling import adapter_disabled


@contextmanager
def preserve_runtime_state(trainer: Any):
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    was_training = trainer.model.training
    active_route = getattr(trainer, "_active_route", None)
    stop_think = getattr(trainer, "_stop_think_at_closure", None)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            trainer.model.train()
        else:
            trainer.model.eval()
        if active_route is None and hasattr(trainer, "_active_route"):
            delattr(trainer, "_active_route")
        elif active_route is not None:
            trainer._active_route = active_route
        if stop_think is not None:
            trainer._stop_think_at_closure = stop_think


def evaluate_preserving_state(evaluator: Any, step: int, reason: str, *, disable_lora=False):
    with preserve_runtime_state(evaluator.trainer):
        context = adapter_disabled(evaluator.trainer.model) if disable_lora else _null_context()
        with context:
            evaluator.evaluate(step, reason)


@contextmanager
def _null_context():
    yield


def _probe_rows(path: str | Path) -> list[dict[str, Any]]:
    probe_path = Path(path)
    if not probe_path.is_file():
        return []
    return [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line]


def _candidate_contract(row: dict[str, Any]) -> dict[str, Any]:
    return {
        route: [
            {
                key: candidate.get(key)
                for key in (
                    "completion_sha256", "reward", "parsed_sid", "closed",
                    "exact", "ab", "a", "invalid", "beam_sids",
                )
            }
            for candidate in row[route]["candidates"]
        ]
        for route in ("think", "nothink")
    }


def verify_step0_parity(probe_path: str | Path, group_ids: Iterable[str]) -> dict[str, Any]:
    rows = _probe_rows(probe_path)
    by_key = {(row.get("probe_suite"), row["group_id"]): row for row in rows if row["step"] == 0}
    mismatches = []
    for group_id in group_ids:
        pure = by_key.get(("step0_pure_full_sft", group_id))
        fresh = by_key.get(("retention", group_id))
        if pure is None or fresh is None:
            mismatches.append({"group_id": group_id, "reason": "missing probe row"})
        elif _candidate_contract(pure) != _candidate_contract(fresh):
            mismatches.append({"group_id": group_id, "reason": "candidate contract mismatch"})
    result = {
        "step0_parity": not mismatches,
        "group_count": len(list(group_ids)),
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError(f"fresh LoRA changed Step0 probe behavior: {mismatches}")
    return result


def summarize_retention(probe_path: str | Path, *, suite="retention") -> dict[str, Any]:
    rows = [row for row in _probe_rows(probe_path) if row.get("probe_suite") == suite]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    result: dict[str, Any] = {}
    for step, step_rows in sorted(by_step.items()):
        think_candidates = [candidate for row in step_rows for candidate in row["think"]["candidates"]]
        no_candidates = [candidate for row in step_rows for candidate in row["nothink"]["candidates"]]
        beam_total = 32 * len(think_candidates)
        think_groups_success = sum(
            any((candidate.get("a") or 0) > 0 for candidate in row["think"]["candidates"])
            for row in step_rows
        )
        result[str(step)] = {
            "group_count": len(step_rows),
            "think": {
                "exact_beam_rate": sum((c.get("exact") or 0) for c in think_candidates) / max(beam_total, 1),
                "ab_plus_beam_rate": sum((c.get("ab") or 0) for c in think_candidates) / max(beam_total, 1),
                "a_plus_beam_rate": sum((c.get("a") or 0) for c in think_candidates) / max(beam_total, 1),
                "mean_reward": statistics.fmean(float(c["reward"]) for c in think_candidates),
                "success_at_k": think_groups_success / max(len(step_rows), 1),
                "success_at_32": sum((c.get("a") or 0) > 0 for c in think_candidates) / max(len(think_candidates), 1),
                "invalid_rate": sum((c.get("invalid") or 0) for c in think_candidates) / max(beam_total, 1),
                "closure_rate": sum(bool(c.get("closed")) for c in think_candidates) / max(len(think_candidates), 1),
            },
            "nothink": {
                "exact_rate": sum(float(c["reward"]) == 8.0 for c in no_candidates) / max(len(no_candidates), 1),
                "ab_plus_rate": sum(float(c["reward"]) >= 2.0 for c in no_candidates) / max(len(no_candidates), 1),
                "a_plus_rate": sum(float(c["reward"]) >= 0.5 for c in no_candidates) / max(len(no_candidates), 1),
                "mean_reward": statistics.fmean(float(c["reward"]) for c in no_candidates),
                "positive_candidate_rate": sum(float(c["reward"]) > 0 for c in no_candidates) / max(len(no_candidates), 1),
                "success_at_k": sum(
                    any(float(c["reward"]) > 0 for c in row["nothink"]["candidates"])
                    for row in step_rows
                ) / max(len(step_rows), 1),
            },
        }
    return result


class MilestoneRetentionCallback(TrainerCallback):
    def __init__(self, pure_evaluator: Any, retention_evaluator: Any, milestones: Iterable[int]):
        self.pure_evaluator = pure_evaluator
        self.retention_evaluator = retention_evaluator
        self.milestones = set(int(step) for step in milestones)

    def on_train_begin(self, args, state, control, **kwargs):
        if int(state.global_step) != 0:
            raise RuntimeError("this pilot requires a fresh Step0 start")
        evaluate_preserving_state(self.pure_evaluator, 0, "pure-full-sft", disable_lora=True)
        evaluate_preserving_state(self.retention_evaluator, 0, "fresh-lora-step0")
        verify_step0_parity(
            self.retention_evaluator.monitor.run_dir / "probes.jsonl",
            self.retention_evaluator.group_ids,
        )

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.milestones and step > 0:
            evaluate_preserving_state(self.retention_evaluator, step, "milestone")

    def on_train_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in self.milestones:
            evaluate_preserving_state(self.retention_evaluator, step, "final")


def write_retention_summary(monitor_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    monitor_dir = Path(monitor_dir)
    value = summarize_retention(monitor_dir / "probes.jsonl")
    write_json_atomic(Path(output_path), value)
    return value
