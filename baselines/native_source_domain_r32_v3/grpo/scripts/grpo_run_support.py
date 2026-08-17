"""Pure helpers for formal GRPO run planning and checkpoint validation."""
from __future__ import annotations

import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"(?:^|[\\/])checkpoint-(\d+)[\\/]?$")


def count_raw_groups(data_path):
    group_ids = set()
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                group_ids.add(json.loads(line)["recommendation_group_id"])
    return len(group_ids)


def resolve_n_groups(value, raw_group_count):
    if str(value).lower() == "all":
        return raw_group_count
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--n-groups must be a positive integer or 'all'") from exc
    if selected < 1 or selected > raw_group_count:
        raise ValueError(
            f"--n-groups must be in [1, {raw_group_count}] or 'all', got {value!r}"
        )
    return selected


def audit_sampler(dataset, sampler):
    rows = list(dataset)
    selected = {
        row["recommendation_group_id"] for row in rows if row["route"] == "think"
    }
    covered = {"think": set(), "no_think": set()}
    rollout_counts = {"think": 0, "no_think": 0}
    route_schedule = []
    for route, indices in sampler._chunks:
        rollout_counts[route] += 1
        route_schedule.append(route)
        covered[route].update(rows[index]["recommendation_group_id"] for index in indices)
    trained = covered["think"] & covered["no_think"]
    dropped = sorted(selected - trained)
    return {
        "selected_groups": len(selected),
        "trained_groups": len(trained),
        "dropped_groups": len(dropped),
        "dropped_group_ids": dropped,
        "think_unique_groups": len(covered["think"]),
        "nothink_unique_groups": len(covered["no_think"]),
        "think_rollouts": rollout_counts["think"],
        "nothink_rollouts": rollout_counts["no_think"],
        "optimizer_steps": len(sampler._chunks) * sampler.repeat_count,
        "repeat_count": sampler.repeat_count,
        "route_schedule_preview": route_schedule[:24],
    }


def validate_save_steps(save_steps):
    if save_steps < 1 or save_steps % 2:
        raise ValueError(
            f"save_steps must be a positive even integer because num_iterations=2; got {save_steps}"
        )
    return save_steps


def checkpoint_step(checkpoint_path):
    match = CHECKPOINT_RE.search(str(Path(checkpoint_path)))
    if not match:
        raise ValueError(
            "resume checkpoint must end in checkpoint-<global_step>, "
            f"got {checkpoint_path!r}"
        )
    return int(match.group(1))


def validate_resume_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return None
    step = checkpoint_step(checkpoint_path)
    if step % 2:
        raise ValueError(
            f"checkpoint-{step} is inside a two-iteration rollout; resume only from even steps"
        )
    return step
