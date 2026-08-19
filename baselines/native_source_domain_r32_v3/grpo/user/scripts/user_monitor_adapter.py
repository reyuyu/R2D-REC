"""Passive bridge from existing User GRPO metrics to MonitorWriter.

The adapter only selects already-computed Python values. It performs no reward
scoring, model work, synchronization, networking, or background execution.
"""

from __future__ import annotations

from typing import Any, Mapping


RUN_KIND = "user_grpo"
ROUTES = frozenset({"action", "chain"})

POLICY_FIELDS = (
    "loss",
    "grad_norm",
    "learning_rate",
    "ratio_mean",
    "clip_fraction",
    "approx_kl",
    "policy_wall_sec",
)

COMMON_ROLLOUT_FIELDS = (
    "task_reward_mean",
    "task_reward_std",
    "zero_std_ratio",
    "sequence_advantage_mean",
    "sequence_advantage_std",
    "token_advantage_mean",
    "token_advantage_std",
    "masked_candidate_rate",
    "masked_token_rate",
    "positive_sequence_masked_token_flip_count",
    "per_kind_masked_token_count",
    "per_kind_incremental_negative_mass",
    "violation_counts",
)

ACTION_FIELDS = (
    "f1_mean",
    "precision_mean",
    "recall_mean",
    "exact_match_rate",
    "hallucination_candidate_rate",
    "duplicate_candidate_rate",
    "wrong_selection_candidate_rate",
)

CHAIN_FIELDS = (
    "total_reward_mean",
    "action_alignment_mean",
    "logic_alignment_mean",
    "grounded_rate",
    "partially_grounded_rate",
    "ungrounded_rate",
)


def user_run_manifest(**values: Any) -> dict[str, Any]:
    """Build the stable User run manifest without touching the filesystem."""
    manifest = {
        "run_kind": RUN_KIND,
        "experiment": "GR_USER_v1",
        "parent": "BATA baseline Epoch2 checkpoint-1106",
        "G": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "reward": {"action": "set_f1", "chain": "action_logic_alignment"},
        "token_penalty": {"strategy": "sqrt", "lambda": 0.5},
    }
    manifest.update(values)
    manifest["run_kind"] = RUN_KIND
    return manifest


def _selected(source: Mapping[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    source = source or {}
    return {field: source[field] for field in fields if field in source}


def build_user_step_event(
    *,
    step: int,
    route: str,
    rollout_metrics: Mapping[str, Any],
    policy_metrics: Mapping[str, Any] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Flatten existing trainer metrics into one append-only step event."""
    if route not in ROUTES:
        raise ValueError(f"unsupported User GRPO route: {route}")
    event = {"step": int(step), "route": route, **metadata}
    event.update(_selected(policy_metrics, POLICY_FIELDS))
    event.update(_selected(rollout_metrics, COMMON_ROLLOUT_FIELDS))
    event.update(_selected(rollout_metrics, ACTION_FIELDS if route == "action" else CHAIN_FIELDS))
    return event


class UserMonitorAdapter:
    """Thin facade over an existing MonitorWriter-compatible object."""

    def __init__(self, writer: Any) -> None:
        self.writer = writer

    def write_manifest(self, **values: Any) -> bool:
        return bool(self.writer.write_manifest(user_run_manifest(**values)))

    def write_step(
        self,
        *,
        step: int,
        route: str,
        rollout_metrics: Mapping[str, Any],
        policy_metrics: Mapping[str, Any] | None = None,
        **metadata: Any,
    ) -> bool:
        return bool(
            self.writer.write_step(
                build_user_step_event(
                    step=step,
                    route=route,
                    rollout_metrics=rollout_metrics,
                    policy_metrics=policy_metrics,
                    **metadata,
                )
            )
        )

    def write_rollout(self, event: Mapping[str, Any]) -> bool:
        return bool(self.writer.write_rollout(dict(event)))

    def write_trace(self, event: Mapping[str, Any]) -> bool:
        return bool(self.writer.write_trace(dict(event)))
