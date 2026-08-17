"""Fail-open append-only writer for passive GRPO monitoring.

This module intentionally has no torch, networking, database, or threading
dependency. Callers must pass ordinary Python scalars and containers only.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ENV_KEYS = (
    "GRPO_MONITOR",
    "GRPO_MONITOR_DIR",
    "GRPO_MONITOR_STEP_EVERY",
    "GRPO_MONITOR_ROLLOUT_EVERY",
    "GRPO_TRACE_EVERY",
    "GRPO_BEAM_RANK_BALANCE",
    "GRPO_DETAILED_MONITOR",
    "GRPO_GENERATION_PROFILE",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned[:160] or "grpo-run"


def json_safe(value: Any) -> Any:
    """Return strict-JSON data without retaining tensor-like objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # numpy scalars and similar CPU scalar wrappers only. GPU tensors are not
    # accepted: their .item() could synchronize the training hot path.
    module = type(value).__module__
    if module.startswith("numpy") and hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


class MonitorWriter:
    """Small synchronous JSONL writer. Every public operation is fail-open."""

    def __init__(
        self,
        enabled: bool,
        root_dir: str | os.PathLike[str] | None = None,
        run_id: str = "grpo-run",
        rank: int = 0,
        step_every: int = 1,
        rollout_every: int = 1,
        trace_every: int = 20,
    ) -> None:
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.step_every = max(1, int(step_every))
        self.rollout_every = max(1, int(rollout_every))
        self.trace_every = max(1, int(trace_every))
        self.errors = 0
        self.run_id = _safe_run_id(run_id)
        self.run_dir = Path(root_dir or ".") / self.run_id
        if self.enabled:
            self._guard(self._create_layout)

    def _create_layout(self) -> None:
        (self.run_dir / "ranks").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "traces").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "probes").mkdir(parents=True, exist_ok=True)

    def _guard(self, operation) -> bool:
        if not self.enabled:
            return False
        try:
            operation()
            return True
        except Exception:
            self.errors += 1
            return False

    def _append(self, relative_path: str, event: Mapping[str, Any]) -> bool:
        payload = json_safe(dict(event))
        payload.setdefault("timestamp", utc_timestamp())
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

        def write() -> None:
            path = self.run_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

        return self._guard(write)

    def write_manifest(self, manifest: Mapping[str, Any]) -> bool:
        if self.rank != 0:
            return False
        payload = json_safe(dict(manifest))
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("start_time", utc_timestamp())
        payload.setdefault("environment", {key: os.environ.get(key) for key in ENV_KEYS})

        def write() -> None:
            path = self.run_dir / "manifest.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)

        return self._guard(write)

    def write_step(self, event: Mapping[str, Any]) -> bool:
        step = int(event.get("step") or 0)
        if self.rank != 0 or step % self.step_every:
            return False
        return self._append("metrics.jsonl", {"type": "step", **event})

    def write_rollout(self, event: Mapping[str, Any]) -> bool:
        rollout_id = int(event.get("rollout_id") or 0)
        if self.rank != 0 or rollout_id % self.rollout_every:
            return False
        return self._append("rollouts.jsonl", {"type": "rollout", **event})

    def write_rank(self, event: Mapping[str, Any]) -> bool:
        rollout_id = int(event.get("rollout_id") or 0)
        if rollout_id % self.rollout_every:
            return False
        return self._append(
            f"ranks/rank{self.rank}.jsonl",
            {"type": "rank", "rank": self.rank, **event},
        )

    def trace_due(self, rollout_id: int) -> bool:
        return self.enabled and self.rank == 0 and rollout_id > 0 and rollout_id % self.trace_every == 0

    def write_trace(self, event: Mapping[str, Any]) -> bool:
        rollout_id = int(event.get("rollout_id") or 0)
        if not self.trace_due(rollout_id):
            return False
        return self._append("traces/traces.jsonl", {"type": "trace", **event})


def monitor_from_env(run_id: str, rank: int) -> MonitorWriter:
    enabled = os.environ.get("GRPO_MONITOR", "0").strip() == "1"
    return MonitorWriter(
        enabled=enabled,
        root_dir=os.environ.get("GRPO_MONITOR_DIR", "/data/GRPO/runs"),
        run_id=os.environ.get("GRPO_RUN_ID", run_id),
        rank=rank,
        step_every=_positive_int("GRPO_MONITOR_STEP_EVERY", 1),
        rollout_every=_positive_int("GRPO_MONITOR_ROLLOUT_EVERY", 1),
        trace_every=_positive_int("GRPO_TRACE_EVERY", 20),
    )
