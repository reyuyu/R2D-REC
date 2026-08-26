"""Detached rank-local monitoring capture and rank-zero JSON interfaces."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.distributed as dist

from rollout_runtime_v1 import BusinessGroupRollout, G
from truerec_loss_v1 import HPR_LAMBDA


POSITION_NAMES = ("A", "B", "C")
ROLLING_WINDOW = 50
TRAIN_GROUP_FIELDS = (
    "global_step", "group_index", "recommendation_group_id", "target_domain",
    "context_token_count", "selected_microbatch_size", "format_valid_rate",
    "A_hit_rate", "AB_hit_rate", "exact_rate", "wrong_history_copy_rate",
    "frontier_value", "hpr_value_raw", "hpr_value_weighted", "total_value",
    "gradient_norm", "wall_time_seconds", "rank_memory",
)
TOKEN_FIELDS = (
    "action_position", "position_name", "token_id", "token_string", "old_logp",
    "current_logp_pre_step", "ppo_ratio", "prefix_gate_active",
    "frontier_token_credit", "format_token_credit", "effective_signed_credit",
    "ppo_unclipped_surrogate", "ppo_clipped_surrogate", "ppo_selected_surrogate",
    "frontier_loss_contribution",
)


class MonitoringContractError(RuntimeError):
    pass


def detached_jsonable(value: Any) -> Any:
    """Detach tensor leaves and reject non-finite or unsupported monitoring values."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise MonitoringContractError("monitoring tensor is connected to autograd")
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): detached_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [detached_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise MonitoringContractError("monitoring contains NaN/Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise MonitoringContractError(f"unsupported monitoring value: {type(value).__name__}")


def capture_frontier_chunk(
    *, group: BusinessGroupRollout, start: int, stop: int, current: torch.Tensor,
    runtime, format_credits: torch.Tensor, hierarchy, format_loss,
) -> list[dict[str, Any]]:
    """Capture the exact tensors consumed by the two Frontier PPO loss calls."""
    rows = []
    for local_row, candidate_index in enumerate(range(start, stop)):
        candidate = group.candidates[candidate_index]
        tokens = []
        for action_position, token_id in enumerate(candidate.completion_ids):
            frontier_credit = runtime.token_credits[candidate_index, action_position]
            format_credit = format_credits[candidate_index, action_position]
            ratio = hierarchy.ratios[0, local_row, action_position]
            clipped_ratio = hierarchy.clipped_ratios[0, local_row, action_position]
            effective_credit = frontier_credit + format_credit
            selected = -(
                hierarchy.token_losses[0, local_row, action_position]
                + format_loss.token_losses[0, local_row, action_position]
            )
            token = {
                "action_position": action_position,
                "position_name": POSITION_NAMES[action_position],
                "token_id": int(token_id),
                "old_logp": float(candidate.old_logps[action_position]),
                "current_logp_pre_step": float(current[local_row, action_position].detach()),
                "ppo_ratio": float(ratio.detach()),
                "prefix_gate_active": bool(runtime.token_credit_mask[candidate_index, action_position]),
                "frontier_token_credit": float(frontier_credit),
                "format_token_credit": float(format_credit),
                "effective_signed_credit": float(effective_credit),
                "ppo_unclipped_surrogate": float((ratio * effective_credit).detach()),
                "ppo_clipped_surrogate": float((clipped_ratio * effective_credit).detach()),
                "ppo_selected_surrogate": float(selected.detach()),
                "frontier_loss_contribution": float((-selected / G).detach()),
            }
            tokens.append(token)
        rows.append({"candidate_index": candidate_index, "tokens": tokens})
    return detached_jsonable(rows)


def capture_hpr_position(
    *, site_index: int, site, candidate_index: int, action_position: int,
    position_loss: torch.Tensor, position_weight: float,
) -> dict[str, Any]:
    raw = position_loss * position_weight
    return detached_jsonable({
        "site_index": site_index,
        "target_position": site.level,
        "candidate_index": candidate_index,
        "action_position": action_position,
        "target_token_ids": list(site.target_token_ids),
        "multi_positive_log_mass": -position_loss.detach(),
        "hpr_position_loss": position_loss.detach(),
        "position_weight": position_weight,
        "raw_hpr_contribution": raw.detach(),
        "weighted_hpr_contribution": (HPR_LAMBDA * raw).detach(),
    })


def runtime_monitoring_snapshot(runtime) -> dict[str, Any]:
    return detached_jsonable({
        "hpr_trigger": runtime.hpr.trigger,
        "hpr_sites": [
            {
                "site_index": site_index,
                "target_position": site.level,
                "target_token_ids": list(site.target_token_ids),
                "onpolicy_positions": [list(item) for item in site.onpolicy_positions],
            }
            for site_index, site in enumerate(runtime.hpr.sites)
        ],
    })


def local_payload_from_backward(backward) -> dict[str, Any]:
    return detached_jsonable({
        "candidates": backward.local_candidate_monitoring,
        "hpr_positions": backward.local_hpr_monitoring,
    })


def gather_detached_monitoring(local_payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Gather graph-free Python objects; only rank zero receives the combined payload."""
    payload = detached_jsonable(local_payload)
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    return gathered if dist.get_rank() == 0 else None


def _trigger_reason(trigger: str) -> str:
    return {
        "HPR_A": "No candidate reached a Gold A prefix.",
        "HPR_B": "At least one Gold A was reached, but no Gold AB prefix was reached.",
        "HPR_C": "At least one Gold AB was reached, but no exact Gold ABC was reached.",
        "HPR_NONE": "At least one exact Gold ABC was reached; rescue is inactive.",
    }[trigger]


def build_train_explain(
    group: BusinessGroupRollout, runtime_snapshot: dict[str, Any], rank_payloads: Sequence[dict[str, Any]],
    id_to_token: Callable[[int], str],
) -> dict[str, Any]:
    candidate_parts: dict[int, dict[str, Any]] = {}
    hpr_parts: list[dict[str, Any]] = []
    for source_rank, payload in enumerate(rank_payloads):
        for item in payload["candidates"]:
            index = int(item["candidate_index"])
            if index in candidate_parts:
                raise MonitoringContractError("duplicate global candidate monitoring")
            candidate_parts[index] = {**item, "source_rank": source_rank}
        hpr_parts.extend(payload["hpr_positions"])
    if sorted(candidate_parts) != list(range(G)):
        raise MonitoringContractError("global candidate monitoring order is not 0..7")

    candidates = []
    for index, candidate in enumerate(group.candidates):
        captured = candidate_parts[index]
        token_rows = captured["tokens"]
        if len(token_rows) != len(candidate.completion_ids):
            raise MonitoringContractError("candidate token monitoring length mismatch")
        for token in token_rows:
            token["token_string"] = str(id_to_token(token["token_id"]))
            if set(token) != set(TOKEN_FIELDS):
                raise MonitoringContractError("token monitoring schema mismatch")
        candidates.append({
            "candidate_index": index,
            "source_rank": captured["source_rank"],
            "completion_ids": list(candidate.completion_ids),
            "completion_tokens": [str(id_to_token(value)) for value in candidate.completion_ids],
            "parsed_abc": candidate.metrics.get("parsed_abc"),
            "format_valid": bool(candidate.metrics["format_valid"]),
            "A_hit": bool(candidate.metrics["A_hit"]),
            "AB_hit": bool(candidate.metrics["AB_hit"]),
            "exact": bool(candidate.metrics["exact"]),
            "wrong_history_copy": bool(candidate.metrics.get("wrong_history_copy", False)),
            "action_tokens": token_rows,
        })

    positions_by_site: dict[int, list[dict[str, Any]]] = {}
    for item in hpr_parts:
        positions_by_site.setdefault(int(item["site_index"]), []).append(item)
    sites = []
    for site in runtime_snapshot["hpr_sites"]:
        site_index = int(site["site_index"])
        positions = sorted(
            positions_by_site.get(site_index, []),
            key=lambda item: (item["candidate_index"], item["action_position"]),
        )
        expected = sorted((int(row), int(position)) for row, position in site["onpolicy_positions"])
        actual = [(item["candidate_index"], item["action_position"]) for item in positions]
        if actual != expected:
            raise MonitoringContractError("HPR monitoring positions differ from global runtime plan")
        sites.append({
            "site_index": site_index,
            "target_position": site["target_position"],
            "target_token_ids": list(site["target_token_ids"]),
            "target_token_strings": [str(id_to_token(value)) for value in site["target_token_ids"]],
            "onpolicy_positions": positions,
        })
    return detached_jsonable({
        "mode": "TRAIN",
        "optimizer_update": True,
        "recommendation_group_id": group.recommendation_group_id,
        "global_candidate_indices": list(range(G)),
        "candidates": candidates,
        "hpr_trigger": runtime_snapshot["hpr_trigger"],
        "trigger_reason": _trigger_reason(runtime_snapshot["hpr_trigger"]),
        "hpr_sites": sites,
    })


def build_train_group_record(
    *, record: dict[str, Any], group: BusinessGroupRollout, backward,
    global_step: int, group_index: int, selected_microbatch_size: int,
    gradient_norm: float, wall_time_seconds: float, rank_memory: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [candidate.metrics for candidate in group.candidates]
    rank_memory = detached_jsonable(rank_memory)
    output = {
        "global_step": int(global_step),
        "group_index": int(group_index),
        "recommendation_group_id": group.recommendation_group_id,
        "target_domain": str(record["target_domain"]),
        "context_token_count": len(group.context_ids),
        "selected_microbatch_size": int(selected_microbatch_size),
        "format_valid_rate": sum(item["format_valid"] for item in metrics) / G,
        "A_hit_rate": sum(item["A_hit"] for item in metrics) / G,
        "AB_hit_rate": sum(item["AB_hit"] for item in metrics) / G,
        "exact_rate": sum(item["exact"] for item in metrics) / G,
        "wrong_history_copy_rate": sum(item.get("wrong_history_copy", False) for item in metrics) / G,
        "frontier_value": float(backward.global_frontier_value),
        "hpr_value_raw": float(backward.global_hpr_value_raw),
        "hpr_value_weighted": float(backward.global_hpr_value_weighted),
        "total_value": float(backward.global_total_value),
        "gradient_norm": float(gradient_norm),
        "wall_time_seconds": float(wall_time_seconds),
        "rank_memory": rank_memory,
        "max_rank_allocated_gb": max(item["allocated_gb"] for item in rank_memory),
        "max_rank_reserved_gb": max(item["reserved_gb"] for item in rank_memory),
        "max_rank_peak_allocated_gb": max(item["peak_allocated_gb"] for item in rank_memory),
        "max_rank_peak_reserved_gb": max(item["peak_reserved_gb"] for item in rank_memory),
    }
    if not set(TRAIN_GROUP_FIELDS).issubset(output):
        raise MonitoringContractError("train group monitoring schema incomplete")
    return detached_jsonable(output)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(detached_jsonable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(detached_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MonitoringWriterV1:
    """Rank-zero owner of append-only train records and atomic live/probe state."""

    def __init__(self, run_directory: Path, *, total_steps: int, rank: int) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.run_directory = Path(run_directory)
        self.total_steps = int(total_steps)
        self.rank = int(rank)
        existing = _read_jsonl(self.run_directory / "train_groups.jsonl") if self.rank == 0 else []
        self.history = deque(existing[-ROLLING_WINDOW:], maxlen=ROLLING_WINDOW)

    def record_train_group(
        self, group_record: dict[str, Any], explain_record: dict[str, Any],
        *, last_checkpoint_step: int | None,
    ) -> bool:
        if self.rank != 0:
            return False
        group_record = detached_jsonable(group_record)
        explain_record = detached_jsonable(explain_record)
        missing = set(TRAIN_GROUP_FIELDS) - set(group_record)
        if missing:
            raise MonitoringContractError(f"train group fields missing: {sorted(missing)}")
        if explain_record.get("global_candidate_indices") != list(range(G)):
            raise MonitoringContractError("train explain candidate order must be 0..7")
        if self.history and group_record["group_index"] != self.history[-1]["group_index"] + 1:
            raise MonitoringContractError("train JSONL append order is not contiguous")
        explain_record.update({
            "global_step": group_record["global_step"],
            "group_index": group_record["group_index"],
        })
        _append_jsonl(self.run_directory / "train_groups.jsonl", group_record)
        _append_jsonl(self.run_directory / "train_explain.jsonl", explain_record)
        self.history.append(group_record)
        self._write_live(group_record, last_checkpoint_step)
        return True

    def _write_live(self, current: dict[str, Any], last_checkpoint_step: int | None) -> None:
        rolling = list(self.history)
        mean_keys = (
            "format_valid_rate", "A_hit_rate", "AB_hit_rate", "exact_rate",
            "wrong_history_copy_rate", "frontier_value", "hpr_value_weighted", "total_value",
        )
        rolling_metrics = {
            key: sum(float(row[key]) for row in rolling) / len(rolling) for key in mean_keys
        }
        seconds = sum(float(row["wall_time_seconds"]) for row in rolling) / len(rolling)
        current_step = int(current["global_step"])
        rank_health = [
            {
                "rank": item["rank"], "healthy": bool(item.get("healthy", True)),
                "allocated_gb": item["allocated_gb"], "reserved_gb": item["reserved_gb"],
                "peak_allocated_gb": item["peak_allocated_gb"],
                "peak_reserved_gb": item["peak_reserved_gb"],
            }
            for item in current["rank_memory"]
        ]
        live = {
            "current_step": current_step,
            "total_steps": self.total_steps,
            "progress_percent": 100.0 * current_step / self.total_steps,
            "current_group_id": current["recommendation_group_id"],
            "domain": current["target_domain"],
            "context_token_count": current["context_token_count"],
            "selected_microbatch_size": current["selected_microbatch_size"],
            "total_loss": current["total_value"],
            "frontier_loss": current["frontier_value"],
            "hpr_weighted": current["hpr_value_weighted"],
            "A_hit_rate": current["A_hit_rate"],
            "AB_hit_rate": current["AB_hit_rate"],
            "exact_rate": current["exact_rate"],
            "rolling_50_metrics": rolling_metrics,
            "seconds_per_group": current["wall_time_seconds"],
            "rolling_50_seconds_per_group": seconds,
            "ETA_seconds": max(self.total_steps - current_step, 0) * seconds,
            "last_checkpoint_step": last_checkpoint_step,
            "rank_health": rank_health,
        }
        _atomic_json(self.run_directory / "live_state.json", live)

    def write_probe(
        self, probe_step: int, *, summary: dict[str, Any],
        groups: Iterable[dict[str, Any]], explains: Iterable[dict[str, Any]],
    ) -> bool:
        if self.rank != 0:
            return False
        directory = self.run_directory / "probe" / str(int(probe_step))
        summary = {**detached_jsonable(summary), "mode": "PROBE", "optimizer_update": False, "probe_step": int(probe_step)}
        _atomic_json(directory / "summary.json", summary)
        for value in groups:
            _append_jsonl(directory / "groups.jsonl", {**detached_jsonable(value), "mode": "PROBE", "optimizer_update": False, "probe_step": int(probe_step)})
        for value in explains:
            explain = {**detached_jsonable(value), "mode": "PROBE", "optimizer_update": False, "probe_step": int(probe_step)}
            _append_jsonl(directory / "explain.jsonl", explain)
        return True
