"""Detached recommendation monitoring for the alpha baseline ablation.

The functions in this module consume the *already-computed* native SID8
per-token CE and the logits of the one training forward.  They never take
part in the loss graph and keep DDP communication outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from rec_pu.recommendation_pu_phase2 import RecPUPackedTarget
from rec_pu.sid8_rec_pu_integration import SIDComponentVocab, coerce_packed_target


IGNORE_INDEX = -100

CORE_METRIC_NAMES = (
    "a_rec_cot_body_ce",
    "b_rec_cot_gold_sid_ce",
    "c_rec_nocot_gold_sid_ce",
    "d_rec_gold_a_ce",
    "e_rec_gold_b_ce",
    "f_rec_gold_c_ce",
    "g_rec_tf_a_hit8",
    "h_rec_tf_a_hit32",
    "i_rec_tf_b_hit8",
    "j_rec_tf_c_hit8",
    "k_rec_tf_chain_32_8_8",
    "l_rec_cot_tf_a_hit32",
    "m_rec_nocot_tf_a_hit32",
    "n_rec_cot_tf_chain_32_8_8",
    "o_rec_nocot_tf_chain_32_8_8",
)

COUNTER_NAMES = (
    "rec_monitor_cot_segments",
    "rec_monitor_nocot_segments",
    "rec_monitor_cot_gold_positions",
    "rec_monitor_nocot_gold_positions",
    "rec_monitor_missing_gold",
    "rec_monitor_invalid_route",
)

ALL_METRIC_NAMES = CORE_METRIC_NAMES + COUNTER_NAMES
_INDEX = {name: index for index, name in enumerate(ALL_METRIC_NAMES)}


@dataclass(frozen=True)
class AlphaMonitorConfig:
    enabled: bool = False
    train_tf_enabled: bool = True
    train_tf_interval: int = 50

    def __post_init__(self) -> None:
        if self.train_tf_interval < 1:
            raise ValueError("alpha_train_tf_interval must be >= 1.")


def _finite_sum_count(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(flat)
    return torch.stack((torch.where(finite, flat, torch.zeros_like(flat)).sum(), finite.sum().to(torch.float64)))


def _add(stats: torch.Tensor, name: str, values: torch.Tensor) -> None:
    stats[_INDEX[name]].add_(_finite_sum_count(values))


def _counter(stats: torch.Tensor, name: str, value: int | torch.Tensor) -> None:
    stats[_INDEX[name], 0].add_(torch.as_tensor(value, dtype=torch.float64, device=stats.device))


def _route(target: RecPUPackedTarget) -> str | None:
    value = target.source_segment
    return value if value in {"recommendation_cot", "recommendation_nocot"} else None


def _topk_current_hits(
    logits: torch.Tensor, current_labels: torch.Tensor, component_ids: Sequence[int], k: int
) -> torch.Tensor:
    """Return current-gold membership in a legal same-level component top-k."""

    ids = torch.as_tensor(component_ids, dtype=torch.long, device=logits.device)
    component_logits = logits.detach().index_select(1, ids).float()
    top_ids = ids[torch.topk(component_logits, k=min(k, ids.numel()), dim=1).indices]
    return (top_ids == current_labels.detach().unsqueeze(1)).any(dim=1).to(dtype=torch.float32)


def collect_alpha_recommendation_monitor(
    *,
    base_per_token_ce: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rec_targets: Sequence[Sequence[RecPUPackedTarget | Mapping[str, Any]]] | None,
    sid_component_vocab: SIDComponentVocab,
    collect_tf: bool,
) -> torch.Tensor:
    """Return compact detached ``[metric, sum/count]`` statistics.

    ``base_per_token_ce`` is the pre-existing shifted native CE.  A target
    position is a shifted-logit position, exactly matching ``label - 1``.
    """

    stats = torch.zeros((len(ALL_METRIC_NAMES), 2), device=logits.device, dtype=torch.float64)
    if rec_targets is None:
        return stats
    if base_per_token_ce.shape != labels[:, 1:].shape:
        raise ValueError("base_per_token_ce must use shifted label geometry.")

    shift_labels = labels[:, 1:]
    shift_length = shift_labels.size(1)
    entries: dict[str, list[tuple[int, RecPUPackedTarget, str]]] = {"a": [], "b": [], "c": []}
    # Validate all final-SID targets together.  Do not synchronize once per
    # target while collecting detached training diagnostics.
    target_position_checks: list[tuple[int, tuple[int, int, int]]] = []
    with torch.no_grad():
        for row, raw_targets in enumerate(rec_targets):
            for raw in raw_targets:
                target = coerce_packed_target(raw)
                route = _route(target)
                if route is None:
                    _counter(stats, "rec_monitor_invalid_route", 1)
                    continue
                if not (0 <= target.segment_start < target.segment_end <= labels.size(1)):
                    _counter(stats, "rec_monitor_missing_gold", 1)
                    continue
                positions = (target.a_logit_position, target.b_logit_position, target.c_logit_position)
                if any(position < 0 or position >= shift_length for position in positions):
                    _counter(stats, "rec_monitor_missing_gold", 1)
                    continue
                target_position_checks.append((row, positions))
                _counter(stats, f"rec_monitor_{'cot' if route.endswith('cot') and not route.endswith('nocot') else 'nocot'}_segments", 1)
                # The causal shift index p predicts original label p+1.
                label_positions = torch.arange(shift_length, device=logits.device) + 1
                segment = (label_positions >= target.segment_start) & (label_positions < target.segment_end)
                valid = shift_labels[row] != IGNORE_INDEX
                final = torch.zeros_like(segment)
                final[torch.tensor(positions, device=logits.device)] = True
                if route == "recommendation_cot":
                    _add(stats, "a_rec_cot_body_ce", base_per_token_ce[row][segment & valid & ~final])
                    _add(stats, "b_rec_cot_gold_sid_ce", base_per_token_ce[row, torch.tensor(positions, device=logits.device)])
                else:
                    _add(stats, "c_rec_nocot_gold_sid_ce", base_per_token_ce[row, torch.tensor(positions, device=logits.device)])
                for level, position, metric in (
                    ("a", target.a_logit_position, "d_rec_gold_a_ce"),
                    ("b", target.b_logit_position, "e_rec_gold_b_ce"),
                    ("c", target.c_logit_position, "f_rec_gold_c_ce"),
                ):
                    entries[level].append((row, target, route))
                _counter(stats, f"rec_monitor_{'cot' if route == 'recommendation_cot' else 'nocot'}_gold_positions", 3)

        if target_position_checks:
            checked_rows = torch.tensor(
                [row for row, positions in target_position_checks for _ in positions], device=logits.device
            )
            checked_positions = torch.tensor(
                [position for _, positions in target_position_checks for position in positions], device=logits.device
            )
            if not bool((shift_labels[checked_rows, checked_positions] != IGNORE_INDEX).all()):
                raise ValueError("Alpha monitor found a final-SID target outside the supervised response span.")

        for level, metric in (("a", "d_rec_gold_a_ce"), ("b", "e_rec_gold_b_ce"), ("c", "f_rec_gold_c_ce")):
            rows = entries[level]
            if not rows:
                continue
            positions = torch.tensor(
                [getattr(target, f"{level}_logit_position") for _, target, _ in rows], device=logits.device
            )
            row_ids = torch.tensor([row for row, _, _ in rows], device=logits.device)
            _add(stats, metric, base_per_token_ce[row_ids, positions])

        if collect_tf:
            outcomes: dict[str, tuple[torch.Tensor, list[str]]] = {}
            for level, rows in entries.items():
                if not rows:
                    continue
                row_ids = torch.tensor([row for row, _, _ in rows], device=logits.device)
                positions = torch.tensor(
                    [getattr(target, f"{level}_logit_position") for _, target, _ in rows], device=logits.device
                )
                selected = logits[:, :-1][row_ids, positions]
                current = shift_labels[row_ids, positions]
                component_ids = sid_component_vocab.for_level(level)
                outcomes[level] = (_topk_current_hits(selected, current, component_ids, 32 if level == "a" else 8), [route for _, _, route in rows])
            if set(outcomes) == {"a", "b", "c"}:
                a32, routes = outcomes["a"]
                a8 = _topk_current_hits(
                    logits[:, :-1][torch.tensor([row for row, _, _ in entries["a"]], device=logits.device),
                                   torch.tensor([target.a_logit_position for _, target, _ in entries["a"]], device=logits.device)],
                    shift_labels[
                        torch.tensor([row for row, _, _ in entries["a"]], device=logits.device),
                        torch.tensor([target.a_logit_position for _, target, _ in entries["a"]], device=logits.device),
                    ], sid_component_vocab.a, 8,
                )
                b8, _ = outcomes["b"]
                c8, _ = outcomes["c"]
                _add(stats, "g_rec_tf_a_hit8", a8)
                _add(stats, "h_rec_tf_a_hit32", a32)
                _add(stats, "i_rec_tf_b_hit8", b8)
                _add(stats, "j_rec_tf_c_hit8", c8)
                chain = a32.bool() & b8.bool() & c8.bool()
                _add(stats, "k_rec_tf_chain_32_8_8", chain.float())
                cot = torch.tensor([route == "recommendation_cot" for route in routes], device=logits.device)
                nocot = ~cot
                _add(stats, "l_rec_cot_tf_a_hit32", a32[cot])
                _add(stats, "m_rec_nocot_tf_a_hit32", a32[nocot])
                _add(stats, "n_rec_cot_tf_chain_32_8_8", chain[cot].float())
                _add(stats, "o_rec_nocot_tf_chain_32_8_8", chain[nocot].float())
    return stats
