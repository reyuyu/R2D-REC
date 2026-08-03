"""Action Select history-validity and length auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ...data.action_select import build_action_semantic_token_ids


ACTION_STAT_NAMES = (
    "segments",
    "parse_ok",
    "valid_segments",
    "history_sid_sum",
    "gold_sid_sum",
    "gold_in_history",
    "gold_total",
    "duplicate_segments",
    "allowed_domain_sum",
    "allowed_domain_count",
    "allowed_a_sum",
    "allowed_a_count",
    "allowed_b_sum",
    "allowed_b_count",
    "allowed_c_sum",
    "allowed_c_count",
    "removed_sid_sum",
    "continue_boundary_sum",
    "no_early_stop_mass_sum",
    "no_early_stop_mass_count",
    "stop_domain_mass_sum",
    "stop_domain_mass_count",
    "trie_loss_sum",
    "continue_loss_sum",
    "stop_loss_sum",
    "aux_loss_sum",
    "action_ce_sum",
    "aux_to_ce_sum",
    "cap_active_sum",
    "warmup_sum",
    "user_base_sum",
    "user_aux_sum",
    "user_total_sum",
    "action_microbatches",
    "parse_ms_sum",
)
ACTION_STAT_INDEX = {name: index for index, name in enumerate(ACTION_STAT_NAMES)}
ACTION_STAT_SIZE = len(ACTION_STAT_NAMES)


@dataclass
class ActionAuxiliaryResult:
    loss: torch.Tensor
    statistics: torch.Tensor


class UserActionAuxiliaryController:
    """Compute the optional objective without another model forward."""

    def __init__(self, tokenizer, data_args) -> None:
        semantic_ids = build_action_semantic_token_ids(tokenizer)
        self._semantic_cpu = {name: tuple(sorted(values)) for name, values in semantic_ids.items()}
        self._semantic_device: dict[tuple[str, torch.device], torch.Tensor] = {}
        self.trie_enabled = bool(data_args.user_action_history_trie_enabled)
        self.trie_weight = float(data_args.user_action_history_trie_weight)
        self.length_guard_enabled = bool(data_args.user_action_length_guard_enabled)
        self.continue_domain_extra = float(data_args.user_action_continue_domain_extra)
        self.continue_separator_extra = float(data_args.user_action_continue_separator_extra)
        self.no_early_stop_weight = float(data_args.user_action_no_early_stop_weight)
        self.stop_domain_weight = float(data_args.user_action_stop_domain_weight)
        self.stop_tail_extra = float(data_args.user_action_stop_tail_extra)
        self.max_stop_tail_positions = int(data_args.user_action_max_stop_tail_positions)
        self.cap_ratio = float(data_args.user_action_aux_cap_ratio)
        self.warmup_steps = int(data_args.user_action_aux_warmup_steps)
        self.eps = 1e-8

    def _type_ids(self, name: str, device: torch.device) -> torch.Tensor:
        key = (name, device)
        if key not in self._semantic_device:
            self._semantic_device[key] = torch.tensor(self._semantic_cpu[name], dtype=torch.long, device=device)
        return self._semantic_device[key]

    @staticmethod
    def warmup_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, max(0.0, float(step) / float(warmup_steps)))

    @staticmethod
    def _mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        finite = [value for value in values if bool(torch.isfinite(value.detach()).item())]
        return torch.stack(finite).mean() if finite else reference.reshape(-1)[0].float() * 0.0

    @staticmethod
    def _row(logits: torch.Tensor, target_position: int) -> torch.Tensor | None:
        # Causal logits at t predict the supervised token at t + 1.
        logit_position = int(target_position) - 1
        if logit_position < 0:
            return None
        if logits.ndim == 3:
            if logits.shape[0] != 1 or logit_position >= logits.shape[1]:
                return None
            return logits[0, logit_position].float()
        if logits.ndim == 2 and logit_position < logits.shape[0]:
            return logits[logit_position].float()
        return None

    def _allowed_mass_loss(
        self,
        logits: torch.Tensor,
        target_position: int,
        type_name: str,
        allowed_ids: set[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        row = self._row(logits, target_position)
        if row is None or not allowed_ids:
            return None
        type_ids = self._type_ids(type_name, row.device)
        allowed = torch.tensor(sorted(allowed_ids), dtype=torch.long, device=row.device)
        if type_ids.numel() == 0 or allowed.numel() == 0:
            return None
        type_lse = torch.logsumexp(row.index_select(0, type_ids), dim=0)
        allowed_lse = torch.logsumexp(row.index_select(0, allowed), dim=0)
        loss = type_lse - allowed_lse
        mass = torch.exp(allowed_lse - type_lse).clamp(0.0, 1.0)
        if not bool(torch.isfinite(loss.detach()).item()) or not bool(torch.isfinite(mass.detach()).item()):
            return None
        return loss, mass

    def _token_ce(self, logits: torch.Tensor, target_position: int, token_id: int) -> torch.Tensor | None:
        row = self._row(logits, target_position)
        if row is None or token_id < 0 or token_id >= row.shape[0]:
            return None
        loss = F.cross_entropy(row.unsqueeze(0), torch.tensor([token_id], device=row.device))
        return loss if bool(torch.isfinite(loss.detach()).item()) else None

    def _negative_mass(
        self, logits: torch.Tensor, target_position: int, negative_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        row = self._row(logits, target_position)
        if row is None or negative_ids.numel() == 0:
            return None
        negative_ids = negative_ids.to(row.device)
        log_mass = torch.logsumexp(row.index_select(0, negative_ids), dim=0) - torch.logsumexp(row, dim=0)
        mass = torch.exp(log_mass).clamp(0.0, 1.0 - self.eps)
        loss = -torch.log1p(-mass)
        if not bool(torch.isfinite(loss.detach()).item()) or not bool(torch.isfinite(mass.detach()).item()):
            return None
        return loss, mass

    @staticmethod
    def _allowed_sets(available: set[tuple[int, int, int, int]], gold: tuple[int, int, int, int]):
        domain, a_id, b_id, _ = gold
        return (
            {sid[0] for sid in available},
            {sid[1] for sid in available if sid[0] == domain},
            {sid[2] for sid in available if sid[:2] == (domain, a_id)},
            {sid[3] for sid in available if sid[:3] == (domain, a_id, b_id)},
        )

    def _segment_trie(
        self,
        logits: torch.Tensor,
        metadata: dict[str, Any],
        stats: torch.Tensor,
    ) -> torch.Tensor:
        reference = logits
        history = {tuple(int(value) for value in sid) for sid in metadata["history_sids"]}
        units = metadata["answer_sid_units"]
        duplicate = bool(metadata.get("gold_duplicate", False))
        used: set[tuple[int, int, int, int]] = set()
        sid_losses: list[torch.Tensor] = []
        for unit in units:
            gold = tuple(int(value) for value in unit["value"])
            stats[ACTION_STAT_INDEX["gold_total"]] += 1
            if gold not in history:
                continue
            stats[ACTION_STAT_INDEX["gold_in_history"]] += 1
            available = history if duplicate else history - used
            allowed_sets = self._allowed_sets(available, gold)
            position_names = ("pos_domain", "pos_a", "pos_b", "pos_c")
            type_names = ("domain", "a", "b", "c")
            position_losses: list[torch.Tensor] = []
            for position_name, type_name, allowed_ids in zip(position_names, type_names, allowed_sets):
                result = self._allowed_mass_loss(logits, int(unit[position_name]), type_name, allowed_ids)
                if result is None:
                    continue
                loss, mass = result
                position_losses.append(loss)
                stats[ACTION_STAT_INDEX[f"allowed_{type_name}_sum"]] += mass.detach()
                stats[ACTION_STAT_INDEX[f"allowed_{type_name}_count"]] += 1
            if position_losses:
                sid_losses.append(torch.stack(position_losses).mean())
            if not duplicate:
                used.add(gold)
        if not duplicate and len(units) > 1:
            prior_legal = {tuple(unit["value"]) for unit in units[:-1]} & history
            stats[ACTION_STAT_INDEX["removed_sid_sum"]] += len(prior_legal)
        return self._mean_or_zero(sid_losses, reference)

    def _segment_length_guard(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        metadata: dict[str, Any],
        stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = logits
        continue_losses: list[torch.Tensor] = []
        boundaries = metadata["continuation_boundaries"]
        stats[ACTION_STAT_INDEX["continue_boundary_sum"]] += len(boundaries)
        for boundary in boundaries:
            boundary_losses: list[torch.Tensor] = []
            next_domain_pos = int(boundary["next_domain_pos"])
            next_domain_ce = self._token_ce(logits, next_domain_pos, int(labels[0, next_domain_pos]))
            if next_domain_ce is not None:
                boundary_losses.append(self.continue_domain_extra * next_domain_ce)
            continue_pos = boundary.get("continue_token_pos")
            if continue_pos is not None and int(continue_pos) != next_domain_pos:
                separator_ce = self._token_ce(logits, int(continue_pos), int(labels[0, int(continue_pos)]))
                if separator_ce is not None:
                    boundary_losses.append(self.continue_separator_extra * separator_ce)
            stop_token_id = boundary.get("stop_token_id")
            if continue_pos is not None and stop_token_id is not None:
                negative_ids = torch.tensor([int(stop_token_id)], dtype=torch.long, device=logits.device)
                negative = self._negative_mass(logits, int(continue_pos), negative_ids)
                if negative is not None:
                    loss, mass = negative
                    boundary_losses.append(self.no_early_stop_weight * loss)
                    stats[ACTION_STAT_INDEX["no_early_stop_mass_sum"]] += mass.detach()
                    stats[ACTION_STAT_INDEX["no_early_stop_mass_count"]] += 1
            if boundary_losses:
                continue_losses.append(torch.stack(boundary_losses).sum())

        tail_positions = [int(position) for position in metadata["final_tail_positions"]]
        domain_ids = self._type_ids("domain", logits.device)
        has_tail_domain = any(int(labels[0, position]) in self._semantic_cpu["domain"] for position in tail_positions)
        stop_tail_losses: list[torch.Tensor] = []
        stop_domain_losses: list[torch.Tensor] = []
        if not has_tail_domain:
            for position in tail_positions[: self.max_stop_tail_positions]:
                tail_ce = self._token_ce(logits, position, int(labels[0, position]))
                if tail_ce is not None:
                    stop_tail_losses.append(tail_ce)
                negative = self._negative_mass(logits, position, domain_ids)
                if negative is not None:
                    loss, mass = negative
                    stop_domain_losses.append(loss)
                    stats[ACTION_STAT_INDEX["stop_domain_mass_sum"]] += mass.detach()
                    stats[ACTION_STAT_INDEX["stop_domain_mass_count"]] += 1
        stop_loss = (
            self.stop_tail_extra * self._mean_or_zero(stop_tail_losses, reference)
            + self.stop_domain_weight * self._mean_or_zero(stop_domain_losses, reference)
        )
        return self._mean_or_zero(continue_losses, reference), stop_loss

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_metadata: list[dict[str, Any]],
        action_ce: torch.Tensor,
        macro_step: int,
    ) -> ActionAuxiliaryResult:
        stats = torch.zeros(ACTION_STAT_SIZE, dtype=torch.float32, device=logits.device)
        stats[ACTION_STAT_INDEX["segments"]] = len(action_metadata)
        valid = [metadata for metadata in action_metadata if metadata.get("parse_valid")]
        stats[ACTION_STAT_INDEX["parse_ok"]] = len(valid)
        stats[ACTION_STAT_INDEX["valid_segments"]] = len(valid)
        stats[ACTION_STAT_INDEX["parse_ms_sum"]] = sum(
            float(metadata.get("parse_ms", 0.0)) for metadata in action_metadata
        )
        trie_losses: list[torch.Tensor] = []
        continue_losses: list[torch.Tensor] = []
        stop_losses: list[torch.Tensor] = []
        for metadata in valid:
            stats[ACTION_STAT_INDEX["history_sid_sum"]] += len(metadata["history_sids"])
            stats[ACTION_STAT_INDEX["gold_sid_sum"]] += len(metadata["answer_sid_units"])
            stats[ACTION_STAT_INDEX["duplicate_segments"]] += float(metadata.get("gold_duplicate", False))
            if self.trie_enabled:
                trie_losses.append(self._segment_trie(logits, metadata, stats))
            else:
                stats[ACTION_STAT_INDEX["gold_total"]] += len(metadata["answer_sid_units"])
                history = {tuple(sid) for sid in metadata["history_sids"]}
                stats[ACTION_STAT_INDEX["gold_in_history"]] += sum(
                    tuple(unit["value"]) in history for unit in metadata["answer_sid_units"]
                )
            if self.length_guard_enabled:
                continue_loss, stop_loss = self._segment_length_guard(logits, labels, metadata, stats)
                continue_losses.append(continue_loss)
                stop_losses.append(stop_loss)

        trie_loss = self._mean_or_zero(trie_losses, logits)
        continue_loss = self._mean_or_zero(continue_losses, logits)
        stop_loss = self._mean_or_zero(stop_losses, logits)
        aux_raw = self.trie_weight * trie_loss + continue_loss + stop_loss
        detached_raw = aux_raw.detach().clamp_min(0.0)
        max_aux = self.cap_ratio * action_ce.detach().float().clamp_min(0.0)
        cap_scale = torch.minimum(
            torch.ones((), device=logits.device),
            max_aux / (detached_raw + self.eps),
        )
        warmup = self.warmup_factor(macro_step, self.warmup_steps)
        aux_loss = aux_raw * cap_scale.to(aux_raw.dtype) * warmup
        ratio = aux_loss.detach().float() / (action_ce.detach().float().abs() + self.eps)
        stats[ACTION_STAT_INDEX["trie_loss_sum"]] = trie_loss.detach()
        stats[ACTION_STAT_INDEX["continue_loss_sum"]] = continue_loss.detach()
        stats[ACTION_STAT_INDEX["stop_loss_sum"]] = stop_loss.detach()
        stats[ACTION_STAT_INDEX["aux_loss_sum"]] = aux_loss.detach()
        stats[ACTION_STAT_INDEX["action_ce_sum"]] = action_ce.detach().float()
        stats[ACTION_STAT_INDEX["aux_to_ce_sum"]] = ratio
        stats[ACTION_STAT_INDEX["cap_active_sum"]] = float(bool((cap_scale.detach() < 1.0).item()))
        stats[ACTION_STAT_INDEX["warmup_sum"]] = warmup
        stats[ACTION_STAT_INDEX["user_base_sum"]] = action_ce.detach().float()
        stats[ACTION_STAT_INDEX["user_aux_sum"]] = aux_loss.detach().float()
        stats[ACTION_STAT_INDEX["user_total_sum"]] = (action_ce + aux_loss).detach().float()
        stats[ACTION_STAT_INDEX["action_microbatches"]] = 1
        return ActionAuxiliaryResult(loss=aux_loss.to(action_ce.dtype), statistics=stats)


def action_statistics_to_metrics(statistics: torch.Tensor) -> dict[str, float]:
    """Convert one globally reduced statistics vector into stable scalar metrics."""
    values = {name: float(statistics[index].item()) for index, name in enumerate(ACTION_STAT_NAMES)}

    def ratio(numerator: str, denominator: str) -> float:
        return values[numerator] / values[denominator] if values[denominator] else 0.0

    valid = values["valid_segments"]
    microbatches = values["action_microbatches"]
    return {
        "a_act_segments": values["segments"],
        "a_act_parse_ok_rate": ratio("parse_ok", "segments"),
        "a_act_gold_in_history_rate": ratio("gold_in_history", "gold_total"),
        "a_act_gold_duplicate_rate": values["duplicate_segments"] / valid if valid else 0.0,
        "b_act_allowed_domain_mass": ratio("allowed_domain_sum", "allowed_domain_count"),
        "b_act_allowed_a_mass": ratio("allowed_a_sum", "allowed_a_count"),
        "b_act_allowed_b_mass": ratio("allowed_b_sum", "allowed_b_count"),
        "b_act_allowed_c_mass": ratio("allowed_c_sum", "allowed_c_count"),
        "c_act_seen_sid_removed_avg": values["removed_sid_sum"] / valid if valid else 0.0,
        "c_act_no_early_stop_mass": ratio("no_early_stop_mass_sum", "no_early_stop_mass_count"),
        "c_act_stop_domain_mass": ratio("stop_domain_mass_sum", "stop_domain_mass_count"),
        "d_act_trie_loss": values["trie_loss_sum"] / microbatches if microbatches else 0.0,
        "d_act_continue_loss": values["continue_loss_sum"] / microbatches if microbatches else 0.0,
        "d_act_stop_loss": values["stop_loss_sum"] / microbatches if microbatches else 0.0,
        "d_act_aux_loss": values["aux_loss_sum"] / microbatches if microbatches else 0.0,
        "d_act_action_ce": values["action_ce_sum"] / microbatches if microbatches else 0.0,
        "d_act_aux_to_ce_ratio": values["aux_to_ce_sum"] / microbatches if microbatches else 0.0,
        "d_act_cap_active": values["cap_active_sum"] / microbatches if microbatches else 0.0,
        "d_act_warmup_factor": values["warmup_sum"] / microbatches if microbatches else 0.0,
    }
