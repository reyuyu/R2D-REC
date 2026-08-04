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
    "effective_trie_sum",
    "effective_length_sum",
    "trie_cap_active_sum",
    "length_cap_active_sum",
    "total_cap_scale_sum",
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


@dataclass
class _AuxiliaryComposition:
    loss: torch.Tensor
    ratio: torch.Tensor
    effective_trie: torch.Tensor
    effective_length: torch.Tensor
    trie_cap_scale: torch.Tensor
    length_cap_scale: torch.Tensor
    total_cap_scale: torch.Tensor
    warmup: float


@dataclass
class _TrieGroupPlan:
    positions: torch.Tensor
    allowed_ids: torch.Tensor
    allowed_mask: torch.Tensor
    sid_ids: torch.Tensor


@dataclass
class ActionAuxBatchPlan:
    valid_segment_count: int
    trie_groups: dict[str, _TrieGroupPlan]
    trie_sid_segment_ids: torch.Tensor
    boundary_segment_ids: torch.Tensor
    continue_ce_positions: torch.Tensor
    continue_ce_weights: torch.Tensor
    continue_ce_boundary_ids: torch.Tensor
    early_stop_positions: torch.Tensor
    early_stop_token_ids: torch.Tensor
    early_stop_boundary_ids: torch.Tensor
    tail_check_positions: torch.Tensor
    tail_check_segment_ids: torch.Tensor
    tail_positions: torch.Tensor
    tail_segment_ids: torch.Tensor


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
        self.split_cap_enabled = bool(getattr(data_args, "user_action_aux_split_cap_enabled", False))
        self.trie_cap_ratio = float(getattr(data_args, "user_action_trie_cap_ratio", 0.06))
        self.length_cap_ratio = float(getattr(data_args, "user_action_length_cap_ratio", 0.02))
        self.warmup_steps = int(data_args.user_action_aux_warmup_steps)
        self.vectorized_enabled = bool(getattr(data_args, "user_action_aux_vectorized_enabled", False))
        self.full_vocab_chunk_size = int(getattr(data_args, "user_action_aux_full_vocab_chunk_size", 64))
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

    def _cap_loss(
        self,
        value: torch.Tensor,
        action_ce: torch.Tensor,
        cap_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detached = value.detach().float().clamp_min(0.0)
        maximum = cap_ratio * action_ce.detach().float().clamp_min(0.0)
        one = torch.ones((), dtype=torch.float32, device=value.device)
        scale = torch.where(
            detached > maximum,
            maximum / (detached + self.eps),
            one,
        )
        return value * scale.to(value.dtype), scale

    def _compose_auxiliary(
        self,
        trie_loss: torch.Tensor,
        continue_loss: torch.Tensor,
        stop_loss: torch.Tensor,
        action_ce: torch.Tensor,
        macro_step: int,
    ) -> _AuxiliaryComposition:
        trie_raw = self.trie_weight * trie_loss
        length_raw = continue_loss + stop_loss
        one = torch.ones((), dtype=torch.float32, device=action_ce.device)
        if self.split_cap_enabled:
            effective_trie, trie_cap_scale = self._cap_loss(trie_raw, action_ce, self.trie_cap_ratio)
            effective_length, length_cap_scale = self._cap_loss(length_raw, action_ce, self.length_cap_ratio)
            combined = effective_trie + effective_length
            combined, total_cap_scale = self._cap_loss(combined, action_ce, self.cap_ratio)
            effective_trie = effective_trie * total_cap_scale.to(effective_trie.dtype)
            effective_length = effective_length * total_cap_scale.to(effective_length.dtype)
        else:
            combined, total_cap_scale = self._cap_loss(trie_raw + length_raw, action_ce, self.cap_ratio)
            effective_trie = trie_raw * total_cap_scale.to(trie_raw.dtype)
            effective_length = length_raw * total_cap_scale.to(length_raw.dtype)
            trie_cap_scale = one
            length_cap_scale = one

        warmup = self.warmup_factor(macro_step, self.warmup_steps)
        effective_trie = effective_trie * warmup
        effective_length = effective_length * warmup
        loss = combined * warmup
        ratio = loss.detach().float() / (action_ce.detach().float().abs() + self.eps)
        return _AuxiliaryComposition(
            loss=loss,
            ratio=ratio,
            effective_trie=effective_trie,
            effective_length=effective_length,
            trie_cap_scale=trie_cap_scale,
            length_cap_scale=length_cap_scale,
            total_cap_scale=total_cap_scale,
            warmup=warmup,
        )

    def _finalize_result(
        self,
        stats: torch.Tensor,
        trie_loss: torch.Tensor,
        continue_loss: torch.Tensor,
        stop_loss: torch.Tensor,
        action_ce: torch.Tensor,
        macro_step: int,
    ) -> ActionAuxiliaryResult:
        composition = self._compose_auxiliary(
            trie_loss,
            continue_loss,
            stop_loss,
            action_ce,
            macro_step,
        )
        stats[ACTION_STAT_INDEX["trie_loss_sum"]] = trie_loss.detach()
        stats[ACTION_STAT_INDEX["continue_loss_sum"]] = continue_loss.detach()
        stats[ACTION_STAT_INDEX["stop_loss_sum"]] = stop_loss.detach()
        stats[ACTION_STAT_INDEX["aux_loss_sum"]] = composition.loss.detach()
        stats[ACTION_STAT_INDEX["action_ce_sum"]] = action_ce.detach().float()
        stats[ACTION_STAT_INDEX["aux_to_ce_sum"]] = composition.ratio
        stats[ACTION_STAT_INDEX["cap_active_sum"]] = float(
            bool((composition.total_cap_scale.detach() < 1.0).item())
        )
        stats[ACTION_STAT_INDEX["effective_trie_sum"]] = composition.effective_trie.detach().float()
        stats[ACTION_STAT_INDEX["effective_length_sum"]] = composition.effective_length.detach().float()
        stats[ACTION_STAT_INDEX["trie_cap_active_sum"]] = float(
            bool((composition.trie_cap_scale.detach() < 1.0).item())
        )
        stats[ACTION_STAT_INDEX["length_cap_active_sum"]] = float(
            bool((composition.length_cap_scale.detach() < 1.0).item())
        )
        stats[ACTION_STAT_INDEX["total_cap_scale_sum"]] = composition.total_cap_scale.detach().float()
        stats[ACTION_STAT_INDEX["warmup_sum"]] = composition.warmup
        stats[ACTION_STAT_INDEX["user_base_sum"]] = action_ce.detach().float()
        stats[ACTION_STAT_INDEX["user_aux_sum"]] = composition.loss.detach().float()
        stats[ACTION_STAT_INDEX["user_total_sum"]] = (action_ce + composition.loss).detach().float()
        stats[ACTION_STAT_INDEX["action_microbatches"]] = 1
        return ActionAuxiliaryResult(loss=composition.loss.to(action_ce.dtype), statistics=stats)

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

    def _compute_legacy(
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
        return self._finalize_result(stats, trie_loss, continue_loss, stop_loss, action_ce, macro_step)

    @staticmethod
    def _device_tensor(values, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    def _build_vectorized_plan(
        self,
        logits: torch.Tensor,
        valid: list[dict[str, Any]],
        stats: torch.Tensor,
    ) -> ActionAuxBatchPlan:
        device = logits.device
        sequence_length = logits.shape[-2]
        trie_entries = {
            name: {"positions": [], "allowed_ids": [], "sid_ids": []}
            for name in ("domain", "a", "b", "c")
        }
        sid_segment_ids: list[int] = []
        boundary_segment_ids: list[int] = []
        continue_ce_positions: list[int] = []
        continue_ce_weights: list[float] = []
        continue_ce_boundary_ids: list[int] = []
        early_stop_positions: list[int] = []
        early_stop_token_ids: list[int] = []
        early_stop_boundary_ids: list[int] = []
        tail_check_positions: list[int] = []
        tail_check_segment_ids: list[int] = []
        tail_positions: list[int] = []
        tail_segment_ids: list[int] = []

        def valid_position(position: int) -> bool:
            return 0 < position < sequence_length

        for segment_id, metadata in enumerate(valid):
            history = {tuple(int(value) for value in sid) for sid in metadata["history_sids"]}
            units = metadata["answer_sid_units"]
            duplicate = bool(metadata.get("gold_duplicate", False))
            # Preserve the legacy metric even if malformed metadata repeats a history SID.
            stats[ACTION_STAT_INDEX["history_sid_sum"]] += len(metadata["history_sids"])
            stats[ACTION_STAT_INDEX["gold_sid_sum"]] += len(units)
            stats[ACTION_STAT_INDEX["duplicate_segments"]] += float(duplicate)

            if self.trie_enabled:
                used: set[tuple[int, int, int, int]] = set()
                for unit in units:
                    gold = tuple(int(value) for value in unit["value"])
                    stats[ACTION_STAT_INDEX["gold_total"]] += 1
                    if gold not in history:
                        continue
                    stats[ACTION_STAT_INDEX["gold_in_history"]] += 1
                    available = history if duplicate else history - used
                    sid_id = len(sid_segment_ids)
                    sid_segment_ids.append(segment_id)
                    for position_name, type_name, allowed_ids in zip(
                        ("pos_domain", "pos_a", "pos_b", "pos_c"),
                        ("domain", "a", "b", "c"),
                        self._allowed_sets(available, gold),
                    ):
                        position = int(unit[position_name])
                        if not allowed_ids or not valid_position(position):
                            continue
                        entry = trie_entries[type_name]
                        entry["positions"].append(position)
                        entry["allowed_ids"].append(sorted(allowed_ids))
                        entry["sid_ids"].append(sid_id)
                    if not duplicate:
                        used.add(gold)
                if not duplicate and len(units) > 1:
                    prior_legal = {tuple(unit["value"]) for unit in units[:-1]} & history
                    stats[ACTION_STAT_INDEX["removed_sid_sum"]] += len(prior_legal)
            else:
                stats[ACTION_STAT_INDEX["gold_total"]] += len(units)
                stats[ACTION_STAT_INDEX["gold_in_history"]] += sum(
                    tuple(unit["value"]) in history for unit in units
                )

            if not self.length_guard_enabled:
                continue
            boundaries = metadata["continuation_boundaries"]
            stats[ACTION_STAT_INDEX["continue_boundary_sum"]] += len(boundaries)
            for boundary in boundaries:
                boundary_id = len(boundary_segment_ids)
                boundary_segment_ids.append(segment_id)
                next_domain_pos = int(boundary["next_domain_pos"])
                if valid_position(next_domain_pos):
                    continue_ce_positions.append(next_domain_pos)
                    continue_ce_weights.append(self.continue_domain_extra)
                    continue_ce_boundary_ids.append(boundary_id)
                continue_pos = boundary.get("continue_token_pos")
                if continue_pos is not None:
                    continue_pos = int(continue_pos)
                    if continue_pos != next_domain_pos and valid_position(continue_pos):
                        continue_ce_positions.append(continue_pos)
                        continue_ce_weights.append(self.continue_separator_extra)
                        continue_ce_boundary_ids.append(boundary_id)
                    stop_token_id = boundary.get("stop_token_id")
                    if stop_token_id is not None and valid_position(continue_pos):
                        early_stop_positions.append(continue_pos)
                        early_stop_token_ids.append(int(stop_token_id))
                        early_stop_boundary_ids.append(boundary_id)

            all_tail_positions = [
                int(position)
                for position in metadata["final_tail_positions"]
                if 0 <= int(position) < sequence_length
            ]
            tail_check_positions.extend(all_tail_positions)
            tail_check_segment_ids.extend([segment_id] * len(all_tail_positions))
            selected_tail = [position for position in all_tail_positions if valid_position(position)][
                : self.max_stop_tail_positions
            ]
            tail_positions.extend(selected_tail)
            tail_segment_ids.extend([segment_id] * len(selected_tail))

        trie_groups: dict[str, _TrieGroupPlan] = {}
        for type_name, entry in trie_entries.items():
            allowed_lists = entry["allowed_ids"]
            max_allowed = max((len(values) for values in allowed_lists), default=0)
            padded = [values + [0] * (max_allowed - len(values)) for values in allowed_lists]
            masks = [[True] * len(values) + [False] * (max_allowed - len(values)) for values in allowed_lists]
            if max_allowed:
                allowed_ids = self._device_tensor(padded, dtype=torch.long, device=device).reshape(
                    len(padded), max_allowed
                )
                allowed_mask = self._device_tensor(masks, dtype=torch.bool, device=device).reshape(
                    len(masks), max_allowed
                )
            else:
                allowed_ids = torch.empty((0, 0), dtype=torch.long, device=device)
                allowed_mask = torch.empty((0, 0), dtype=torch.bool, device=device)
            trie_groups[type_name] = _TrieGroupPlan(
                positions=self._device_tensor(entry["positions"], dtype=torch.long, device=device),
                allowed_ids=allowed_ids,
                allowed_mask=allowed_mask,
                sid_ids=self._device_tensor(entry["sid_ids"], dtype=torch.long, device=device),
            )

        return ActionAuxBatchPlan(
            valid_segment_count=len(valid),
            trie_groups=trie_groups,
            trie_sid_segment_ids=self._device_tensor(sid_segment_ids, dtype=torch.long, device=device),
            boundary_segment_ids=self._device_tensor(boundary_segment_ids, dtype=torch.long, device=device),
            continue_ce_positions=self._device_tensor(continue_ce_positions, dtype=torch.long, device=device),
            continue_ce_weights=self._device_tensor(continue_ce_weights, dtype=torch.float32, device=device),
            continue_ce_boundary_ids=self._device_tensor(
                continue_ce_boundary_ids, dtype=torch.long, device=device
            ),
            early_stop_positions=self._device_tensor(early_stop_positions, dtype=torch.long, device=device),
            early_stop_token_ids=self._device_tensor(early_stop_token_ids, dtype=torch.long, device=device),
            early_stop_boundary_ids=self._device_tensor(
                early_stop_boundary_ids, dtype=torch.long, device=device
            ),
            tail_check_positions=self._device_tensor(tail_check_positions, dtype=torch.long, device=device),
            tail_check_segment_ids=self._device_tensor(
                tail_check_segment_ids, dtype=torch.long, device=device
            ),
            tail_positions=self._device_tensor(tail_positions, dtype=torch.long, device=device),
            tail_segment_ids=self._device_tensor(tail_segment_ids, dtype=torch.long, device=device),
        )

    def _group_mean(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        group_ids: torch.Tensor,
        group_count: int,
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sums = torch.zeros(group_count, dtype=torch.float32, device=zero.device) + zero
        counts = torch.zeros(group_count, dtype=torch.float32, device=zero.device)
        if values.numel():
            finite_values = torch.where(valid_mask, values.float(), torch.zeros_like(values, dtype=torch.float32))
            sums = sums.index_add(0, group_ids, finite_values)
            counts = counts.index_add(0, group_ids, valid_mask.float())
        means = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums) + zero)
        return means, counts

    def _vectorized_trie(
        self,
        logits: torch.Tensor,
        plan: ActionAuxBatchPlan,
        stats: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        sid_count = int(plan.trie_sid_segment_ids.numel())
        sid_sums = torch.zeros(sid_count, dtype=torch.float32, device=logits.device) + zero
        sid_counts = torch.zeros(sid_count, dtype=torch.float32, device=logits.device)
        sequence_logits = logits[0] if logits.ndim == 3 else logits
        for type_name, group in plan.trie_groups.items():
            losses: list[torch.Tensor] = []
            masses: list[torch.Tensor] = []
            for start in range(0, group.positions.numel(), self.full_vocab_chunk_size):
                end = start + self.full_vocab_chunk_size
                logit_positions = group.positions[start:end] - 1
                type_ids = self._type_ids(type_name, logits.device)
                type_logits = sequence_logits[
                    logit_positions.unsqueeze(1), type_ids.unsqueeze(0)
                ].float()
                allowed_ids = group.allowed_ids[start:end]
                allowed_logits = sequence_logits[
                    logit_positions.unsqueeze(1), allowed_ids
                ].float()
                allowed_logits = allowed_logits.masked_fill(~group.allowed_mask[start:end], -torch.inf)
                type_lse = torch.logsumexp(type_logits, dim=1)
                allowed_lse = torch.logsumexp(allowed_logits, dim=1)
                chunk_loss = type_lse - allowed_lse
                chunk_mass = torch.exp(allowed_lse - type_lse).clamp(0.0, 1.0)
                losses.append(chunk_loss)
                masses.append(chunk_mass)
            if not losses:
                continue
            loss = torch.cat(losses)
            mass = torch.cat(masses)
            finite = torch.isfinite(loss.detach()) & torch.isfinite(mass.detach())
            sid_sums = sid_sums.index_add(
                0,
                group.sid_ids,
                torch.where(finite, loss, torch.zeros_like(loss)),
            )
            sid_counts = sid_counts.index_add(0, group.sid_ids, finite.float())
            stats[ACTION_STAT_INDEX[f"allowed_{type_name}_sum"]] += torch.where(
                finite, mass.detach(), torch.zeros_like(mass)
            ).sum()
            stats[ACTION_STAT_INDEX[f"allowed_{type_name}_count"]] += finite.sum()

        sid_valid = sid_counts > 0
        sid_means = torch.where(
            sid_valid,
            sid_sums / sid_counts.clamp_min(1.0),
            torch.zeros_like(sid_sums) + zero,
        )
        segment_means, _ = self._group_mean(
            sid_means,
            sid_valid,
            plan.trie_sid_segment_ids,
            plan.valid_segment_count,
            zero,
        )
        return segment_means.mean() if segment_means.numel() else zero

    def _full_vocab_log_z(
        self,
        logits: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        sequence_logits = logits[0] if logits.ndim == 3 else logits
        chunks: list[torch.Tensor] = []
        for start in range(0, target_positions.numel(), self.full_vocab_chunk_size):
            rows = sequence_logits.index_select(0, target_positions[start : start + self.full_vocab_chunk_size] - 1)
            chunks.append(torch.logsumexp(rows.float(), dim=1))
        return torch.cat(chunks) if chunks else sequence_logits.reshape(-1)[0].float().reshape(1)[:0]

    def _vectorized_length_guard(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        plan: ActionAuxBatchPlan,
        stats: torch.Tensor,
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        segment_count = plan.valid_segment_count
        if segment_count == 0:
            return zero, zero
        event_positions = torch.cat(
            (
                plan.continue_ce_positions,
                plan.early_stop_positions,
                plan.tail_positions,
                plan.tail_positions,
            )
        )
        if event_positions.numel():
            unique_positions, inverse = torch.unique(event_positions, sorted=True, return_inverse=True)
            log_z = self._full_vocab_log_z(logits, unique_positions)
        else:
            inverse = torch.empty(0, dtype=torch.long, device=logits.device)
            log_z = torch.empty(0, dtype=torch.float32, device=logits.device)

        sizes = (
            plan.continue_ce_positions.numel(),
            plan.early_stop_positions.numel(),
            plan.tail_positions.numel(),
            plan.tail_positions.numel(),
        )
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + int(size))
        ce_rows = inverse[offsets[0] : offsets[1]]
        early_rows = inverse[offsets[1] : offsets[2]]
        tail_rows = inverse[offsets[2] : offsets[3]]
        domain_rows = inverse[offsets[3] : offsets[4]]
        sequence_logits = logits[0] if logits.ndim == 3 else logits
        label_row = labels[0] if labels.ndim == 2 else labels
        vocab_size = sequence_logits.shape[-1]

        continue_values: list[torch.Tensor] = []
        continue_valid: list[torch.Tensor] = []
        continue_group_ids: list[torch.Tensor] = []
        if plan.continue_ce_positions.numel():
            targets = label_row.index_select(0, plan.continue_ce_positions)
            safe_targets = targets.clamp(0, vocab_size - 1)
            gold_logits = sequence_logits[
                plan.continue_ce_positions - 1,
                safe_targets,
            ].float()
            ce = log_z.index_select(0, ce_rows) - gold_logits
            finite = (targets >= 0) & (targets < vocab_size) & torch.isfinite(ce.detach())
            continue_values.append(ce * plan.continue_ce_weights)
            continue_valid.append(finite)
            continue_group_ids.append(plan.continue_ce_boundary_ids)

        if plan.early_stop_positions.numel():
            safe_stop_ids = plan.early_stop_token_ids.clamp(0, vocab_size - 1)
            stop_logits = sequence_logits[
                plan.early_stop_positions - 1,
                safe_stop_ids,
            ].float()
            stop_mass = torch.exp(stop_logits - log_z.index_select(0, early_rows)).clamp(
                0.0, 1.0 - self.eps
            )
            early_loss = -torch.log1p(-stop_mass)
            finite = (
                (plan.early_stop_token_ids >= 0)
                & (plan.early_stop_token_ids < vocab_size)
                & torch.isfinite(early_loss.detach())
                & torch.isfinite(stop_mass.detach())
            )
            continue_values.append(self.no_early_stop_weight * early_loss)
            continue_valid.append(finite)
            continue_group_ids.append(plan.early_stop_boundary_ids)
            stats[ACTION_STAT_INDEX["no_early_stop_mass_sum"]] += torch.where(
                finite, stop_mass.detach(), torch.zeros_like(stop_mass)
            ).sum()
            stats[ACTION_STAT_INDEX["no_early_stop_mass_count"]] += finite.sum()

        boundary_count = int(plan.boundary_segment_ids.numel())
        boundary_sums = torch.zeros(boundary_count, dtype=torch.float32, device=logits.device) + zero
        boundary_counts = torch.zeros(boundary_count, dtype=torch.float32, device=logits.device)
        for values, finite, group_ids in zip(continue_values, continue_valid, continue_group_ids):
            boundary_sums = boundary_sums.index_add(
                0, group_ids, torch.where(finite, values, torch.zeros_like(values))
            )
            boundary_counts = boundary_counts.index_add(0, group_ids, finite.float())
        segment_continue, _ = self._group_mean(
            boundary_sums,
            boundary_counts > 0,
            plan.boundary_segment_ids,
            segment_count,
            zero,
        )

        has_tail_domain = torch.zeros(segment_count, dtype=torch.bool, device=logits.device)
        if plan.tail_check_positions.numel():
            tail_labels = label_row.index_select(0, plan.tail_check_positions)
            domain_ids = self._type_ids("domain", logits.device)
            is_domain = (tail_labels.unsqueeze(1) == domain_ids.unsqueeze(0)).any(dim=1)
            domain_counts = torch.zeros(segment_count, dtype=torch.long, device=logits.device).index_add(
                0, plan.tail_check_segment_ids, is_domain.long()
            )
            has_tail_domain = domain_counts > 0

        tail_losses = torch.empty(0, dtype=torch.float32, device=logits.device)
        tail_finite = torch.empty(0, dtype=torch.bool, device=logits.device)
        if plan.tail_positions.numel():
            tail_targets = label_row.index_select(0, plan.tail_positions)
            safe_targets = tail_targets.clamp(0, vocab_size - 1)
            tail_gold_logits = sequence_logits[plan.tail_positions - 1, safe_targets].float()
            tail_losses = log_z.index_select(0, tail_rows) - tail_gold_logits
            tail_finite = (
                (tail_targets >= 0)
                & (tail_targets < vocab_size)
                & ~has_tail_domain.index_select(0, plan.tail_segment_ids)
                & torch.isfinite(tail_losses.detach())
            )
        tail_means, _ = self._group_mean(
            tail_losses,
            tail_finite,
            plan.tail_segment_ids,
            segment_count,
            zero,
        )

        domain_losses = torch.empty(0, dtype=torch.float32, device=logits.device)
        domain_finite = torch.empty(0, dtype=torch.bool, device=logits.device)
        if plan.tail_positions.numel():
            domain_ids = self._type_ids("domain", logits.device)
            domain_logits = sequence_logits[
                (plan.tail_positions - 1).unsqueeze(1), domain_ids.unsqueeze(0)
            ].float()
            domain_log_mass = torch.logsumexp(domain_logits, dim=1) - log_z.index_select(0, domain_rows)
            domain_mass = torch.exp(domain_log_mass).clamp(0.0, 1.0 - self.eps)
            domain_losses = -torch.log1p(-domain_mass)
            domain_finite = (
                ~has_tail_domain.index_select(0, plan.tail_segment_ids)
                & torch.isfinite(domain_losses.detach())
                & torch.isfinite(domain_mass.detach())
            )
            stats[ACTION_STAT_INDEX["stop_domain_mass_sum"]] += torch.where(
                domain_finite, domain_mass.detach(), torch.zeros_like(domain_mass)
            ).sum()
            stats[ACTION_STAT_INDEX["stop_domain_mass_count"]] += domain_finite.sum()
        domain_means, _ = self._group_mean(
            domain_losses,
            domain_finite,
            plan.tail_segment_ids,
            segment_count,
            zero,
        )
        stop_per_segment = self.stop_tail_extra * tail_means + self.stop_domain_weight * domain_means
        return segment_continue.mean(), stop_per_segment.mean()

    def _compute_vectorized(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_metadata: list[dict[str, Any]],
        action_ce: torch.Tensor,
        macro_step: int,
    ) -> ActionAuxiliaryResult:
        # Macro packing currently produces one packed sequence. Keep the legacy
        # edge behavior for unsupported shapes instead of silently changing it.
        if logits.ndim not in (2, 3) or (logits.ndim == 3 and logits.shape[0] != 1):
            return self._compute_legacy(logits, labels, action_metadata, action_ce, macro_step)
        stats = torch.zeros(ACTION_STAT_SIZE, dtype=torch.float32, device=logits.device)
        stats[ACTION_STAT_INDEX["segments"]] = len(action_metadata)
        valid = [metadata for metadata in action_metadata if metadata.get("parse_valid")]
        stats[ACTION_STAT_INDEX["parse_ok"]] = len(valid)
        stats[ACTION_STAT_INDEX["valid_segments"]] = len(valid)
        stats[ACTION_STAT_INDEX["parse_ms_sum"]] = sum(
            float(metadata.get("parse_ms", 0.0)) for metadata in action_metadata
        )
        zero = logits.reshape(-1)[0].float() * 0.0
        plan = self._build_vectorized_plan(logits, valid, stats)
        trie_loss = self._vectorized_trie(logits, plan, stats, zero) if self.trie_enabled else zero
        if self.length_guard_enabled:
            continue_loss, stop_loss = self._vectorized_length_guard(logits, labels, plan, stats, zero)
        else:
            continue_loss, stop_loss = zero, zero
        return self._finalize_result(stats, trie_loss, continue_loss, stop_loss, action_ce, macro_step)

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_metadata: list[dict[str, Any]],
        action_ce: torch.Tensor,
        macro_step: int,
    ) -> ActionAuxiliaryResult:
        if self.vectorized_enabled:
            return self._compute_vectorized(logits, labels, action_metadata, action_ce, macro_step)
        return self._compute_legacy(logits, labels, action_metadata, action_ce, macro_step)


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
        "d_act_effective_trie_loss": values["effective_trie_sum"] / microbatches if microbatches else 0.0,
        "d_act_effective_length_loss": values["effective_length_sum"] / microbatches if microbatches else 0.0,
        "d_act_trie_cap_active": values["trie_cap_active_sum"] / microbatches if microbatches else 0.0,
        "d_act_length_cap_active": values["length_cap_active_sum"] / microbatches if microbatches else 0.0,
        "d_act_total_cap_scale": values["total_cap_scale_sum"] / microbatches if microbatches else 0.0,
        "d_act_warmup_factor": values["warmup_sum"] / microbatches if microbatches else 0.0,
    }
