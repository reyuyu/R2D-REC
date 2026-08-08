"""Recommendation No-think multi-positive prefix-trie loss replacement.

This controller consumes the already-produced teacher-forced logits. It never
performs a model forward and returns a differentiable numerator delta so the
existing normalized SID-weighted CE remains the denominator authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ...extras.constants import IGNORE_INDEX


REC_MP_STAT_NAMES = (
    "segments",
    "group_size_sum",
    "position_count",
    "single_ce_sum",
    "trie_nll_sum",
    "delta_sum",
    "allowed_a_sum",
    "allowed_b_sum",
    "allowed_c_sum",
    "multi_a_count",
    "multi_b_count",
    "multi_c_count",
    "invariant_violation_max",
)
REC_MP_STAT_INDEX = {name: index for index, name in enumerate(REC_MP_STAT_NAMES)}
REC_MP_STAT_SIZE = len(REC_MP_STAT_NAMES)

SID_RE = re.compile(
    r"<\|(?P<domain>prod|video|living|ad)_begin\|>"
    r"(?P<a><s_a_\d+>)(?P<b><s_b_\d+>)(?P<c><s_c_\d+>)"
)


@dataclass
class RecommendationMultiPositiveResult:
    loss_delta: torch.Tensor
    statistics: torch.Tensor


class RecommendationMultiPositiveTrieController:
    """Build exact prefix sets from V3 metadata and replace SID CE by set NLL."""

    def __init__(self, tokenizer, data_args) -> None:
        self.enabled = bool(getattr(data_args, "recommendation_multi_positive_trie_enabled", False))
        self.sid_weight = float(
            getattr(data_args, "recommendation_final_sid_weight", getattr(data_args, "sid_token_weight", 8.0))
        )
        if self.sid_weight <= 0:
            raise ValueError("Recommendation multi-positive trie requires a positive SID weight.")
        self._tokenizer = tokenizer
        self._group_cache: dict[str, tuple[tuple[int, int, int, int], ...]] = {}

    def _token_id(self, token: str) -> int:
        try:
            token_id = self._tokenizer.convert_tokens_to_ids(token)
        except Exception as exc:  # pragma: no cover - tokenizer-specific failure
            raise ValueError(f"Cannot convert recommendation SID token {token!r}.") from exc
        if isinstance(token_id, (list, tuple)) or token_id is None or int(token_id) < 0:
            raise ValueError(f"Recommendation SID token is not a single vocabulary token: {token!r}")
        return int(token_id)

    def _parse_sid(self, value: str) -> tuple[int, int, int, int]:
        match = SID_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"Malformed recommendation gold SID: {value!r}")
        return (
            self._token_id(f"<|{match.group('domain')}_begin|>"),
            self._token_id(match.group("a")),
            self._token_id(match.group("b")),
            self._token_id(match.group("c")),
        )

    def _group_paths(self, metadata: Mapping[str, Any]) -> tuple[tuple[int, int, int, int], ...]:
        group_id = metadata.get("group_id")
        all_gold = metadata.get("all_gold_sids")
        group_size = metadata.get("group_size")
        current = metadata.get("current_gold_sid")
        if not isinstance(group_id, str) or not isinstance(all_gold, list) or not all_gold:
            raise ValueError(
                "Recommendation V3 metadata is missing group_id or all_gold_sids; "
                f"keys={sorted(metadata.keys())}, types="
                f"group_id={type(group_id).__name__}, all_gold={type(all_gold).__name__}, metadata={metadata!r}."
            )
        if not isinstance(group_size, int) or group_size != len(all_gold):
            raise ValueError("Recommendation V3 group_size does not match all_gold_sids.")
        if not isinstance(current, str) or current not in all_gold:
            raise ValueError("Recommendation V3 current_gold_sid is not present in all_gold_sids.")
        cached = self._group_cache.get(group_id)
        if cached is not None:
            parsed_paths = tuple(self._parse_sid(value) for value in all_gold)
            if parsed_paths != cached:
                raise ValueError("Recommendation V3 group metadata changed for an existing group_id.")
            if self._parse_sid(current) not in cached:
                raise ValueError("Cached recommendation group does not contain current_gold_sid.")
            return cached
        paths = tuple(self._parse_sid(value) for value in all_gold)
        if len(set(paths)) != len(paths):
            raise ValueError("Recommendation V3 all_gold_sids contains duplicate full SID paths.")
        domains = {path[0] for path in paths}
        if len(domains) != 1:
            raise ValueError("Recommendation V3 group contains multiple domain marker tokens.")
        self._group_cache[group_id] = paths
        return paths

    @staticmethod
    def _metadata_for_segment(sample_metadata: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
        if index >= len(sample_metadata):
            raise ValueError("Recommendation pack metadata is shorter than segment offsets.")
        metadata = sample_metadata[index]
        if not isinstance(metadata, Mapping):
            raise ValueError("Recommendation segment metadata must be a mapping.")
        nested = metadata.get("recommendation_multi_positive")
        # Accept the equivalent flat form produced by older tokenized caches.
        # It still requires all four V3 fields, so corrupt/missing metadata
        # remains a hard error when REC_G2 is enabled.
        if not isinstance(nested, Mapping):
            flat_keys = (
                "recommendation_group_id",
                "recommendation_group_size",
                "recommendation_all_gold_sids",
                "recommendation_current_gold_sid",
            )
            if all(key in metadata for key in flat_keys):
                nested = {
                    "group_id": metadata["recommendation_group_id"],
                    "group_size": metadata["recommendation_group_size"],
                    "all_gold_sids": metadata["recommendation_all_gold_sids"],
                    "current_gold_sid": metadata["recommendation_current_gold_sid"],
                }
        if not isinstance(nested, Mapping):
            raise ValueError(
                "REC_G2 requires recommendation_multi_positive metadata for every No-think segment; "
                f"received keys={sorted(metadata.keys())}."
            )
        normalized = dict(nested)
        flat_to_nested = {
            "group_id": "recommendation_group_id",
            "group_size": "recommendation_group_size",
            "all_gold_sids": "recommendation_all_gold_sids",
            "current_gold_sid": "recommendation_current_gold_sid",
        }
        for key, flat_key in flat_to_nested.items():
            # V3 caches can carry the raw field names either directly on the
            # segment metadata or inside recommendation_multi_positive.
            if key not in normalized and flat_key in normalized:
                normalized[key] = normalized[flat_key]
            if key not in normalized and flat_key in metadata:
                normalized[key] = metadata[flat_key]
        # Some older tokenized caches retained all gold paths but omitted the
        # provenance id. The full path set is still authoritative, so derive a
        # stable cache key instead of weakening any path or current-gold checks.
        if "group_id" not in normalized and isinstance(normalized.get("all_gold_sids"), list):
            payload = json.dumps(normalized["all_gold_sids"], ensure_ascii=False, separators=(",", ":"))
            normalized["group_id"] = "synthetic:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return normalized

    @staticmethod
    def _find_path_positions(labels: torch.Tensor, start: int, end: int, path: tuple[int, int, int, int]) -> tuple[int, int, int]:
        values = labels[start:end].tolist()
        target = list(path)
        starts = [offset for offset in range(0, max(0, len(values) - len(target) + 1)) if values[offset : offset + len(target)] == target]
        if len(starts) != 1:
            raise ValueError(
                "REC_G2 could not uniquely locate the current recommendation SID in supervised labels "
                f"(matches={len(starts)}, segment=({start},{end}))."
            )
        offset = starts[0]
        return start + offset + 1, start + offset + 2, start + offset + 3

    @staticmethod
    def _set_nll(position_logits: torch.Tensor, allowed: set[int]) -> torch.Tensor:
        if not allowed:
            raise ValueError("REC_G2 generated an empty prefix allowed set.")
        allowed_ids = torch.tensor(sorted(allowed), dtype=torch.long, device=position_logits.device)
        if int(allowed_ids.max().item()) >= position_logits.shape[-1]:
            raise ValueError("REC_G2 allowed SID token is outside model vocabulary.")
        values = position_logits.float()
        return torch.logsumexp(values, dim=-1) - torch.logsumexp(values.index_select(0, allowed_ids), dim=0)

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_metadata: Sequence[Mapping[str, Any]],
        segment_offsets: torch.Tensor,
        task_name: str,
        subtask_name: str,
        denominator: torch.Tensor,
    ) -> RecommendationMultiPositiveResult:
        statistics = torch.zeros(REC_MP_STAT_SIZE, dtype=torch.float32, device=logits.device)
        zero = logits.sum() * 0.0
        if not self.enabled or task_name != "recommendation" or subtask_name != "nocot":
            return RecommendationMultiPositiveResult(loss_delta=zero, statistics=statistics)
        if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
            raise ValueError("REC_G2 expects logits [batch, sequence, vocab] and labels [batch, sequence].")
        if logits.shape[0] != 1:
            raise ValueError("REC_G2 currently expects TaskPackCollator batch size 1.")
        if segment_offsets.ndim != 2 or segment_offsets.shape[-1] != 2:
            raise ValueError("REC_G2 requires segment_offsets with shape [segments, 2].")
        if denominator.detach().float().item() <= 0:
            raise ValueError("REC_G2 received a non-positive SID-weighted CE denominator.")

        delta_numerator = zero
        labels_row = labels[0]
        offsets = segment_offsets.detach().cpu().tolist()
        for segment_index, (start, end) in enumerate(offsets):
            metadata = self._metadata_for_segment(sample_metadata, segment_index)
            paths = self._group_paths(metadata)
            current_path = self._parse_sid(metadata["current_gold_sid"])
            positions = self._find_path_positions(labels_row, int(start), int(end), current_path)
            a_allowed = {path[1] for path in paths}
            b_allowed = {path[2] for path in paths if path[1] == current_path[1]}
            c_allowed = {path[3] for path in paths if path[1] == current_path[1] and path[2] == current_path[2]}
            allowed_sets = (a_allowed, b_allowed, c_allowed)
            single_values, trie_values = [], []
            for label_position, allowed in zip(positions, allowed_sets):
                logit_position = label_position - 1
                if logit_position < int(start):
                    raise ValueError("REC_G2 causal shift crossed a segment boundary.")
                position_logits = logits[0, logit_position]
                label_id = int(labels_row[label_position].item())
                if label_id == IGNORE_INDEX or label_id not in allowed:
                    raise ValueError("REC_G2 current gold component is not in its own prefix allowed set.")
                single_ce = -torch.log_softmax(position_logits.float(), dim=-1)[label_id]
                trie_nll = self._set_nll(position_logits, allowed)
                single_values.append(single_ce)
                trie_values.append(trie_nll)
                delta_numerator = delta_numerator + self.sid_weight * (trie_nll - single_ce)

            singles = torch.stack(single_values)
            tries = torch.stack(trie_values)
            deltas = tries - singles
            statistics[REC_MP_STAT_INDEX["segments"]] += 1
            statistics[REC_MP_STAT_INDEX["group_size_sum"]] += len(paths)
            statistics[REC_MP_STAT_INDEX["position_count"]] += 3
            statistics[REC_MP_STAT_INDEX["single_ce_sum"]] += singles.detach().sum()
            statistics[REC_MP_STAT_INDEX["trie_nll_sum"]] += tries.detach().sum()
            statistics[REC_MP_STAT_INDEX["delta_sum"]] += deltas.detach().sum()
            statistics[REC_MP_STAT_INDEX["allowed_a_sum"]] += len(a_allowed)
            statistics[REC_MP_STAT_INDEX["allowed_b_sum"]] += len(b_allowed)
            statistics[REC_MP_STAT_INDEX["allowed_c_sum"]] += len(c_allowed)
            statistics[REC_MP_STAT_INDEX["multi_a_count"]] += int(len(a_allowed) > 1)
            statistics[REC_MP_STAT_INDEX["multi_b_count"]] += int(len(b_allowed) > 1)
            statistics[REC_MP_STAT_INDEX["multi_c_count"]] += int(len(c_allowed) > 1)
            statistics[REC_MP_STAT_INDEX["invariant_violation_max"]] = torch.maximum(
                statistics[REC_MP_STAT_INDEX["invariant_violation_max"]], deltas.detach().clamp_min(0.0).max()
            )
        return RecommendationMultiPositiveResult(
            loss_delta=delta_numerator / denominator.detach().clamp_min(1.0),
            statistics=statistics,
        )


def recommendation_multi_positive_statistics_to_metrics(statistics: torch.Tensor) -> dict[str, float]:
    values = statistics.detach().float().cpu()
    get = lambda name: float(values[REC_MP_STAT_INDEX[name]].item())
    segments = get("segments")
    positions = get("position_count")
    return {
        "rec_mp_segments": segments,
        "rec_mp_group_size": get("group_size_sum") / segments if segments else 0.0,
        "rec_mp_single_ce": get("single_ce_sum") / positions if positions else 0.0,
        "rec_mp_trie_nll": get("trie_nll_sum") / positions if positions else 0.0,
        "rec_mp_delta": get("delta_sum") / positions if positions else 0.0,
        "rec_mp_allowed_a": get("allowed_a_sum") / segments if segments else 0.0,
        "rec_mp_allowed_b": get("allowed_b_sum") / segments if segments else 0.0,
        "rec_mp_allowed_c": get("allowed_c_sum") / segments if segments else 0.0,
        "rec_mp_multi_a_ratio": get("multi_a_count") / segments if segments else 0.0,
        "rec_mp_multi_b_ratio": get("multi_b_count") / segments if segments else 0.0,
        "rec_mp_multi_c_ratio": get("multi_c_count") / segments if segments else 0.0,
        "rec_mp_invariant_violation_max": get("invariant_violation_max"),
    }
