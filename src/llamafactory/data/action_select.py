"""One-time metadata parsing for the Action Select auxiliary objective."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..extras.constants import IGNORE_INDEX


DOMAIN_TOKEN_STRINGS = {
    "<|video_begin|>",
    "<|prod_begin|>",
    "<|ad_begin|>",
    "<|living_begin|>",
}


def build_action_semantic_token_ids(tokenizer) -> dict[str, frozenset[int]]:
    """Read the semantic token vocabularies from the active tokenizer once."""
    groups: dict[str, set[int]] = {"domain": set(), "a": set(), "b": set(), "c": set()}
    for token, token_id in tokenizer.get_added_vocab().items():
        token_id = int(token_id)
        if token in DOMAIN_TOKEN_STRINGS:
            groups["domain"].add(token_id)
        elif token.startswith("<s_a_") and token.endswith(">"):
            groups["a"].add(token_id)
        elif token.startswith("<s_b_") and token.endswith(">"):
            groups["b"].add(token_id)
        elif token.startswith("<s_c_") and token.endswith(">"):
            groups["c"].add(token_id)

    missing = [name for name, values in groups.items() if not values]
    if missing:
        raise ValueError(f"The active tokenizer is missing Action Select semantic token groups: {missing}.")
    return {name: frozenset(values) for name, values in groups.items()}


def _supervised_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    supervised = labels != IGNORE_INDEX
    padded = np.pad(supervised, (1, 1), constant_values=False)
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


class ActionSelectMetadataParser:
    """Parse complete history and answer SIDs without decoding text."""

    def __init__(self, tokenizer) -> None:
        self.semantic_ids = build_action_semantic_token_ids(tokenizer)
        self._all_semantic_ids = frozenset().union(*self.semantic_ids.values())
        max_token_id = max(self._all_semantic_ids)
        self._token_type_lookup = np.zeros(max_token_id + 1, dtype=np.uint8)
        for type_index, name in enumerate(("domain", "a", "b", "c"), start=1):
            self._token_type_lookup[list(self.semantic_ids[name])] = type_index
        self.parse_count = 0

    def _token_types(self, input_ids: np.ndarray) -> np.ndarray:
        types = np.zeros(input_ids.shape, dtype=np.uint8)
        valid = (input_ids >= 0) & (input_ids < self._token_type_lookup.size)
        types[valid] = self._token_type_lookup[input_ids[valid]]
        return types

    @staticmethod
    def _find_sid_starts(
        token_types: np.ndarray,
        labels: np.ndarray,
        start: int,
        end: int,
        supervised: bool,
    ) -> list[int]:
        if end - start < 4:
            return []
        typed = (
            (token_types[start : end - 3] == 1)
            & (token_types[start + 1 : end - 2] == 2)
            & (token_types[start + 2 : end - 1] == 3)
            & (token_types[start + 3 : end] == 4)
        )
        supervised_mask = labels != IGNORE_INDEX
        label_match = (
            supervised_mask[start : end - 3]
            & supervised_mask[start + 1 : end - 2]
            & supervised_mask[start + 2 : end - 1]
            & supervised_mask[start + 3 : end]
        )
        if not supervised:
            label_match = ~(
                supervised_mask[start : end - 3]
                | supervised_mask[start + 1 : end - 2]
                | supervised_mask[start + 2 : end - 1]
                | supervised_mask[start + 3 : end]
            )
        return [int(position + start) for position in np.flatnonzero(typed & label_match)]

    @staticmethod
    def _failure(started: float, reason: str) -> dict[str, Any]:
        return {
            "action_select": True,
            "parse_valid": False,
            "parse_error": reason,
            "parse_ms": (time.perf_counter() - started) * 1000.0,
            "history_sids": [],
            "answer_sid_units": [],
            "continuation_boundaries": [],
            "final_tail_positions": [],
        }

    def parse(self, input_ids: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
        started = time.perf_counter()
        self.parse_count += 1
        input_ids = np.asarray(input_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        if len(input_ids) != len(labels):
            return self._failure(started, "input_label_length_mismatch")

        spans = _supervised_spans(labels)
        if len(spans) != 1:
            return self._failure(started, "expected_one_supervised_answer_span")
        answer_start, answer_end = spans[0]
        token_types = self._token_types(input_ids)
        answer_starts = self._find_sid_starts(token_types, labels, answer_start, answer_end, supervised=True)
        if not answer_starts:
            return self._failure(started, "no_complete_answer_sid")
        answer_units = [
            {
                "value": [int(value) for value in input_ids[position : position + 4]],
                "pos_domain": position,
                "pos_a": position + 1,
                "pos_b": position + 2,
                "pos_c": position + 3,
            }
            for position in answer_starts
        ]

        used_positions = {
            int(unit[key])
            for unit in answer_units
            for key in ("pos_domain", "pos_a", "pos_b", "pos_c")
        }
        semantic_answer_positions = {
            position
            for position in range(answer_start, answer_end)
            if token_types[position] != 0
        }
        if semantic_answer_positions != used_positions:
            return self._failure(started, "malformed_answer_sid_structure")

        history_starts = self._find_sid_starts(token_types, labels, 0, answer_start, supervised=False)
        history_sids = [
            [int(value) for value in input_ids[position : position + 4]] for position in history_starts
        ]
        last_sid = answer_units[-1]
        tail_positions = [
            position
            for position in range(int(last_sid["pos_c"]) + 1, answer_end)
            if labels[position] != IGNORE_INDEX
        ]
        continuation_boundaries: list[dict[str, int | None]] = []
        tail_ids = [int(labels[position]) for position in tail_positions]
        for current, following in zip(answer_units[:-1], answer_units[1:]):
            gap_positions = [
                position
                for position in range(int(current["pos_c"]) + 1, int(following["pos_domain"]))
                if labels[position] != IGNORE_INDEX
            ]
            gap_ids = [int(labels[position]) for position in gap_positions]
            common = 0
            while common < min(len(gap_ids), len(tail_ids)) and gap_ids[common] == tail_ids[common]:
                common += 1
            continue_position = gap_positions[common] if common < len(gap_positions) else int(following["pos_domain"])
            stop_token_id = tail_ids[common] if common < len(tail_ids) else None
            if stop_token_id == int(labels[continue_position]):
                stop_token_id = None
            continuation_boundaries.append(
                {
                    "next_domain_pos": int(following["pos_domain"]),
                    "continue_token_pos": int(continue_position),
                    "stop_token_id": None if stop_token_id is None else int(stop_token_id),
                }
            )

        values = [tuple(unit["value"]) for unit in answer_units]
        return {
            "action_select": True,
            "parse_valid": True,
            "parse_error": None,
            "parse_ms": (time.perf_counter() - started) * 1000.0,
            "answer_start": answer_start,
            "answer_end": answer_end,
            "history_sids": history_sids,
            "answer_sid_units": answer_units,
            "continuation_boundaries": continuation_boundaries,
            "final_tail_positions": tail_positions,
            "gold_duplicate": len(values) != len(set(values)),
        }


def offset_action_metadata(metadata: dict[str, Any], segment_start: int) -> dict[str, Any]:
    """Translate local sample positions to packed positions."""
    packed = dict(metadata)
    packed["answer_sid_units"] = [dict(unit) for unit in metadata.get("answer_sid_units", [])]
    packed["continuation_boundaries"] = [
        dict(boundary) for boundary in metadata.get("continuation_boundaries", [])
    ]
    for key in ("answer_start", "answer_end"):
        if key in packed:
            packed[key] = int(packed[key]) + segment_start
    for unit in packed.get("answer_sid_units", []):
        for key in ("pos_domain", "pos_a", "pos_b", "pos_c"):
            unit[key] = int(unit[key]) + segment_start
    for boundary in packed.get("continuation_boundaries", []):
        boundary["next_domain_pos"] = int(boundary["next_domain_pos"]) + segment_start
        if boundary.get("continue_token_pos") is not None:
            boundary["continue_token_pos"] = int(boundary["continue_token_pos"]) + segment_start
    packed["final_tail_positions"] = [
        int(position) + segment_start for position in packed.get("final_tail_positions", [])
    ]
    return packed
