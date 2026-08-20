"""Fail-closed projection of MC credit units onto generated token IDs."""

from __future__ import annotations

import copy
import difflib
from typing import Any, Mapping, Sequence


class MCCreditProjectionError(ValueError):
    pass


def _id_list(values: Any) -> list[int]:
    items = values.tolist() if hasattr(values, "tolist") else list(values)
    if items and isinstance(items[0], list):
        raise MCCreditProjectionError("token IDs must be one-dimensional")
    return [int(item) for item in items]


def _validate_canonical_unit(
    unit: Mapping[str, Any],
    canonical_ids: Sequence[int],
) -> tuple[int, int, list[int]]:
    start = int(unit["token_start"])
    end = int(unit["token_end"])
    if start < 0 or end > len(canonical_ids) or start >= end:
        raise MCCreditProjectionError("credit unit has an empty or invalid canonical token span")
    ids = _id_list(unit["token_ids"])
    if ids != list(canonical_ids[start:end]):
        raise MCCreditProjectionError("credit unit canonical token IDs do not match its span")
    return start, end, ids


def _is_contiguous(indices: Sequence[int]) -> bool:
    return bool(indices) and list(indices) == list(range(indices[0], indices[-1] + 1))


def project_mc_credit_units_to_generated(
    completion: str,
    canonical_units: Sequence[Mapping[str, Any]],
    generated_ids: Sequence[int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Project canonical MC credit spans to the exact generated ID sequence."""

    encoded = tokenizer(completion, add_special_tokens=False)
    canonical_ids = _id_list(encoded["input_ids"])
    generated_ids = _id_list(generated_ids)
    canonical_text = tokenizer.decode(canonical_ids, skip_special_tokens=False)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    if canonical_text != completion:
        raise MCCreditProjectionError("canonical token IDs do not decode to the scored completion")
    if generated_text != completion:
        raise MCCreditProjectionError("generated token IDs do not decode to the scored completion")

    validated_units = [
        (*_validate_canonical_unit(unit, canonical_ids), unit)
        for unit in canonical_units
    ]
    projection_required = canonical_ids != generated_ids
    if projection_required:
        matcher = difflib.SequenceMatcher(a=canonical_ids, b=generated_ids, autojunk=False)
        opcodes = list(matcher.get_opcodes())
    else:
        opcodes = [("equal", 0, len(canonical_ids), 0, len(generated_ids))]

    for tag, old_start, old_end, _new_start, _new_end in opcodes:
        if tag in {"insert", "delete"}:
            raise MCCreditProjectionError(f"unsafe MC token projection opcode: {tag}")
        if tag != "replace":
            continue
        replacement_block = set(range(old_start, old_end))
        for unit_start, unit_end, _unit_ids, _unit in validated_units:
            covered = replacement_block.intersection(range(unit_start, unit_end))
            if covered and covered != replacement_block:
                raise MCCreditProjectionError(
                    "credit unit partially intersects a non-canonical replacement block"
                )

    canonical_to_generated = {index: set() for index in range(len(canonical_ids))}
    for tag, old_start, old_end, new_start, new_end in opcodes:
        if tag == "equal":
            for offset in range(old_end - old_start):
                canonical_to_generated[old_start + offset].add(new_start + offset)
        elif tag == "replace":
            replacement = set(range(new_start, new_end))
            for old_index in range(old_start, old_end):
                canonical_to_generated[old_index].update(replacement)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    projected_units = []
    for unit_start, unit_end, unit_ids, unit in validated_units:
        generated_indices = sorted(
            {
                generated_index
                for canonical_index in range(unit_start, unit_end)
                for generated_index in canonical_to_generated[canonical_index]
            }
        )
        if not generated_indices:
            raise MCCreditProjectionError("credit unit disappeared during token projection")
        if generated_indices[-1] >= len(generated_ids):
            raise MCCreditProjectionError("projected credit unit exceeds generated token IDs")
        generated_unit_ids = [generated_ids[index] for index in generated_indices]
        if pad_token_id is not None and any(token_id == pad_token_id for token_id in generated_unit_ids):
            raise MCCreditProjectionError("projected credit unit maps to padding")
        if not _is_contiguous(generated_indices):
            raise MCCreditProjectionError("projected credit unit is unexpectedly non-contiguous")

        projected = copy.deepcopy(dict(unit))
        projected.update(
            {
                "canonical_token_start": unit_start,
                "canonical_token_end": unit_end,
                "canonical_token_ids": unit_ids,
                "generated_token_indices": generated_indices,
                "generated_token_ids": generated_unit_ids,
                "generated_token_start": generated_indices[0],
                "generated_token_end": generated_indices[-1] + 1,
                "token_start": generated_indices[0],
                "token_end": generated_indices[-1] + 1,
                "token_ids": generated_unit_ids,
                "projection_required": projection_required,
            }
        )
        if projected["generated_token_ids"] != [
            generated_ids[index] for index in projected["generated_token_indices"]
        ]:
            raise MCCreditProjectionError("projected generated token IDs failed exact validation")
        projected_units.append(projected)

    if len(projected_units) != len(canonical_units):
        raise MCCreditProjectionError("a credit unit disappeared during projection")
    return {
        "projection_required": projection_required,
        "opcodes": [list(opcode) for opcode in opcodes] if projection_required else [],
        "canonical_ids": canonical_ids,
        "generated_ids": generated_ids,
        "projected_units": projected_units,
    }


__all__ = ["MCCreditProjectionError", "project_mc_credit_units_to_generated"]
