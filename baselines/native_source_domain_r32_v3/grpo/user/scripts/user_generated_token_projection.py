"""Fail-closed projection from canonical text tokens to generated token IDs."""

from __future__ import annotations

import copy
import difflib


class GeneratedTokenProjectionError(ValueError):
    pass


def project_compiled_mask_to_generated(
    compiled,
    generated_ids,
    *,
    tokenizer=None,
    completion=None,
):
    """Project compiler masks onto the model's original tokenization.

    A non-canonical BPE sequence may decode to the same text as the compiler's
    canonical tokenization. Replacement blocks are safe only when each included
    record covers all or none of the canonical block. Partial coverage would
    broaden locality, so it fails closed.
    """
    canonical_ids = list(compiled["input_ids"])
    generated_ids = list(generated_ids)
    if tokenizer is not None:
        if completion is None:
            raise GeneratedTokenProjectionError("completion is required with tokenizer validation")
        canonical_text = tokenizer.decode(canonical_ids, skip_special_tokens=False)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if canonical_text != completion or generated_text != completion:
            raise GeneratedTokenProjectionError("token sequences do not decode to the scored completion")
    if canonical_ids == generated_ids:
        output = copy.deepcopy(compiled)
        output["tokenization_projection"] = {"required": False, "opcodes": []}
        return output

    matcher = difflib.SequenceMatcher(a=canonical_ids, b=generated_ids, autojunk=False)
    opcodes = list(matcher.get_opcodes())
    for tag, old_start, old_end, _new_start, _new_end in opcodes:
        if tag in {"insert", "delete"}:
            raise GeneratedTokenProjectionError(f"unsafe token projection opcode: {tag}")
        if tag != "replace":
            continue
        block = set(range(old_start, old_end))
        for record in compiled["records"]:
            if not record["included"]:
                continue
            covered = block.intersection(record["masked_token_indices"])
            if covered and covered != block:
                raise GeneratedTokenProjectionError(
                    "violation partially intersects a non-canonical token replacement block"
                )

    canonical_to_generated = {index: set() for index in range(len(canonical_ids))}
    serialized_opcodes = []
    for tag, old_start, old_end, new_start, new_end in opcodes:
        serialized_opcodes.append([tag, old_start, old_end, new_start, new_end])
        if tag == "equal":
            for delta in range(old_end - old_start):
                canonical_to_generated[old_start + delta].add(new_start + delta)
        else:
            replacement = set(range(new_start, new_end))
            for old_index in range(old_start, old_end):
                canonical_to_generated[old_index].update(replacement)

    output = copy.deepcopy(compiled)
    projected_per_kind = {}
    for kind, canonical_mask in compiled["per_kind_masks"].items():
        projected = [False] * len(generated_ids)
        for old_index, masked in enumerate(canonical_mask):
            if masked:
                for new_index in canonical_to_generated[old_index]:
                    projected[new_index] = True
        projected_per_kind[kind] = projected

    for record in output["records"]:
        old_indices = list(record["masked_token_indices"])
        new_indices = sorted({new for old in old_indices for new in canonical_to_generated[old]})
        if record["included"] and old_indices and not new_indices:
            raise GeneratedTokenProjectionError("included violation became empty during projection")
        record["canonical_masked_token_indices"] = old_indices
        record["masked_token_indices"] = new_indices
        for span in record["token_spans"]:
            old_span = range(span["token_start"], span["token_end"])
            mapped = sorted({new for old in old_span for new in canonical_to_generated[old]})
            if not mapped:
                raise GeneratedTokenProjectionError("violation token span became empty during projection")
            span["canonical_token_start"] = span["token_start"]
            span["canonical_token_end"] = span["token_end"]
            span["token_start"] = mapped[0]
            span["token_end"] = mapped[-1] + 1
            span["token_ids"] = generated_ids[span["token_start"] : span["token_end"]]

    penalty_mask = [
        any(mask[index] for mask in projected_per_kind.values()) for index in range(len(generated_ids))
    ]
    output.update(
        {
            "token_count": len(generated_ids),
            "input_ids": generated_ids,
            "penalty_mask": penalty_mask,
            "per_kind_masks": projected_per_kind,
            "masked_token_count": sum(penalty_mask),
            "masked_token_fraction": sum(penalty_mask) / len(generated_ids) if generated_ids else 0.0,
            "tokenization_projection": {"required": True, "opcodes": serialized_opcodes},
        }
    )
    return output
