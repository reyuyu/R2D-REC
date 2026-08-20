"""Production policy-batch builder for MC_USER_v1 scored rollouts."""

from __future__ import annotations

import copy
import operator
from typing import Any, Mapping

import torch


class MCBatchError(ValueError):
    pass


def _render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    from run_user_grpo_smoke import render_prompt

    return render_prompt(tokenizer, row)


def _validate_rollout(rollout: Mapping[str, Any]) -> int:
    if rollout.get("K") != 2:
        raise MCBatchError("MC policy batch requires K=2 rollout")
    if rollout.get("token_span_space") != "generated_completion_ids":
        raise MCBatchError("credit units must use generated_completion_ids token space")

    required = (
        "expanded_rows",
        "completion_ids_list",
        "credit_units_per_candidate",
    )
    if any(key not in rollout for key in required):
        raise MCBatchError("rollout is missing required scored-rollout fields")
    candidate_count = len(rollout["expanded_rows"])
    if candidate_count == 0 or candidate_count % 2:
        raise MCBatchError("K=2 rollout must contain a positive even candidate count")
    if len(rollout["completion_ids_list"]) != candidate_count:
        raise MCBatchError("completion candidate count does not match expanded_rows")
    if len(rollout["credit_units_per_candidate"]) != candidate_count:
        raise MCBatchError("credit-unit candidate count does not match expanded_rows")
    if int(rollout.get("candidate_count", candidate_count)) != candidate_count:
        raise MCBatchError("rollout candidate_count is inconsistent")

    for pair_start in range(0, candidate_count, 2):
        first = rollout["expanded_rows"][pair_start]
        second = rollout["expanded_rows"][pair_start + 1]
        if first.get("sample_id") != second.get("sample_id") or first != second:
            raise MCBatchError("expanded_rows do not preserve row0-c0,row0-c1 ordering")
    return candidate_count


def _validate_credit_units(
    completion_ids_list: list[list[int]],
    units_per_candidate: list[list[Mapping[str, Any]]],
) -> None:
    for candidate_index, (completion_ids, units) in enumerate(
        zip(completion_ids_list, units_per_candidate)
    ):
        real_length = len(completion_ids)
        if real_length == 0:
            raise MCBatchError(f"candidate {candidate_index} has empty completion_ids")
        for unit_index, unit in enumerate(units):
            raw_indices = unit.get("generated_token_indices")
            raw_ids = unit.get("generated_token_ids")
            if raw_indices is None or raw_ids is None:
                raise MCBatchError("credit unit lacks generated token indices or IDs")
            try:
                indices = [operator.index(index) for index in raw_indices]
                token_ids = [operator.index(token_id) for token_id in raw_ids]
            except TypeError as exc:
                raise MCBatchError("credit unit token indices and IDs must be integers") from exc
            if not indices or len(indices) != len(token_ids):
                raise MCBatchError("credit unit token indices and IDs must align")
            for offset, (index, expected_id) in enumerate(zip(indices, token_ids)):
                if index < 0 or index >= real_length:
                    raise MCBatchError(
                        f"candidate {candidate_index} unit {unit_index} touches padding"
                    )
                if completion_ids[index] != expected_id:
                    raise MCBatchError(
                        f"candidate {candidate_index} unit {unit_index} token mismatch "
                        f"at offset {offset}"
                    )


def make_mc_policy_batch(
    tokenizer: Any,
    rollout: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Convert a complete K=2 scored rollout into padded policy tensors."""

    candidate_count = _validate_rollout(rollout)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise MCBatchError("tokenizer must define pad_token_id")

    prompt_ids_list = []
    for row in rollout["expanded_rows"]:
        rendered = _render_prompt(tokenizer, row)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not prompt_ids:
            raise MCBatchError("rendered prompt tokenization is empty")
        prompt_ids_list.append([operator.index(token_id) for token_id in prompt_ids])

    completion_ids_list = [
        [operator.index(token_id) for token_id in completion_ids]
        for completion_ids in rollout["completion_ids_list"]
    ]
    units_per_candidate = copy.deepcopy(rollout["credit_units_per_candidate"])
    _validate_credit_units(completion_ids_list, units_per_candidate)

    prompt_width = max(map(len, prompt_ids_list))
    completion_lengths = [len(completion_ids) for completion_ids in completion_ids_list]
    completion_width = max(completion_lengths)
    target_device = torch.device(device)
    prompt_ids = torch.full(
        (candidate_count, prompt_width),
        int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    completion_ids = torch.full(
        (candidate_count, completion_width),
        int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    completion_mask = torch.zeros(
        (candidate_count, completion_width),
        dtype=torch.float32,
        device=target_device,
    )
    for index, ids in enumerate(prompt_ids_list):
        prompt_ids[index, -len(ids) :] = torch.tensor(
            ids, dtype=torch.long, device=target_device
        )
        prompt_mask[index, -len(ids) :] = 1
    for index, ids in enumerate(completion_ids_list):
        completion_ids[index, : len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=target_device
        )
        completion_mask[index, : len(ids)] = 1

    for index, real_length in enumerate(completion_lengths):
        if not bool(torch.all(completion_mask[index, :real_length] == 1)):
            raise MCBatchError("completion real-token mask construction failed")
        if not bool(torch.all(completion_mask[index, real_length:] == 0)):
            raise MCBatchError("completion padding mask construction failed")

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "credit_units_per_candidate": units_per_candidate,
        "candidate_count": candidate_count,
        "completion_lengths": completion_lengths,
        "sample_ids": [row["sample_id"] for row in rollout["expanded_rows"]],
        "candidate_indices": [index % 2 for index in range(candidate_count)],
    }


__all__ = ["MCBatchError", "make_mc_policy_batch"]
