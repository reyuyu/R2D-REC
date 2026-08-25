"""Padding-aware collation for one preserved TrueRec G8 business group."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from action_alignment import align_action
from rollout_runtime_v1 import BusinessGroupRollout, G


@dataclass(frozen=True)
class PaddedBusinessGroup:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    action_indices: torch.Tensor
    causal_logit_indices: torch.Tensor
    context_lengths: torch.Tensor
    padding_side: str
    pad_token_id: int


def collate_business_group(group: BusinessGroupRollout, pad_token_id: int, padding_side: str = "right") -> PaddedBusinessGroup:
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be left or right")
    sequences = [list(group.context_ids) + list(candidate.completion_ids) for candidate in group.candidates]
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((G, max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((G, max_length), dtype=torch.bool)
    completion_ids = torch.full((G, 3), int(pad_token_id), dtype=torch.long)
    completion_mask = torch.zeros((G, 3), dtype=torch.bool)
    old_logps = torch.zeros((G, 3), dtype=torch.float32)
    action_indices = torch.zeros((G, 3), dtype=torch.long)
    causal_indices = torch.zeros((G, 3), dtype=torch.long)
    context_lengths = torch.full((G,), len(group.context_ids), dtype=torch.long)
    for row, (sequence, candidate) in enumerate(zip(sequences, group.candidates)):
        offset = 0 if padding_side == "right" else max_length - len(sequence)
        input_ids[row, offset: offset + len(sequence)] = torch.tensor(sequence)
        attention_mask[row, offset: offset + len(sequence)] = True
        length = len(candidate.completion_ids)
        completion_ids[row, :length] = torch.tensor(candidate.completion_ids)
        completion_mask[row, :length] = True
        old_logps[row, :length] = torch.tensor(candidate.old_logps)
        logical = align_action(len(group.context_ids), [0, 0, 0])
        action_indices[row] = torch.tensor(logical.action_indices) + offset
        causal_indices[row] = torch.tensor(logical.logit_indices) + offset
        actual_action_start = offset + len(group.context_ids)
        if not torch.equal(action_indices[row], torch.arange(actual_action_start, actual_action_start + 3)):
            raise ValueError("padding action alignment failed")
        if not bool(attention_mask[row, action_indices[row, :length]].all()):
            raise ValueError("actual completion token aligned to padding")
    return PaddedBusinessGroup(input_ids, attention_mask, completion_ids, completion_mask, old_logps, action_indices, causal_indices, context_lengths, padding_side, int(pad_token_id))
