"""Deterministic full-sequence policy scoring shared by rollout runtime and audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


POLICY_SCORING_MODE = "eval"
SCORING_MICROBATCH_SIZE = 2


@dataclass(frozen=True)
class FullSequenceScore:
    logps: tuple[tuple[float, ...], ...]
    requires_grad_by_microbatch: tuple[bool, ...]
    graph_connected_by_microbatch: tuple[bool, ...]


def graph_connected_to_parameters(tensor, parameters: Iterable) -> bool:
    """Inspect autograd leaves without executing backward or computing gradients."""
    targets = {id(parameter) for parameter in parameters if parameter.requires_grad}
    pending = [tensor.grad_fn]
    visited: set[Any] = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        # Keep wrappers alive: CPython may reuse ids during next_functions traversal.
        visited.add(node)
        variable = getattr(node, "variable", None)
        if variable is not None and id(variable) in targets:
            return True
        pending.extend(child for child, _ in getattr(node, "next_functions", ()) if child is not None)
    return False


def parameter_versions(model) -> tuple[int, ...]:
    return tuple(int(parameter._version) for parameter in model.parameters())


def score_full_sequences(
    model,
    context_ids: Sequence[int],
    completion_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device | str | None = None,
    *,
    grad_enabled: bool,
    trainable_parameters: Iterable = (),
    scoring_microbatch_size: int | None = None,
) -> FullSequenceScore:
    """Score sampled tokens from complete context+completion rows in fixed chunks."""
    if model.training:
        raise ValueError("policy scoring requires model.eval()")
    if len(completion_ids) != 8:
        raise ValueError("policy scoring requires one ordered G8 group")
    normalized = tuple(tuple(int(token) for token in row) for row in completion_ids)
    if any(not 0 < len(row) <= 3 for row in normalized):
        raise ValueError("each sampled completion must contain 1..3 tokens")
    context = tuple(int(token) for token in context_ids)
    if not context:
        raise ValueError("empty rollout context")
    if device is None:
        device = next(model.parameters()).device
    microbatch_size = SCORING_MICROBATCH_SIZE if scoring_microbatch_size is None else int(scoring_microbatch_size)
    if microbatch_size not in (1, 2):
        raise ValueError("scoring_microbatch_size must be 1 or 2")

    values: list[tuple[float, ...]] = []
    requires_grad: list[bool] = []
    graph_connected: list[bool] = []
    parameters = tuple(trainable_parameters)
    for start in range(0, 8, microbatch_size):
        rows = normalized[start:start + microbatch_size]
        lengths = [len(row) for row in rows]
        completion_width = max(lengths)
        sequence_length = len(context) + completion_width
        input_ids = torch.full((len(rows), sequence_length), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        sampled = torch.full((len(rows), completion_width), int(pad_token_id), dtype=torch.long, device=device)
        for row_index, row in enumerate(rows):
            sequence = context + row
            input_ids[row_index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[row_index, :len(sequence)] = True
            sampled[row_index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with grad_context:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, "logits") else output
            row_indices = torch.arange(len(rows), device=device).unsqueeze(1).expand(len(rows), completion_width)
            causal_indices = torch.arange(len(context) - 1, len(context) - 1 + completion_width, device=device).unsqueeze(0).expand(len(rows), completion_width)
            selected = logits[row_indices, causal_indices]
            log_probs = torch.log_softmax(selected, dim=-1)
            chunk = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        requires_grad.append(bool(chunk.requires_grad))
        graph_connected.append(graph_connected_to_parameters(chunk, parameters) if grad_enabled else False)
        chunk_cpu = chunk.detach().float().cpu()
        values.extend(tuple(float(value) for value in chunk_cpu[index, :length]) for index, length in enumerate(lengths))
        del chunk_cpu, chunk, log_probs, selected, logits, output, sampled, input_ids, attention_mask
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return FullSequenceScore(tuple(values), tuple(requires_grad), tuple(graph_connected))
