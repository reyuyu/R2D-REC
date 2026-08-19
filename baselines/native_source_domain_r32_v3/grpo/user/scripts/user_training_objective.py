"""Pure User GRPO group normalization, token advantages, and clipped loss."""

from __future__ import annotations

import collections
import math

import torch


G = 4
ADVANTAGE_EPSILON = 1e-4
BASE_LAMBDA = 0.50
EPSILON = 0.20
WHITELISTS = {
    "action": frozenset({"hallucinated_sid", "duplicate_sid"}),
    "chain": frozenset(
        {
            "hallucinated_sid",
            "date_mismatch",
            "action_mismatch",
            "duplicate_event",
            "chronology_violation",
            "excess_event",
        }
    ),
}


def group_population_advantages(rewards, sample_ids, *, group_size=G, epsilon=ADVANTAGE_EPSILON):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group_size:
        raise ValueError("rewards must be a non-empty flat tensor divisible by G")
    if len(sample_ids) != rewards.numel():
        raise ValueError("sample_ids and rewards differ in length")
    for start in range(0, len(sample_ids), group_size):
        group_ids = sample_ids[start : start + group_size]
        if len(set(group_ids)) != 1:
            raise ValueError(f"G={group_size} group mixes sample IDs: {group_ids}")
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, correction=0, keepdim=True)
    centered = grouped - means
    advantages = torch.where(
        stds == 0,
        torch.zeros_like(centered),
        centered / (stds + epsilon),
    )
    return advantages.reshape(-1), means.squeeze(1), stds.squeeze(1)


def _included_records(compiled, route):
    if route not in WHITELISTS:
        raise ValueError(f"unknown route: {route}")
    for record in compiled["records"]:
        if record["included"] and record["kind"] in WHITELISTS[route]:
            yield record


def build_token_advantages(
    sequence_advantages,
    compiled_penalties,
    routes,
    completion_lengths,
    *,
    max_length=None,
    base_lambda=BASE_LAMBDA,
):
    """Build padded sqrt-normalized token advantages from projected records."""
    sequence_advantages = torch.as_tensor(sequence_advantages, dtype=torch.float32)
    batch_size = sequence_advantages.numel()
    if not (len(compiled_penalties) == len(routes) == len(completion_lengths) == batch_size):
        raise ValueError("token-advantage batch fields differ in length")
    max_length = max_length if max_length is not None else max(completion_lengths, default=0)
    token_advantages = sequence_advantages[:, None].expand(batch_size, max_length).clone()
    effective_penalties = torch.zeros_like(token_advantages)
    local_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    winning_kinds = [[set() for _ in range(max_length)] for _ in range(batch_size)]
    per_kind_masked_tokens = collections.Counter()

    for row, (compiled, route, completion_length) in enumerate(
        zip(compiled_penalties, routes, completion_lengths)
    ):
        if compiled["token_count"] != completion_length:
            raise ValueError("projected penalty token count differs from generated completion")
        if completion_length > max_length:
            raise ValueError("completion length exceeds padded width")
        kind_indices = collections.defaultdict(set)
        for record in _included_records(compiled, route):
            indices = sorted(set(record["masked_token_indices"]))
            if not indices or indices[-1] >= completion_length:
                raise ValueError("included violation has an invalid generated-token span")
            lambda_eff = base_lambda / math.sqrt(len(indices))
            for token_index in indices:
                current = float(effective_penalties[row, token_index])
                if lambda_eff > current + 1e-12:
                    effective_penalties[row, token_index] = lambda_eff
                    winning_kinds[row][token_index] = {record["kind"]}
                elif math.isclose(lambda_eff, current, rel_tol=0.0, abs_tol=1e-12):
                    winning_kinds[row][token_index].add(record["kind"])
                kind_indices[record["kind"]].add(token_index)
        for kind, indices in kind_indices.items():
            per_kind_masked_tokens[kind] += len(indices)

    local_mask = effective_penalties > 0
    token_advantages = torch.where(
        local_mask,
        torch.minimum(token_advantages, -effective_penalties),
        token_advantages,
    )
    valid_mask = torch.arange(max_length, device=token_advantages.device)[None, :] < torch.tensor(
        completion_lengths, device=token_advantages.device
    )[:, None]
    if bool((local_mask & ~valid_mask).any()):
        raise AssertionError("local penalty touches right padding")
    incremental_by_kind = collections.Counter()
    positive_flip_count = 0
    for row in range(batch_size):
        baseline_negative = max(-float(sequence_advantages[row]), 0.0)
        for token_index in torch.where(local_mask[row])[0].tolist():
            final = float(token_advantages[row, token_index])
            incremental = max(-final, 0.0) - baseline_negative
            if float(sequence_advantages[row]) > 0 and final < 0:
                positive_flip_count += 1
            winners = winning_kinds[row][token_index]
            if winners:
                for kind in winners:
                    incremental_by_kind[kind] += max(incremental, 0.0) / len(winners)

    metadata = {
        "local_mask": local_mask,
        "effective_penalties": effective_penalties,
        "per_kind_masked_token_count": dict(per_kind_masked_tokens),
        "per_kind_incremental_negative_mass": dict(incremental_by_kind),
        "positive_sequence_masked_token_flip_count": positive_flip_count,
    }
    return token_advantages, metadata


def clipped_grpo_loss_from_logps(
    per_token_logps,
    old_per_token_logps,
    token_advantages,
    completion_mask,
    *,
    epsilon=EPSILON,
    delta=None,
):
    if not (
        per_token_logps.shape
        == old_per_token_logps.shape
        == token_advantages.shape
        == completion_mask.shape
    ):
        raise ValueError("GRPO loss tensors must have identical [B,T] shape")
    log_ratio = per_token_logps - old_per_token_logps
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
    if delta is not None:
        coef_1 = torch.clamp(coef_1, max=delta)
    per_token_loss1 = coef_1 * token_advantages
    per_token_loss2 = coef_2 * token_advantages
    per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)
    mask = completion_mask.to(per_token_loss.dtype)
    per_sample_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    return per_sample_loss.mean(), per_token_loss, log_ratio


def standard_sequence_grpo_loss_from_logps(
    per_token_logps,
    old_per_token_logps,
    sequence_advantages,
    completion_mask,
    *,
    epsilon=EPSILON,
    delta=None,
):
    token_advantages = torch.as_tensor(
        sequence_advantages,
        dtype=per_token_logps.dtype,
        device=per_token_logps.device,
    )[:, None].expand_as(per_token_logps)
    return clipped_grpo_loss_from_logps(
        per_token_logps,
        old_per_token_logps,
        token_advantages,
        completion_mask,
        epsilon=epsilon,
        delta=delta,
    )
