#!/usr/bin/env python3
"""Real 8B+BATA DSR-Simple gradient audit with zero optimizer steps."""
from __future__ import annotations

import statistics

import torch

import gr_rec_dsr_v1.run_gradient_budget_audit as base
from gr_rec_dsr_v1.dsr_parser import parse_interest_section

from .simple_objectives import (
    beam_a_diversity,
    choose_simple_think_aux_scores,
    interest_count_score,
    simple_group_advantages,
)


base.OUTPUT = base.Path(
    "/data/GRPO/outputs/GR-REC-DSR-SIMPLE-V1-GRADIENT-AUDIT.json"
)


def _simple_think_gradient(model, tokenizer, parameters, record):
    candidates = []
    for item in record["candidates"]:
        parsed = parse_interest_section(item["completion"], record["prompt"])
        diversity = beam_a_diversity(item["beam_sids"], record["target_domain"])
        candidates.append({
            "primary_reward": item["reward"],
            "s_n": interest_count_score(parsed.bullet_count),
            **diversity,
        })
    scores, branch = choose_simple_think_aux_scores(candidates)
    aux_advantages = simple_group_advantages(scores).to(model.device)
    rewards = torch.tensor(
        [item["reward"] for item in record["candidates"]], device=model.device
    )
    primary_advantages = base.group_advantages_population(rewards, 4).to(model.device)
    logps, mask = base._per_token_logps(
        model,
        tokenizer,
        record["prompt"],
        [item["completion_ids"] for item in record["candidates"]],
    )
    ratio = torch.exp(logps - logps.detach())
    primary_per_sample = (
        (-(ratio * primary_advantages[:, None]) * mask).sum(-1) / mask.sum(-1)
    )
    aux_per_sample = (
        (-(ratio * aux_advantages[:, None]) * mask).sum(-1) / mask.sum(-1)
    )
    primary_loss = base.ROUTE_LOSS_W["think"] * primary_per_sample.mean()
    weighted_auxiliary_loss = 0.10 * aux_per_sample.mean()
    metrics = base._three_backwards(
        model, parameters, primary_loss, weighted_auxiliary_loss
    )
    primary_std = float(rewards.std(correction=0))
    classification = "NORMAL-SIGNAL" if primary_std > 0 else (
        "DEAD-ZERO" if all(float(value) == 0.0 for value in rewards) else "ZERO-STD-OTHER"
    )
    metrics.update({
        "group_id": record["group_id"],
        "source": record["source"],
        "domain": record["target_domain"],
        "gold_a_count": base._gold_a_count(record["gold_sids"]),
        "gold_a_bucket": base._bucket(base._gold_a_count(record["gold_sids"])),
        "classification": classification,
        "primary_rewards": [float(value) for value in rewards.cpu()],
        "primary_std": primary_std,
        "branch": branch,
        "raw_interest_n": [
            parse_interest_section(item["completion"], record["prompt"]).bullet_count
            for item in record["candidates"]
        ],
        "unique_valid_target_a": [item["unique_valid_target_a"] for item in candidates],
        "d_a": [item["d_a"] for item in candidates],
        "s_aux": scores,
        "a_aux": aux_advantages.detach().cpu().tolist(),
        "s_aux_std": float(torch.tensor(scores).std(correction=0)),
        "primary_loss": float(primary_loss.detach()),
        "weighted_auxiliary_loss": float(weighted_auxiliary_loss.detach()),
        "primary_signal_aux_gradient_exact_zero": (
            classification != "NORMAL-SIGNAL" or metrics["auxiliary_norm"] == 0.0
        ),
        "completion_sha256": [
            base._completion_sha256(item["completion_ids"])
            for item in record["candidates"]
        ],
    })
    return metrics


def _simple_summary(groups):
    normal = [item for item in groups if item["classification"] == "NORMAL-SIGNAL"]
    rescue = [
        item for item in groups
        if item["classification"] != "NORMAL-SIGNAL" and item["s_aux_std"] > 0.0
    ]
    if not normal:
        raise RuntimeError("gradient audit found no primary-signal group")
    if not rescue:
        raise RuntimeError("gradient audit found no zero-std group with active rescue")
    if any(item["auxiliary_norm"] != 0.0 for item in normal):
        raise RuntimeError("Simple auxiliary gradient was nonzero on primary-signal group")
    weighted_rescue_norms = [item["auxiliary_norm"] for item in rescue]
    if not any(value > 0.0 for value in weighted_rescue_norms):
        raise RuntimeError("Simple rescue gradient was zero on every eligible group")
    return {
        "primary_signal_group_count": len(normal),
        "primary_signal_aux_gradient_exact_zero": True,
        "rescue_group_count": len(rescue),
        "weighted_rescue_gradient_norm": {
            "mean": statistics.fmean(weighted_rescue_norms),
            "max": max(weighted_rescue_norms),
        },
        "primary_signal_group_ids": [item["group_id"] for item in normal],
        "rescue_group_ids": [item["group_id"] for item in rescue],
    }


def main():
    base._think_gradient = _simple_think_gradient
    base._summary = _simple_summary
    base.main()


if __name__ == "__main__":
    main()
