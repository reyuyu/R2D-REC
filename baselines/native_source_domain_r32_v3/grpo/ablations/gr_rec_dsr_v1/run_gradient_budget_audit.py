#!/usr/bin/env python3
"""Real 8B+BATA DSR gradient-budget audit with zero optimizer steps."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

from grpo_model import ADAPTER, BASE, encode_prompt, generate_batch, load_model
from grpo_sid import parse_sid, q_reward
from grpo_trl_trainer import (
    ROUTE_G,
    ROUTE_LOSS_W,
    RouteAwareRepeatSampler,
    build_route_dataset,
    group_advantages_population,
)
from run_grpo_trl_smoke import DATA, make_beam32_fn
from trl.trainer.utils import selective_log_softmax

from gr_rec_dsr_v1.dsr_contract import PROBE_GROUP_IDS, PROBE_SEED, TRAIN_SEED
from gr_rec_dsr_v1.dsr_objectives import (
    build_nothink_rescue_plan,
    choose_think_aux_scores,
    cot_score,
    exploration_score,
    group_aux_advantages,
    nothink_unlikelihood_loss,
    prefix_support,
)
from gr_rec_dsr_v1.dsr_parser import parse_interest_section
from gr_rec_dsr_v1.dsr_runtime import locate_final_sid_a_position


OUTPUT = Path("/data/GRPO/outputs/GR-REC-DSR-V1-GRADIENT-BUDGET-AUDIT.json")
NOTHINK_TRACE = Path("/data/GRPO/outputs/GR-REC-DSR-V1-NOTHINK-PROBE.json")
DOMAINS = ("video", "living", "prod", "ad")


def _gold_set(values):
    return {sid for value in values if (sid := parse_sid(value)) is not None}


def _gold_a_count(values):
    return len({sid[:2] for value in values if (sid := parse_sid(value)) is not None})


def _bucket(count):
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def _completion_sha256(ids):
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _trim_think(ids, close_id):
    if close_id in ids:
        return ids[: ids.index(close_id) + 1]
    return ids


def _read_rows():
    by_group = defaultdict(dict)
    with open(DATA, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_group[row["recommendation_group_id"]][row["route"]] = row
    return by_group


def _select_generated_rows(by_group):
    dataset = build_route_dataset(
        DATA, n_groups=len(by_group), seed=int(TRAIN_SEED), chunk=8,
        exclude_group_ids=PROBE_GROUP_IDS,
    )
    sampler = RouteAwareRepeatSampler(dataset, generation_batch_size=16, repeat_count=2, shuffle=False)
    rows = list(dataset)
    trained = {
        rows[index]["recommendation_group_id"]
        for _route, indices in sampler._chunks for index in indices
    }
    order = []
    seen = set()
    for item in rows:
        group_id = item["recommendation_group_id"]
        if group_id in trained and group_id not in seen:
            seen.add(group_id)
            order.append(group_id)
    selected = {}
    for domain in DOMAINS:
        candidates = [
            by_group[group_id]["think"] for group_id in order
            if by_group[group_id]["think"]["target_domain"] == domain
        ]
        buckets = defaultdict(list)
        for row in candidates:
            buckets[_bucket(_gold_a_count(row["all_gold_sids"]))].append(row)
        domain_rows = []
        for label in ("1", "3-4", "5+"):
            if buckets[label]:
                domain_rows.append(buckets[label][0])
        for row in candidates:
            if len(domain_rows) == 3:
                break
            if row not in domain_rows:
                domain_rows.append(row)
        if len(domain_rows) != 3:
            raise RuntimeError(f"not enough stratified rows for {domain}")
        selected[domain] = domain_rows
    return selected


def _generated_record(model, tokenizer, beam32_fn, row, seed, source):
    prompt = row["prompt"]
    prompt_ids = encode_prompt(tokenizer, prompt)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    texts, completion_ids = generate_batch(
        model, tokenizer, [prompt_ids] * 4,
        max_new_tokens=2048, do_sample=True, temperature=0.9, top_p=0.95,
        return_ids=True,
    )
    close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise RuntimeError(f"expected one </think> token, got {close}")
    completion_ids = [_trim_think(ids, close[0]) for ids in completion_ids]
    texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in completion_ids]
    gold = _gold_set(row["all_gold_sids"])
    rewards = beam32_fn([prompt] * 4, texts, completion_ids, [gold] * 4)
    results = model._beam_stats["last_call"]["local_results"]
    candidates = []
    for text, ids, reward, result in zip(texts, completion_ids, rewards, results):
        candidates.append({
            "completion": text,
            "completion_ids": ids,
            "completion_sha256": _completion_sha256(ids),
            "reward": float(reward),
            "exact": int(result["exact"]),
            "beam_sids": [tuple(value) if value is not None else None for value in result["beam_sids"]],
        })
    return {
        "source": source,
        "group_id": row["recommendation_group_id"],
        "prompt": prompt,
        "target_domain": row["target_domain"],
        "gold_sids": row["all_gold_sids"],
        "candidates": candidates,
    }


def _per_token_logps(model, tokenizer, prompt, completion_ids):
    prompt_ids = encode_prompt(tokenizer, prompt)
    max_completion = max(len(ids) for ids in completion_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    completion = torch.full(
        (len(completion_ids), max_completion), pad_id, dtype=torch.long, device=model.device
    )
    completion_mask = torch.zeros_like(completion, dtype=torch.float32)
    for index, ids in enumerate(completion_ids):
        completion[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        completion_mask[index, :len(ids)] = 1.0
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).repeat(len(completion_ids), 1)
    input_ids = torch.cat([prompt_tensor, completion], dim=1)
    attention_mask = torch.cat([torch.ones_like(prompt_tensor), completion_mask.long()], dim=1)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=max_completion + 1,
        use_cache=False,
    ).logits[:, :-1, :]
    if logits.size(1) != max_completion:
        logits = logits[:, -max_completion:, :]
    return selective_log_softmax(logits, completion), completion_mask


def _lora_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if "lora" in name.lower()]


def _release_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _snapshot(parameters):
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in parameters if parameter.grad is not None
    }


def _dot(left, right, names):
    value = torch.zeros((), dtype=torch.float32, device=next(iter(left.values())).device if left else next(iter(right.values())).device)
    for name in names:
        a = left.get(name)
        b = right.get(name)
        if a is not None and b is not None:
            value = value + (a.float() * b.float()).sum()
    return float(value)


def _norm(values, names):
    return math.sqrt(max(_dot(values, values, names), 0.0))


def _gradient_metrics(primary, auxiliary, total):
    names = sorted(set(primary) | set(auxiliary) | set(total))
    p_norm = _norm(primary, names)
    a_norm = _norm(auxiliary, names)
    dot = _dot(primary, auxiliary, names)
    cosine = dot / (p_norm * a_norm) if p_norm > 0 and a_norm > 0 else None
    difference = {}
    for name in names:
        reference = total.get(name)
        if reference is None:
            reference = torch.zeros_like(primary.get(name, auxiliary[name]))
        difference[name] = reference - primary.get(name, torch.zeros_like(reference)) - auxiliary.get(name, torch.zeros_like(reference))
    total_norm = _norm(total, names)
    diff_norm = _norm(difference, names)
    result = {
        "primary_norm": p_norm,
        "auxiliary_norm": a_norm,
        "ratio": a_norm / (p_norm + 1e-12),
        "cosine": cosine,
        "total_norm": total_norm,
        "total_additivity_diff_norm": diff_norm,
        "total_additivity_relative_error": diff_norm / (total_norm + 1e-12),
    }
    for label, marker in (("lora_A", "lora_a"), ("lora_B", "lora_b")):
        layer_names = [name for name in names if marker in name.lower()]
        lp = _norm(primary, layer_names)
        la = _norm(auxiliary, layer_names)
        ld = _dot(primary, auxiliary, layer_names)
        result[label] = {
            "primary_norm": lp,
            "auxiliary_norm": la,
            "ratio": la / (lp + 1e-12),
            "cosine": ld / (lp * la) if lp > 0 and la > 0 else None,
        }
    return result


def _three_backwards(model, parameters, primary_loss, auxiliary_loss):
    model.zero_grad(set_to_none=True)
    primary_loss.backward(retain_graph=True)
    primary = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    auxiliary_loss.backward(retain_graph=True)
    auxiliary = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    (primary_loss + auxiliary_loss).backward()
    total = _snapshot(parameters)
    model.zero_grad(set_to_none=True)
    return _gradient_metrics(primary, auxiliary, total)


def _think_gradient(model, tokenizer, parameters, record):
    candidates = []
    gold = _gold_set(record["gold_sids"])
    for item in record["candidates"]:
        parsed = parse_interest_section(item["completion"], record["prompt"])
        cot = cot_score(parsed)
        prefix = prefix_support(item["beam_sids"], gold)
        explore = exploration_score(item["beam_sids"], record["target_domain"])
        candidates.append({
            "primary_reward": item["reward"],
            "has_exact": item["exact"] > 0,
            **cot,
            **prefix,
            **explore,
            "s_dead": float(cot["s_cot"]) * float(explore["s_explore"]),
        })
    scores, branch = choose_think_aux_scores(candidates)
    aux_advantages = group_aux_advantages(scores, 4).to(model.device)
    rewards = torch.tensor([item["reward"] for item in record["candidates"]], device=model.device)
    primary_advantages = group_advantages_population(rewards, 4).to(model.device)
    logps, mask = _per_token_logps(
        model, tokenizer, record["prompt"],
        [item["completion_ids"] for item in record["candidates"]],
    )
    ratio = torch.exp(logps - logps.detach())
    primary_per_sample = (-(ratio * primary_advantages[:, None]) * mask).sum(-1) / mask.sum(-1)
    aux_per_sample = (-(ratio * aux_advantages[:, None]) * mask).sum(-1) / mask.sum(-1)
    primary_loss = ROUTE_LOSS_W["think"] * primary_per_sample.mean()
    auxiliary_loss = 0.10 * aux_per_sample.mean()
    metrics = _three_backwards(model, parameters, primary_loss, auxiliary_loss)
    primary_std = float(rewards.std(correction=0))
    metrics.update({
        "group_id": record["group_id"],
        "source": record["source"],
        "domain": record["target_domain"],
        "gold_a_count": _gold_a_count(record["gold_sids"]),
        "gold_a_bucket": _bucket(_gold_a_count(record["gold_sids"])),
        "classification": "NORMAL-SIGNAL" if primary_std > 0 else (
            "DEAD-ZERO" if all(float(value) == 0.0 for value in rewards) else "ZERO-STD-OTHER"
        ),
        "primary_rewards": [float(value) for value in rewards.cpu()],
        "primary_std": primary_std,
        "branch": branch,
        "s_aux": scores,
        "s_aux_std": float(torch.tensor(scores).std(correction=0)),
        "primary_loss": float(primary_loss.detach()),
        "weighted_auxiliary_loss": float(auxiliary_loss.detach()),
        "completion_sha256": [_completion_sha256(item["completion_ids"]) for item in record["candidates"]],
    })
    return metrics


def _nothink_gradient(model, tokenizer, parameters, by_group):
    trace = json.loads(NOTHINK_TRACE.read_text(encoding="utf-8"))
    group = trace["groups"][0]
    row = by_group[group["group_id"]]["no_think"]
    completion_ids = group.get("completion_ids")
    if not completion_ids or len(completion_ids) != 8:
        raise RuntimeError("NoThink audit requires exact raw completion IDs from the zero-update probe")
    predicted = []
    positions = []
    for ids in completion_ids:
        value, position = locate_final_sid_a_position(ids, tokenizer)
        predicted.append(value)
        positions.append(position)
    rewards = [float(value) for value in group["rewards"]]
    plan = build_nothink_rescue_plan(rewards, predicted, positions, group["gold_as"])
    logps, mask = _per_token_logps(model, tokenizer, row["prompt"], completion_ids)
    primary_advantages = group_advantages_population(torch.tensor(rewards, device=model.device), 8)
    ratio = torch.exp(logps - logps.detach())
    primary_per_sample = (-(ratio * primary_advantages[:, None]) * mask).sum(-1) / mask.sum(-1)
    primary_loss = ROUTE_LOSS_W["no_think"] * primary_per_sample.mean()
    rescue_loss = nothink_unlikelihood_loss(
        logps,
        torch.tensor(positions, dtype=torch.long, device=model.device),
        torch.tensor(plan.frequency_weights, dtype=torch.float32, device=model.device),
        torch.full((8,), plan.coefficient, dtype=torch.float32, device=model.device),
    )
    token_gradient = torch.autograd.grad(rescue_loss, logps, retain_graph=True)[0]
    active_positions = {(index, position) for index, position in enumerate(positions)}
    nonzero_positions = {
        (row_index, column_index)
        for row_index, column in enumerate(token_gradient)
        for column_index, value in enumerate(column)
        if abs(float(value)) > 0.0
    }
    metrics = _three_backwards(model, parameters, primary_loss, rescue_loss)
    inactive = trace["groups"][1]
    inactive_plan = build_nothink_rescue_plan(
        inactive["rewards"], inactive["predicted_as"], inactive["sa_positions"], inactive["gold_as"]
    )
    metrics.update({
        "group_id": group["group_id"],
        "primary_rewards": rewards,
        "predicted_as": predicted,
        "frequency_weights": list(plan.frequency_weights),
        "concentration": plan.concentration,
        "max_repeated_wrong_a_frequency": max(plan.frequency_weights),
        "gold_unique_a": plan.gold_unique_a,
        "lambda_a": plan.lambda_a,
        "coefficient_lambda_times_c": plan.coefficient,
        "positions": positions,
        "rescue_active": plan.active,
        "rescue_loss": float(rescue_loss.detach()),
        "active_token_count": len(nonzero_positions),
        "sa_only_gradient_exact": nonzero_positions == active_positions,
        "inactive_case": {
            "group_id": inactive["group_id"],
            "rewards": inactive["rewards"],
            "active": inactive_plan.active,
            "coefficient": inactive_plan.coefficient,
        },
    })
    return metrics


def _logit_direction_proof():
    logits = torch.tensor([0.3, -0.2, 1.1], dtype=torch.float64, requires_grad=True)
    wrong_index = 1
    probability = logits.softmax(-1)[wrong_index]
    loss = -torch.log1p(-probability)
    gradient = torch.autograd.grad(loss, logits)[0]
    updated = logits.detach() - 0.1 * gradient
    updated_probability = updated.softmax(-1)[wrong_index]
    return {
        "probability_before": float(probability),
        "d_loss_d_wrong_logit": float(gradient[wrong_index]),
        "gradient_descent_probability_after": float(updated_probability),
        "direction_correct": bool(gradient[wrong_index] > 0 and updated_probability < probability),
    }


def _summary(groups):
    normal = [item for item in groups if item["classification"] == "NORMAL-SIGNAL"][:16]
    if len(normal) < 8:
        raise RuntimeError(f"gradient audit produced only {len(normal)} normal-signal groups")
    ratios = sorted(item["ratio"] for item in normal)
    cosines = [item["cosine"] for item in normal if item["cosine"] is not None]
    p90 = ratios[max(0, math.ceil(0.9 * len(ratios)) - 1)]
    return {
        "normal_signal_group_count": len(normal),
        "group_ids": [item["group_id"] for item in normal],
        "ratio": {
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios),
            "p90": p90,
            "max": max(ratios),
        },
        "cosine": {
            "mean": statistics.fmean(cosines) if cosines else None,
            "median": statistics.median(cosines) if cosines else None,
            "min": min(cosines) if cosines else None,
        },
        "domains": sorted({item["domain"] for item in normal}),
        "gold_a_buckets": sorted({item["gold_a_bucket"] for item in normal}),
    }


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("gradient-budget audit requires exactly four ranks")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name.lower()
    parameters = _lora_parameters(model)
    by_group = _read_rows()
    selected = _select_generated_rows(by_group)
    monitor = SimpleNamespace(enabled=True)
    beam32_fn = make_beam32_fn(model, tokenizer, monitor_writer=monitor)

    local_metrics = []
    domain = DOMAINS[rank]
    for round_index, row in enumerate(selected[domain]):
        model.eval()
        record = _generated_record(
            model, tokenizer, beam32_fn, row,
            int(TRAIN_SEED) + rank + 1000 * (round_index + 1),
            "generated_bata_stratified",
        )
        _release_cuda_cache()
        model.train()
        local_metrics.append(_think_gradient(model, tokenizer, parameters, record))
        del record
        _release_cuda_cache()

    probe_row = by_group[PROBE_GROUP_IDS[rank]]["think"]
    model.eval()
    record = _generated_record(
        model, tokenizer, beam32_fn, probe_row,
        int(PROBE_SEED) + rank,
        "generated_bata_fixed_probe",
    )
    _release_cuda_cache()
    model.train()
    local_metrics.append(_think_gradient(model, tokenizer, parameters, record))
    del record
    _release_cuda_cache()

    nothink = _nothink_gradient(model, tokenizer, parameters, by_group) if rank == 0 else None
    gathered = [None] * 4
    dist.all_gather_object(gathered, local_metrics)
    if rank == 0:
        groups = [item for rank_items in gathered for item in rank_items]
        payload = {
            "model": BASE,
            "adapter": ADAPTER,
            "world_size": 4,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "trainable_scope": "LoRA only",
            "think": {
                "groups": groups,
                "normal_signal_summary": _summary(groups),
                "dead_zero": [item for item in groups if item["classification"] == "DEAD-ZERO"],
            },
            "nothink": nothink,
            "wrong_a_logit_direction": _logit_direction_proof(),
            "route_weighting": {
                "think": "L_total = 1.0 * L_primary_unweighted + 0.10 * L_Think_aux",
                "nothink": "L_total = 0.5 * L_primary_unweighted + L_NoThink_rescue",
                "rescue_receives_route_0.5": False,
                "gradient_accumulation_steps": 1,
            },
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
