#!/usr/bin/env python3
"""No-update BATA probe for real NoThink all-zero groups.

This diagnostic is not part of the training path. It performs one policy
forward only to report the sampled s_A log probability requested by the smoke
audit; DSR training reuses its existing policy forward.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

from grpo_model import encode_prompt, generate_batch, load_model
from grpo_sid import final_sid, parse_sid, q_reward
from gr_rec_dsr_v1.dsr_objectives import build_nothink_rescue_plan
from gr_rec_dsr_v1.dsr_runtime import audit_sa_tokenization, locate_final_sid_a_position


DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
GROUP_IDS = (
    "a0468cf984189a88f3f86f8e0f6792303fcd50f41cde3dd12c0d75965625833f",
    "e50063759a750bb5a033d1096f22824d55d488fed783766b453651e2bd15d2ce",
)
OUTPUT = "/data/GRPO/outputs/GR-REC-DSR-V1-NOTHINK-PROBE.json"


def load_rows():
    wanted = set(GROUP_IDS)
    rows = {}
    with open(DATA, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gid = row["recommendation_group_id"]
            if gid in wanted and row["route"] == "no_think":
                rows[gid] = row
    if set(rows) != wanted:
        raise RuntimeError(f"missing probe rows: {wanted.difference(rows)}")
    return rows


@torch.inference_mode()
def sampled_sa_logp(model, prompt_ids, completion_ids, position):
    if position < 0:
        return None
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    logits = model(input_ids=ids, use_cache=False).logits
    prediction_position = len(prompt_ids) + position - 1
    token_id = completion_ids[position]
    return float(torch.log_softmax(logits[0, prediction_position].float(), dim=-1)[token_id])


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(20260816 + rank)
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    tokenizer_audit = audit_sa_tokenization(tokenizer)
    rows = load_rows()
    gid = GROUP_IDS[rank // 2]
    row = rows[gid]
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    _, completion_ids = generate_batch(
        model,
        tokenizer,
        [prompt_ids] * 4,
        max_new_tokens=2048,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        return_ids=True,
    )
    golds = {sid for value in row["all_gold_sids"] if (sid := parse_sid(value)) is not None}
    local = []
    for ids in completion_ids:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        sid = final_sid(text)
        predicted_a, position = locate_final_sid_a_position(ids, tokenizer)
        logp = sampled_sa_logp(model, prompt_ids, ids, position)
        local.append({
            "group_id": gid,
            "completion": text,
            "completion_ids": list(ids),
            "parsed_sid": sid,
            "predicted_a": predicted_a,
            "sa_position": position,
            "sa_logp": logp,
            "sa_probability": math.exp(logp) if logp is not None else None,
            "reward": q_reward(sid, golds),
            "gold_as": sorted({gold[1] for gold in golds}),
        })
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if rank == 0:
        records = [item for rank_records in gathered for item in rank_records]
        groups = []
        for group_index, group_id in enumerate(GROUP_IDS):
            group = records[group_index * 8:(group_index + 1) * 8]
            if {item["group_id"] for item in group} != {group_id}:
                raise RuntimeError("probe gather did not preserve G=8 group alignment")
            plan = build_nothink_rescue_plan(
                [item["reward"] for item in group],
                [item["predicted_a"] for item in group],
                [item["sa_position"] for item in group],
                group[0]["gold_as"],
            )
            terms = []
            for item, weight in zip(group, plan.frequency_weights):
                probability = item["sa_probability"]
                ul = -math.log1p(-min(probability, 1.0 - 1e-6)) if probability is not None else None
                terms.append(weight * ul if ul is not None else 0.0)
            rescue_loss = plan.coefficient * sum(terms) / 8.0 if plan.active else 0.0
            groups.append({
                "group_id": group_id,
                "rewards": [item["reward"] for item in group],
                "predicted_as": [item["predicted_a"] for item in group],
                "frequency_weights": list(plan.frequency_weights),
                "concentration": plan.concentration,
                "gold_as": group[0]["gold_as"],
                "gold_unique_a": plan.gold_unique_a,
                "lambda_a": plan.lambda_a,
                "sa_positions": [item["sa_position"] for item in group],
                "sa_logps": [item["sa_logp"] for item in group],
                "sa_probabilities": [item["sa_probability"] for item in group],
                "active": plan.active,
                "rescue_loss": rescue_loss,
                "completions": [item["completion"] for item in group],
                "completion_ids": [item["completion_ids"] for item in group],
            })
        payload = {
            "parent": "BATA baseline",
            "optimizer_steps": 0,
            "model_updates": False,
            "sampling": {"g": 8, "temperature": 1.0, "top_p": 1.0},
            "tokenizer_audit": tokenizer_audit,
            "groups": groups,
        }
        with open(OUTPUT, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
