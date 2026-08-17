# -*- coding: utf-8 -*-
"""Rollout-only correctness check (single GPU, NO training/backward/optimizer).
Runs 1 Think + 1 NoThink generation batch through the real trainer path:
- group_id hard assert inside _generate_and_score_completions
- reward routing (NaN pattern per reward func)
- Beam32 call count == 0 for NoThink
- true sampler group_id expansion"""
import json
import os
import sys
import time

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29599")
sys.path.insert(0, "/data/GRPO/scripts")
import trl_import_fix  # noqa

import torch
from grpo_trl_trainer import (RecGRPOTrainer, build_route_dataset,
                              make_nothink_reward_func, make_think_reward_func, ROUTE_G)
from grpo_sid import think_reward
from grpo_model import load_model
from trl import GRPOConfig

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
device = "cuda:0"
model, tokenizer, template = load_model(device)
for n, p in model.named_parameters():
    p.requires_grad = False  # NO training this round
model.eval()

ds = build_route_dataset(DATA, n_groups=8, seed=20260816, chunk=8)
rows = list(ds)

stats = dict(n_cot=0, calls=0)
def beam32_fn(prompts, completions, completion_ids, gold_sets):
    stats["calls"] += 1
    stats["n_cot"] += len(completion_ids)
    return [think_reward([("video", 1, 2, 3)], gs)[0] for gs in gold_sets]

cfg = GRPOConfig(
    output_dir="/tmp/grpo_rollout_check",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_completion_length=512,
    num_iterations=2,
    steps_per_generation=1,
    beta=0.0, epsilon=0.2, loss_type="grpo", scale_rewards="group",
    disable_dropout=True, learning_rate=1e-6, max_grad_norm=1.0,
    lr_scheduler_type="constant", max_steps=0, save_strategy="no",
    report_to="none", use_vllm=False, shuffle_dataset=False, seed=20260816,
)
trainer = RecGRPOTrainer(
    model=model, args=cfg, processing_class=tokenizer, train_dataset=ds,
    reward_funcs=[make_nothink_reward_func(tokenizer=tokenizer),
                  make_think_reward_func(beam32_fn=beam32_fn)],
)
trainer.model.eval()
print("reward funcs:", trainer.reward_func_names, flush=True)

sampler = trainer._get_train_sampler()
idx_seq = list(sampler)
print("sampler len:", len(idx_seq), "chunks:", len(sampler._chunks), flush=True)
print("chunks:", [(r, len(inds)) for r, inds in sampler._chunks], flush=True)
print("first chunk gid expansion (Think):",
      [rows[i]["recommendation_group_id"] for i in sampler._chunks[0][1]], flush=True)
# find first NoThink chunk
n_chunk = next((inds for r, inds in sampler._chunks if r == "no_think"), None)
print("first NoThink chunk gid expansion:",
      [rows[i]["recommendation_group_id"] for i in n_chunk], flush=True)

def run_batch(tag, idxs):
    gen_batch = [dict(rows[i]) for i in idxs]
    routes = {g["route"] for g in gen_batch}
    print(f"\n=== {tag}: route={gen_batch[0]['route']} unique={len(routes)} ===", flush=True)
    # FULL path: _prepare_inputs sets route temp/top_p/G, generates, buffers,
    # splits -> returns loss-ready inputs (route_id included).
    trainer._step = 0
    trainer._buffered_inputs = None
    t0 = time.time()
    out = trainer._prepare_inputs(gen_batch)
    dt = time.time() - t0
    assert "route_id" in out, "route_id missing from generation output"
    assert bool((out["route_id"] == out["route_id"][0]).all()), "route_id mixed" 
    rw = {k: list(v) for k, v in trainer._logs["rewards"].items()}
    # last 16 entries per func
    tail = {k: v[-16:] for k, v in rw.items()}
    import math
    nan_pat = {k: [("NaN" if (x != x) else round(float(x), 3)) for x in v] for k, v in tail.items()}
    print(f"{tag} rollout_sec={dt:.1f}s", flush=True)
    for k, pat in nan_pat.items():
        print(f"{tag} {k}: {pat}", flush=True)
    print(f"{tag} beam calls={stats['calls']} n_cot={stats['n_cot']}", flush=True)
    return out

# Think batch: first chunk (2 unique x 4 repeats = 8 samples)
think_idx = idx_seq[:8]
run_batch("THINK", think_idx)
# NoThink batch: first no_think chunk (1 unique x 8 repeats = 8 samples)
no_idx = list(n_chunk) * 8
print("\nNoThink batch gid order:", [rows[i]["recommendation_group_id"] for i in no_idx], flush=True)
run_batch("NOTHINK", no_idx)

print("\n=== RESULT ===", flush=True)
print(json.dumps({"beam_calls": stats["calls"], "beam_cot_total": stats["n_cot"],
                  "sampler_len": len(idx_seq), "chunks": len(sampler._chunks)},
                 indent=2), flush=True)
