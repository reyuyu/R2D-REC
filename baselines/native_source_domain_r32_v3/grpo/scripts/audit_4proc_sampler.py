# -*- coding: utf-8 -*-
"""4-process sampler/DataLoader audit (torchrun --nproc_per_node=4).
NO 8B model: tiny dummy Llama. Real trainer path:
RouteAwareRepeatSampler -> DataLoader -> Accelerator.prepare.
Checks gathered global batches: Think 4 unique x4, NoThink 2 unique x8,
view(-1,G) group_id identical, num_iterations=2 back-to-back reuse,
and SFT-vs-TRL prompt_ids parity through the real _generate_single_turn."""
import json
import os
import sys

sys.path.insert(0, "/data/GRPO/scripts")
import trl_import_fix  # noqa

import torch
import torch.distributed as dist

from grpo_trl_trainer import (RecGRPOTrainer, build_route_dataset,
                              make_nothink_reward_func, make_think_reward_func,
                              ROUTE_G, ROUTE_ID)
from grpo_sid import think_reward
from trl import GRPOConfig
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from llamafactory.data.template import TEMPLATES

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
is_main = rank == 0

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
ds = build_route_dataset(DATA, n_groups=8, seed=20260816, chunk=8)  # 16 records
rows = list(ds)

tok = AutoTokenizer.from_pretrained(
    "/data/models/onereason-8b-pretrain-competition", trust_remote_code=True)
tok.pad_token = tok.eos_token
mcfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                   num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=4, max_position_embeddings=131072)
model = LlamaForCausalLM(mcfg)

def fake_beam(prompts, completions, completion_ids, gold_sets):
    return [think_reward([("video", 1, 2, 3)], gs)[0] for gs in gold_sets]

cfg = GRPOConfig(
    output_dir="/tmp/audit4",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=8192,
    max_completion_length=64,
    num_iterations=2,
    steps_per_generation=1,
    beta=0.0, epsilon=0.2, loss_type="grpo", scale_rewards="group",
    disable_dropout=True, learning_rate=1e-6, max_grad_norm=1.0,
    lr_scheduler_type="constant", max_steps=0, save_strategy="no",
    report_to="none", use_vllm=False, shuffle_dataset=False, seed=20260816,
)
trainer = RecGRPOTrainer(
    model=model, args=cfg, processing_class=tok, train_dataset=ds,
    reward_funcs=[make_nothink_reward_func(tokenizer=tok),
                  make_think_reward_func(beam32_fn=fake_beam)],
)
if is_main:
    print("gen_batch_size:", cfg.generation_batch_size, flush=True)

def gather_list(x):
    out = [None] * world
    dist.all_gather_object(out, x)
    return [v for o in out for v in o]

results = {}
dl = trainer.get_train_dataloader()
it = iter(dl)
seen_batches = []
for step in range(8):
    batch = next(it)
    # batch is a list of example dicts (TRL collate), local 4 samples
    local_gids = [x["recommendation_group_id"] for x in batch]
    local_routes = [x["route"] for x in batch]
    gids_all = gather_list(local_gids)
    routes_all = gather_list(local_routes)
    route = routes_all[0]
    G = ROUTE_G[route]
    assert len(set(routes_all)) == 1, f"mixed routes in global batch: {routes_all}"
    assert len(gids_all) == 16, f"gathered {len(gids_all)} != 16"
    # view(-1,G) group_id identical
    ok_gid = all(len(set(gids_all[i:i + G])) == 1 for i in range(0, 16, G))
    # unique count
    uniq = [gids_all[i] for i in range(0, 16, G)]
    ok_unique = len(set(uniq)) == 16 // G
    ok_repeat = all(gids_all.count(u) == G for u in uniq)
    seen_batches.append({"step": step, "route": route, "gids": gids_all,
                         "ok_gid": ok_gid, "ok_unique": ok_unique, "ok_repeat": ok_repeat})
    if is_main:
        print(f"batch{step} route={route} G={G} ok_gid={ok_gid} "
              f"ok_unique={ok_unique} ok_repeat={ok_repeat}", flush=True)
        print(f"  gids: {gids_all}", flush=True)

# reuse: batch0 == batch1, batch2 == batch3, ... (chunk replay)
reuse = [seen_batches[i]["gids"] == seen_batches[i + 1]["gids"]
         for i in range(0, 8, 2)]
if is_main:
    print("reuse:", reuse, flush=True)

# ---- prompt parity through REAL _generate_single_turn ----
prompts = [rows[0]["prompt"], rows[1]["prompt"], rows[2]["prompt"], rows[3]["prompt"]]
prompt_ids_list, completion_ids_list, logprobs, fwd = trainer._generate_single_turn(prompts)
tpl = TEMPLATES["qwen3_nothink"]
sft_ids = [tpl.encode_multiturn(
    tok, [{"role": "user", "content": p}, {"role": "assistant", "content": ""}],
    system=None)[0][0] for p in prompts]
parity = [pids == sids for pids, sids in zip(prompt_ids_list, sft_ids)]
if is_main:
    print("prompt parity (TRL _generate_single_turn vs SFT ids):", parity, flush=True)
    print("prompt lens TRL:", [len(x) for x in prompt_ids_list],
          "SFT:", [len(x) for x in sft_ids], flush=True)

if is_main:
    results = {
        "world": world, "gen_batch_size": cfg.generation_batch_size,
        "batches": seen_batches,
        "reuse_pairs": reuse,
        "prompt_parity": parity,
        "prompt_len_trl": [len(x) for x in prompt_ids_list],
        "prompt_len_sft": [len(x) for x in sft_ids],
        "route_ids": {k: v for k, v in ROUTE_ID.items()},
    }
    with open("/data/GRPO/logs/audit_4proc.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=== AUDIT4 DONE ===", flush=True)
dist.barrier()
