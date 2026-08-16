# -*- coding: utf-8 -*-
"""Correctness round tests (CPU):
1. route-specific reward routing + NoThink beam call count == 0
2. RouteAwareRepeatSampler true dynamic G (4x4 / 2x8, group_id asserts, reuse)
3. Think/NoThink loss multiplier math
4. hierarchical shared-prefix dedup (think_credits)
5. shared prompt renderer parity (real prompt)
"""
import sys
sys.path.insert(0, "/data/GRPO/scripts")

import json
import torch

import trl_import_fix
from grpo_trl_trainer import (
    build_route_dataset, RouteAwareRepeatSampler,
    make_nothink_reward_func, make_think_reward_func,
    ROUTE_G, ROUTE_LOSS_W, M_THINK, M_NO,
)
from grpo_sid import think_credits, think_reward

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
failures = []

def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

# ============ 1. route-specific reward ============
beam_calls = {"n": 0}
def counting_beam(prompts, completions, completion_ids, gold_sets):
    beam_calls["n"] += 1
    return [think_reward([("video", 1, 2, 3)], gs)[0] for gs in gold_sets]

think_rf = make_think_reward_func(beam32_fn=counting_beam)
no_rf = make_nothink_reward_func()
golds = [["<|video_begin|><s_a_1><s_b_2><s_c_3>"]] * 4

r_think = think_rf(prompts=["p"] * 4, completions=["c"] * 4,
                   completion_ids=[[1, 2]] * 4, all_gold_sids=golds, route=["think"] * 4)
r_no_on_think = no_rf(prompts=["p"] * 4, completions=["c"] * 4,
                      all_gold_sids=golds, route=["think"] * 4)
check("Think batch: think_reward numeric", all(isinstance(x, float) for x in r_think))
check("Think batch: nothink_reward all None", r_no_on_think == [None] * 4)
check("Think batch: beam called once (think subset)", beam_calls["n"] == 1)

beam_calls["n"] = 0
r_no = no_rf(prompts=["p"] * 4, completions=["c"] * 4,
             all_gold_sids=golds, route=["no_think"] * 4)
r_think_on_no = think_rf(prompts=["p"] * 4, completions=["c"] * 4,
                         completion_ids=[[1, 2]] * 4, all_gold_sids=golds,
                         route=["no_think"] * 4)
check("NoThink batch: nothink_reward numeric", all(isinstance(x, float) for x in r_no))
check("NoThink batch: think_reward all None", r_think_on_no == [None] * 4)
check("NoThink batch: Beam32 call count == 0", beam_calls["n"] == 0, f"calls={beam_calls['n']}")

beam_calls["n"] = 0
r_mix = think_rf(prompts=["p"] * 4, completions=["c"] * 4, completion_ids=[[1, 2]] * 4,
                 all_gold_sids=golds, route=["no_think", "think", "no_think", "think"])
check("Mixed batch: beam runs on think subset only", beam_calls["n"] == 1)
check("Mixed batch: routing per sample", r_mix[0] is None and isinstance(r_mix[1], float)
      and r_mix[2] is None and isinstance(r_mix[3], float))

try:
    no_rf(prompts=["p"], completions=["c"], all_gold_sids=golds)
    check("missing route column raises", False)
except RuntimeError:
    check("missing route column raises", True)

# ============ 2. RouteAwareRepeatSampler ============
ds = build_route_dataset(DATA, n_groups=8, seed=20260816, chunk=4)
rows = list(ds)
sampler = RouteAwareRepeatSampler(ds, repeat_count=2, shuffle=False)
idx_seq = list(sampler)
check("sampler __len__ matches", len(idx_seq) == len(sampler) == 192, str(len(idx_seq)))

def route_of(i):
    return rows[i]["route"]
def gid_of(i):
    return rows[i]["recommendation_group_id"]

ok_struct = True
for b in range(0, len(idx_seq), 16):
    if len({route_of(i) for i in idx_seq[b:b + 16]}) != 1:
        ok_struct = False
check("every 16-sample global batch route-homogeneous", ok_struct)

ok_g4 = ok_g8 = ok_adj = ok_gid = True
for b in range(0, len(idx_seq), 16):
    batch = idx_seq[b:b + 16]
    rt = route_of(batch[0])
    g = ROUTE_G[rt]
    for start in range(0, 16, g):
        seg = batch[start:start + g]
        if len(set(seg)) != 1:
            ok_adj = False
        if len({gid_of(i) for i in seg}) != 1:
            ok_gid = False
    uniq = [batch[i] for i in range(0, 16, g)]
    if len(set(uniq)) != 16 // g or any(batch.count(u) != g for u in uniq):
        if rt == "think":
            ok_g4 = False
        else:
            ok_g8 = False
check("G=4: exactly 4 unique prompts x 4 repeats", ok_g4)
check("G=8: exactly 2 unique prompts x 8 repeats", ok_g8)
check("repeats contiguous (A A A A B B B B ...)", ok_adj)
check("group_id identical inside every view(-1,G)", ok_gid)

ok_reuse = all(idx_seq[b:b + 16] == idx_seq[b + 16:b + 32]
               for b in range(0, len(idx_seq), 32))
check("num_iterations=2: chunk replayed back-to-back (rollout reuse)", ok_reuse)

seq_routes = [route_of(idx_seq[b]) for b in range(0, 192, 16)]
check("route sequence per rollout", seq_routes ==
      ["think", "think", "no_think", "no_think", "no_think", "no_think",
       "think", "think", "no_think", "no_think", "no_think", "no_think"],
      str(seq_routes))

# ============ 3. loss multiplier math ============
w_t, w_n = ROUTE_LOSS_W["think"], ROUTE_LOSS_W["no_think"]
check("multipliers 1.0 / 0.5", w_t == 1.0 and w_n == 0.5)
check("single-group total weight equal: 4*1.0 == 8*0.5",
      M_THINK * w_t == M_NO * w_n, f"{M_THINK*w_t} vs {M_NO*w_n}")
N_G = 1549
tot_t = N_G * M_THINK * w_t
tot_n = N_G * M_NO * w_n
check("epoch route totals equal (1549 groups each)",
      abs(tot_t - tot_n) < 1e-9, f"{tot_t} vs {tot_n}")
per_sample_t = torch.full((16,), 2.0)
per_sample_n = torch.full((16,), 4.0)
loss_t = (per_sample_t * w_t).mean()
loss_n = (per_sample_n * w_n).mean()
check("weighted per-sample mean math", abs(loss_t.item() - 2.0) < 1e-6
      and abs(loss_n.item() - 2.0) < 1e-6, f"{loss_t.item()} vs {loss_n.item()}")

# ============ 4. hierarchical shared-prefix dedup ============
g12 = {("video", 1, 2, 1), ("video", 1, 2, 2)}
cr, ec, ac, a2 = think_credits([("video", 1, 2, 1)], g12)
check("exact covers sibling AB: credits [8] only",
      cr == [8.0] and ec == 1 and ac == 0 and a2 == 0, str(cr))
cr, ec, ac, a2 = think_credits([("video", 1, 2, 1), ("video", 1, 2, 9)], g12)
check("AB-matched non-gold still no extra AB",
      cr == [8.0] and ec == 1 and ac == 0 and a2 == 0, str(cr))
g_ab = {("video", 1, 2, 1), ("video", 3, 4, 5)}
cr, ec, ac, a2 = think_credits([("video", 1, 2, 1), ("video", 3, 4, 9)], g_ab)
check("free AB prefix credited 2", cr == [8.0, 2.0] and ec == 1 and ac == 1 and a2 == 0, str(cr))
g_a = {("video", 1, 2, 1), ("video", 7, 8, 9)}
cr, ec, ac, a2 = think_credits([("video", 7, 8, 99)], g_a)
check("AB covers A prefix: no extra 0.5",
      cr == [2.0] and ec == 0 and ac == 1 and a2 == 0, str(cr))
g_pure_a = {("video", 1, 2, 1), ("video", 4, 5, 6)}
cr, ec, ac, a2 = think_credits([("video", 4, 9, 9)], g_pure_a)
check("pure A prefix credited 0.5", cr == [0.5] and ec == 0 and ac == 0 and a2 == 1, str(cr))
cr, ec, ac, a2 = think_credits([("video", 9, 9, 9)], g12)
check("no hit -> empty credits", cr == [] and ec == 0 and ac == 0 and a2 == 0)
r, _, _, _ = think_reward([("video", 1, 2, 1)], g12)
check("reward value exact 8.0 (unchanged)", r == 8.0)
r, _, _, _ = think_reward([("video", 1, 2, 1), ("video", 3, 4, 9)], g_ab)
check("reward [8,2] -> 9.0 geometric decay (unchanged)", abs(r - 9.0) < 1e-9, str(r))
r, _, _, _ = think_reward([("video", 4, 9, 9)], g_pure_a)
check("reward 0.5 (unchanged)", abs(r - 0.5) < 1e-9)

# ============ 5. shared renderer parity ============
from grpo_model import render_prompt, encode_prompt
from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/data/models/onereason-8b-pretrain-competition", trust_remote_code=True)
tpl = TEMPLATES["qwen3_nothink"]
rows_all = [json.loads(l) for l in open(DATA, encoding="utf-8")]
p = rows_all[0]["prompt"]
a_ids = tpl.encode_multiturn(
    tok, [{"role": "user", "content": p}, {"role": "assistant", "content": ""}],
    system=None)[0][0]
b_ids = tok.encode(render_prompt(tok, p), add_special_tokens=False)
c_ids = encode_prompt(tok, p)
check("SFT == TRL render ids", a_ids == b_ids, f"{len(a_ids)} vs {len(b_ids)}")
check("SFT == Beam ids", a_ids == c_ids, f"{len(a_ids)} vs {len(c_ids)}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL CORRECTNESS V2 TESTS PASSED")
