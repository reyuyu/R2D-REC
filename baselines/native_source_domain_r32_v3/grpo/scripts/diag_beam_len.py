# -*- coding: utf-8 -*-
"""Decisive: equal-length batch (no pad) vs unequal (left pad), FA2 vs eager."""
import collections, json, random, sys
import torch
sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt
from grpo_sid import final_sid
from transformers import StoppingCriteria, StoppingCriteriaList

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
by_group = collections.defaultdict(dict)
for r in rows:
    by_group[r["recommendation_group_id"]][r["route"]] = r
gids = sorted(by_group.keys())
rng = random.Random(20260816)
rng.shuffle(gids)
prompts = [by_group[g]["think"]["prompt"] for g in gids[:4]]

model, tokenizer, template = load_model("cuda:0")
model.eval()
tid = tokenizer.encode("</think>", add_special_tokens=False)[0]
pad_id = tokenizer.pad_token_id

class _Stop(StoppingCriteria):
    def __call__(self, input_ids, scores, **kw):
        return input_ids[:, -1] == tid

contexts = []
with torch.inference_mode():
    for p in prompts:
        pids = encode_prompt(tokenizer, p)
        inp = {"input_ids": torch.tensor([pids], device=model.device, dtype=torch.long),
               "attention_mask": torch.ones(1, len(pids), device=model.device, dtype=torch.long)}
        gen = model.generate(**inp, max_new_tokens=2048, do_sample=True,
                             temperature=0.9, top_p=0.95, num_return_sequences=1,
                             pad_token_id=pad_id,
                             stopping_criteria=StoppingCriteriaList([_Stop()]))[0]
        cids = gen[len(pids):].tolist()
        try:
            pos = cids.index(tid)
            cids = cids[:pos + 1]
        except ValueError:
            pass
        contexts.append(pids + cids)
print("ctx lens:", [len(c) for c in contexts], flush=True)

def run_beam(inps, tag):
    B = len(inps)
    max_len = max(len(x) for x in inps)
    input_t = torch.full((B, max_len), pad_id, dtype=torch.long, device=model.device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=model.device)
    for i, ids in enumerate(inps):
        input_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attn[i, :len(ids)] = 1
    out = model.generate(inputs=input_t, attention_mask=attn,
                         max_new_tokens=128, num_beams=32, num_return_sequences=32,
                         pad_token_id=pad_id)
    for local in range(B):
        sub = []
        for k in range(32):
            row = out[local * 32 + k, len(inps[local]):].tolist()
            while row and row[-1] == pad_id:
                row.pop()
            sub.append(tokenizer.decode(row, skip_special_tokens=False))
        inv = sum(1 for s in (final_sid(t) for t in sub) if s is None)
        print(f"[{tag}] ctx{local} len={len(inps[local])} invalid={inv}/32 "
              f"head={sub[0][:40]!r}", flush=True)

with torch.inference_mode():
    # equal-length: same context twice (no pad needed)
    run_beam([contexts[0], contexts[0]], "equal-len(2xctx0)")
    # unequal: ctx0 (short) + ctx1 (long) -> left pad
    run_beam([contexts[0], contexts[1]], "unequal(ctx0,ctx1)")
    # unequal reversed: long first
    run_beam([contexts[1], contexts[0]], "unequal(ctx1,ctx0)")
print("DONE", flush=True)
