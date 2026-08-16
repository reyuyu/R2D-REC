# -*- coding: utf-8 -*-
"""Diagnose cbs=2 beam output for the SHORT context (left-padded to longer)."""
import collections, json, random, sys
import torch
sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt, generate_batch
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
                             pad_token_id=tokenizer.pad_token_id,
                             stopping_criteria=StoppingCriteriaList([_Stop()]))[0]
        cids = gen[len(pids):].tolist()
        try:
            pos = cids.index(tid)
            cids = cids[:pos + 1]
        except ValueError:
            pass
        contexts.append(pids + cids)
print("ctx lens:", [len(c) for c in contexts], flush=True)

pad_id = tokenizer.pad_token_id

def run_batch(idxs, tag):
    inps = [contexts[i] for i in idxs]
    texts = generate_batch(model, tokenizer, inps, max_new_tokens=128,
                           num_beams=32, num_return_sequences=32)
    for local, gi in enumerate(idxs):
        sub = texts[local * 32:(local + 1) * 32]
        sids = [final_sid(t) for t in sub]
        inv = sum(1 for s in sids if s is None)
        print(f"[{tag}] ctx{gi} len={len(contexts[gi])} invalid={inv}/32", flush=True)
        print(f"   cand0 head: {sub[0][:70]!r}", flush=True)
        print(f"   cand1 head: {sub[1][:70]!r}", flush=True)
        print(f"   cand2 head: {sub[2][:70]!r}", flush=True)
    return texts

# determinism: cbs=1 twice
run_batch([0], "cbs1-a")
run_batch([0], "cbs1-b")
# cbs=2 with short+long
run_batch([0, 1], "cbs2")
run_batch([2, 3], "cbs2")
print("PAD_ID:", pad_id, flush=True)
