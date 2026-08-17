# -*- coding: utf-8 -*-
"""Float-noise vs state-pollution: batch-internal consistency + diff scale."""
import collections, json, random, sys
import torch
sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt, generate_batch
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
short, long_ = contexts[0], contexts[3]
print("short", len(short), "long", len(long_), flush=True)

def beam(inps):
    _, ids = generate_batch(model, tokenizer, inps, max_new_tokens=128,
                            num_beams=32, num_return_sequences=32, return_ids=True)
    return ids

def diff_stats(x, y, label):
    n_diff = sum(1 for a, b in zip(x, y) if a != b)
    first = None
    for k, (a, b) in enumerate(zip(x, y)):
        if a != b:
            first = k
            break
    # first differing token position across all differing candidates
    pos = []
    for a, b in zip(x, y):
        if a != b:
            m = min(len(a), len(b))
            p = next((i for i in range(m) if a[i] != b[i]), m)
            pos.append(p)
    print(f"{label}: diff_candidates={n_diff}/32 first_cand={first} "
          f"diff_positions={sorted(set(pos))[:10]}", flush=True)

with torch.inference_mode():
    A = beam([short])
    B = beam([long_])
    C = beam([short, long_])
    D = beam([long_, short])
    E = beam([short, long_])
    diff_stats(C[:32], A, "C.short  vs A (single)")
    diff_stats(C[32:], B, "C.long   vs B (single)")
    diff_stats(D[32:], A, "D.short  vs A (single)")
    diff_stats(D[:32], B, "D.long   vs B (single)")
    diff_stats(C[:32], D[32:], "C.short  vs D.short (batch-internal)")
    diff_stats(C[32:], D[:32], "C.long   vs D.long (batch-internal)")
    diff_stats(C[:32], E[:32], "C.short  vs E.short (same-shape repeat)")
    diff_stats(C[32:], E[32:], "C.long   vs E.long (same-shape repeat)")
print("DONE", flush=True)
