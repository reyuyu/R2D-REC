# -*- coding: utf-8 -*-
"""Equal-length batch (NO padding at all) vs single-run: if candidates still
differ, the cross-shape difference is pure float noise, not padding."""
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

def beam(inps):
    _, ids = generate_batch(model, tokenizer, inps, max_new_tokens=128,
                            num_beams=32, num_return_sequences=32, return_ids=True)
    return ids

with torch.inference_mode():
    A = beam([contexts[0]])
    # equal-length batch: same context twice, NO padding anywhere
    E = beam([contexts[0], contexts[0]])
    n_diff = sum(1 for a, b in zip(E[:32], A) if a != b)
    print(f"equal-len batch vs single: diff_candidates={n_diff}/32", flush=True)
    n_diff2 = sum(1 for a, b in zip(E[32:], A) if a != b)
    print(f"equal-len batch 2nd copy vs single: diff_candidates={n_diff2}/32", flush=True)
print("DONE", flush=True)
