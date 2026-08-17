# -*- coding: utf-8 -*-
"""Batch-mode think generation test: 4 prompts x 4 = 16 sequences in ONE generate call,
with the TRL-style .any() StoppingCriteria. Tests the early-stop hypothesis."""
import collections, json, random, sys, time
import torch
sys.path.insert(0, "/data/GRPO/scripts")
from transformers import StoppingCriteria, StoppingCriteriaList
from grpo_model import load_model, encode_prompt

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816

class _ThinkStop(StoppingCriteria):
    def __init__(self, tid): self.tid = tid
    def __call__(self, input_ids, scores, **kw):
        return bool((input_ids[:, -1] == self.tid).any())

def main():
    model, tokenizer, template = load_model("cuda:0")
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]
    sc = StoppingCriteriaList([_ThinkStop(tid)])
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)
    prompts = [by_group[g]["think"]["prompt"] for g in gids[:4]]

    pids_list = [encode_prompt(tokenizer, p) for p in prompts]
    input_ids_list, attn_list = [], []
    for pids in pids_list:
        for _ in range(4):
            input_ids_list.append(pids)
    max_len = max(len(x) for x in input_ids_list)
    input_t = torch.zeros(len(input_ids_list), max_len, dtype=torch.long, device=model.device)
    attn = torch.zeros(len(input_ids_list), max_len, dtype=torch.long, device=model.device)
    for i, ids in enumerate(input_ids_list):
        input_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attn[i, :len(ids)] = 1
    t0 = time.time()
    out = model.generate(inputs=input_t, attention_mask=attn, max_new_tokens=2048,
                         do_sample=True, temperature=0.9, top_p=0.95,
                         num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                         stopping_criteria=sc)
    dt = time.time() - t0
    rows_out = []
    for i in range(out.shape[0]):
        plen = len(input_ids_list[i])
        gen = out[i, plen:].tolist()
        rows_out.append({"i": i, "prompt": i // 4, "n_tokens": len(gen),
                         "close": tid in gen, "pos": gen.index(tid) if tid in gen else None,
                         "last": gen[-1]})
    print(json.dumps({"total_sec": round(dt, 1), "rows": rows_out}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
