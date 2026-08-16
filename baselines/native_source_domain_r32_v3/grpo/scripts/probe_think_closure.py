# -*- coding: utf-8 -*-
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
    tok = tokenizer.encode("</think>", add_special_tokens=False)
    print("encode('</think>') =", tok, flush=True)
    tid = tok[0]
    sc = StoppingCriteriaList([_ThinkStop(tid)])

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    smoke_think = [rows[i]["prompt"] for i in (1, 3, 5, 7)]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)
    bench_think = [by_group[g]["think"]["prompt"] for g in gids[:4]]

    results = []
    for label, prompts in (("smoke_head", smoke_think), ("bench4", bench_think)):
        for i, p in enumerate(prompts):
            pids = encode_prompt(tokenizer, p)
            inp = {"input_ids": torch.tensor([pids], device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(1, len(pids), device=model.device, dtype=torch.long)}
            t0 = time.time()
            out = model.generate(**inp, max_new_tokens=3072, do_sample=True,
                                 temperature=0.9, top_p=0.95, num_return_sequences=1,
                                 pad_token_id=tokenizer.eos_token_id, stopping_criteria=sc)
            dt = time.time() - t0
            gen = out[0][len(pids):]
            text = tokenizer.decode(gen, skip_special_tokens=False)
            pos = (gen == tid).nonzero().flatten()
            r = {
                "label": label, "idx": i,
                "prompt_head": p[:70],
                "new_tokens": int(gen.shape[0]),
                "first_151668": int(pos[0].item()) if pos.numel() else None,
                "has_think_marker": "<|im_start|>think" in text,
                "has_close_str": "</think>" in text,
                "last_is_151668": bool(gen[-1] == tid),
                "sec": round(dt, 1),
                "head": text[:120],
                "tail": text[-120:],
            }
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    with open("/data/GRPO/logs/probe_think_closure.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
