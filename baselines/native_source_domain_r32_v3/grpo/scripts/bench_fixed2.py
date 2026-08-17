# -*- coding: utf-8 -*-
"""Control run: verbatim copy of bench_fixed.py Think block + in-process control samples."""
import collections
import json
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_sid import final_sid, parse_sid, think_reward
from grpo_model import load_model, encode_prompt, generate_batch
from grpo_trl_trainer import make_think_reward_func, make_nothink_reward_func  # same import as bench

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816


def main():
    model, tokenizer, template = load_model("cuda:0")
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    import collections as c
    by_group = c.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = __import__("random").Random(SEED)
    rng.shuffle(gids)
    groups = [by_group[g] for g in gids[:4]]

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _ThinkStop(StoppingCriteria):
        def __init__(self, tid):
            self.tid = tid

        def __call__(self, input_ids, scores, **kw):
            return bool((input_ids[:, -1] == self.tid).any())

    think_tok = tokenizer.encode("</think>", add_special_tokens=False)[0]
    print("tid =", think_tok, flush=True)
    sc = StoppingCriteriaList([_ThinkStop(think_tok)])

    def gen_with_stop(prompt_ids, **kw):
        inputs = {"input_ids": torch.tensor([prompt_ids], device=model.device, dtype=torch.long),
                  "attention_mask": torch.ones(1, len(prompt_ids), device=model.device, dtype=torch.long)}
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=True,
                             temperature=0.9, top_p=0.95, num_return_sequences=1,
                             pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=sc)
        return out, tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=False)

    print("=== Think batch (16 samples, verbatim) ===", flush=True)
    t0 = time.time()
    cots = []
    details = []
    for g in groups:
        pids = encode_prompt(tokenizer, g["think"]["prompt"])
        for _ in range(4):
            out, txt = gen_with_stop(pids)
            cots.append(txt)
            gen = out[0][len(pids):]
            details.append({"new_tokens": int(gen.shape[0]), "close": "</think>" in txt,
                            "last": int(gen[-1])})
    gen_time = time.time() - t0
    lens = [len(c) for c in cots]
    closure = sum(1 for c in cots if "</think>" in c)
    print(f"gen(with stop): {gen_time:.1f}s, mean_len={sum(lens)/len(lens):.0f} closure={closure}/16", flush=True)
    for i, d in enumerate(details):
        print("  ", i, d, flush=True)

    print("=== Control: same-process, groups[0] x4 more samples ===", flush=True)
    pids0 = encode_prompt(tokenizer, groups[0]["think"]["prompt"])
    t0 = time.time()
    for k in range(4):
        out, txt = gen_with_stop(pids0)
        gen = out[0][len(pids0):]
        print(json.dumps({"ctrl": k, "new_tokens": int(gen.shape[0]),
                          "close": "</think>" in txt, "last": int(gen[-1]),
                          "head": txt[:50]}, ensure_ascii=False), flush=True)
    print(f"ctrl total: {time.time()-t0:.1f}s", flush=True)

    print("=== SUMMARY ===", flush=True)
    print(json.dumps(dict(think_gen_sec=round(gen_time, 1), closure=closure,
                          lens=[d["new_tokens"] for d in details],
                          last=[d["last"] for d in details]), indent=2), flush=True)


if __name__ == "__main__":
    main()
