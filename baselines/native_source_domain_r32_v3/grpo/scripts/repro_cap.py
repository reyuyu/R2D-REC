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
    print("PEFT adapters:", list(model.peft_config.keys()) if hasattr(model, "peft_config") else "n/a", flush=True)
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]
    sc = StoppingCriteriaList([_ThinkStop(tid)])
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)
    prompts = [by_group[g]["think"]["prompt"] for g in gids[:2]]
    for cap in (2048, 3072):
        for i, p in enumerate(prompts):
            pids = encode_prompt(tokenizer, p)
            inp = {"input_ids": torch.tensor([pids], device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(1, len(pids), device=model.device, dtype=torch.long)}
            t0 = time.time()
            out = model.generate(**inp, max_new_tokens=cap, do_sample=True,
                                 temperature=0.9, top_p=0.95, num_return_sequences=1,
                                 pad_token_id=tokenizer.eos_token_id, stopping_criteria=sc)
            dt = time.time() - t0
            gen = out[0][len(pids):]
            text = tokenizer.decode(gen, skip_special_tokens=False)
            print(json.dumps({"cap": cap, "idx": i, "new_tokens": int(gen.shape[0]),
                              "close": "</think>" in text, "last": int(gen[-1]),
                              "sec": round(dt,1), "head": text[:60]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
