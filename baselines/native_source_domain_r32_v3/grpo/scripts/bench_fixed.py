# -*- coding: utf-8 -*-
"""Fixed-batch benchmark after stop/invalid fixes: 1 Think batch + 1 NoThink batch.
Measures generation time, </think> true stop rate, mean tokens, beam invalid rate."""
import collections
import json
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_sid import final_sid, parse_sid, think_reward
from grpo_model import load_model, encode_prompt, generate_batch
from grpo_trl_trainer import make_think_reward_func, make_nothink_reward_func

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

    # ---------------- Think batch (4 prompts x 4) with token-id stop ----------------
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _ThinkStop(StoppingCriteria):
        def __init__(self, tid):
            self.tid = tid

        def __call__(self, input_ids, scores, **kw):
            return bool((input_ids[:, -1] == self.tid).any())

    think_tok = tokenizer.encode("</think>", add_special_tokens=False)[0]
    sc = StoppingCriteriaList([_ThinkStop(think_tok)])

    def gen_with_stop(prompt_ids, **kw):
        inputs = {"input_ids": torch.tensor([prompt_ids], device=model.device, dtype=torch.long),
                  "attention_mask": torch.ones(1, len(prompt_ids), device=model.device, dtype=torch.long)}
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=True,
                             temperature=0.9, top_p=0.95, num_return_sequences=1,
                             pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=sc)
        return tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=False)

    print("=== Think batch ===")
    t0 = time.time()
    cots = []
    for g in groups:
        pids = encode_prompt(tokenizer, g["think"]["prompt"])
        for _ in range(4):
            cots.append((g["think"]["prompt"], gen_with_stop(pids)))
    gen_time = time.time() - t0
    lens = [len(c) for c, _ in cots]
    closure = sum(1 for c, _ in cots if "</think>" in c)
    # measure without stop (1 sample) for comparison
    pids0 = encode_prompt(tokenizer, groups[0]["think"]["prompt"])
    t1 = time.time()
    _ = generate_batch(model, tokenizer, [pids0], max_new_tokens=2048, do_sample=True,
                       temperature=0.9, top_p=0.95, num_return_sequences=1)
    nostop_time = time.time() - t1
    print(f"gen(with stop): {gen_time:.1f}s, mean_len={sum(lens)/len(lens):.0f} "
          f"closure={closure}/16, no-stop single sample: {nostop_time:.1f}s")

    t0 = time.time()
    gold_sets = []
    beam_invalid = 0
    n_beam = 0
    for g in groups:
        gs = set()
        for s in g["think"]["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gs.add(t)
        gold_sets.append(gs)
    # beam over all 16 cots (serial, as smoke)
    total_beam = 0.0
    for i, (p, cot) in enumerate(cots):
        idx = cot.find("</think>")
        cot_trim = cot[: idx + len("</think>")] if idx >= 0 else cot
        pids = encode_prompt(tokenizer, p)
        cids = tokenizer.encode(cot_trim, add_special_tokens=False)
        t1 = time.time()
        texts = generate_batch(model, tokenizer, [pids + cids], max_new_tokens=128,
                               num_beams=32, num_return_sequences=32)
        total_beam += time.time() - t1
        sids = [final_sid(t) for t in texts]
        beam_invalid += sum(1 for s in sids if s is None)
        n_beam += len(sids)
    print(f"beam32: {total_beam:.1f}s (16 cots, {total_beam/16:.1f}s/cot), "
          f"invalid={beam_invalid}/{n_beam} ({beam_invalid/n_beam*100:.1f}%)")

    # ---------------- NoThink batch (4 prompts x 8) ----------------
    print("=== NoThink batch ===")
    t0 = time.time()
    nt_lens, sid_pos, extra = [], [], []
    for g in groups:
        pids = encode_prompt(tokenizer, g["no_think"]["prompt"])
        texts = generate_batch(model, tokenizer, [pids], max_new_tokens=128,
                               do_sample=True, temperature=1.0, top_p=1.0,
                               num_return_sequences=8)
        for t in texts:
            nt_lens.append(len(t))
            m = __import__("re").search(
                r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>", t)
            if m:
                sid_pos.append(m.start())
                extra.append(len(t) - m.start() - len(m.group(0)))
    nt_time = time.time() - t0
    print(f"nothink gen: {nt_time:.1f}s (32 samples), mean_len={sum(nt_lens)/len(nt_lens):.0f} "
          f"first_sid_mean={sum(sid_pos)/len(sid_pos):.0f} extra_after_sid_mean={sum(extra)/len(extra):.0f}")

    print("\n=== SUMMARY ===")
    print(json.dumps(dict(
        think_gen_sec=round(gen_time, 1), think_gen_mean_len=int(sum(lens)/len(lens)),
        think_closure_rate=closure/16, think_no_stop_single_sec=round(nostop_time, 1),
        beam_sec=round(total_beam, 1), beam_sec_per_cot=round(total_beam/16, 1),
        beam_invalid_rate=round(beam_invalid/n_beam, 4),
        nothink_gen_sec=round(nt_time, 1), nothink_mean_len=int(sum(nt_lens)/len(nt_lens)),
        nothink_first_sid_mean=int(sum(sid_pos)/len(sid_pos)),
        nothink_extra_after_sid_mean=int(sum(extra)/len(extra)),
    ), indent=2))


if __name__ == "__main__":
    main()
