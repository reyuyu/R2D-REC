# -*- coding: utf-8 -*-
"""Diagnose: (a) </think> token decode & stop behavior, (b) Beam invalid classification,
(c) NoThink completion length vs first-SID position. GPU 0, single process."""
import collections
import json
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_sid import final_sid, parse_sid
from grpo_model import load_model, encode_prompt, generate_batch

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"


def main():
    model, tokenizer, template = load_model("cuda:0")

    # (a) </think> token identity + decode
    think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    print("</think> token ids:", think_ids, "| decode:", repr(tokenizer.decode(think_ids)))
    print("decode(151668):", repr(tokenizer.decode([151668])))

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    think_rec = [r for r in rows if r["route"] == "think"][:1]
    prompt = think_rec[0]["prompt"]

    # ---- Think: sample 4 CoTs, check stop behavior ----
    ids = encode_prompt(tokenizer, prompt)
    t0 = time.time()
    texts = generate_batch(model, tokenizer, [ids], max_new_tokens=2048,
                           do_sample=True, temperature=0.9, top_p=0.95,
                           num_return_sequences=4)
    gen_time = time.time() - t0
    print(f"\nThink gen: 4 samples {gen_time:.1f}s")
    for i, t in enumerate(texts):
        n = len(t)
        has_close = "</think>" in t
        idx = t.find("</think>")
        pos = (idx, idx + len("</think>")) if idx >= 0 else (-1, -1)
        tail = repr(t[max(0, idx - 30): idx + 40]) if idx >= 0 else repr(t[-60:])
        print(f"  sample{i}: len={n} has_close={has_close} close_at={pos} tail={tail[:110]}")

    # ---- Think Beam: invalid classification ----
    print("\n=== Beam invalid classification (2 CoTs) ===")
    golds = set()
    for s in think_rec[0]["all_gold_sids"]:
        t = parse_sid(s)
        if t:
            golds.add(t)
    for ci in range(2):
        cot = texts[ci]
        idx = cot.find("</think>")
        cot_trim = cot[: idx + len("</think>")] if idx >= 0 else cot
        pids = encode_prompt(tokenizer, prompt)
        cids = tokenizer.encode(cot_trim, add_special_tokens=False)
        btexts = generate_batch(model, tokenizer, [pids + cids], max_new_tokens=128,
                                num_beams=32, num_return_sequences=32)
        cats = collections.Counter()
        examples = []
        for bt in btexts:
            sid = final_sid(bt)
            if sid is None:
                has_sid = bool(parse_sid(bt))
                full = bool(parse_sid(bt))
                # classify: contains any complete sid? ends mid-sid? truncated?
                import re
                sid_re = re.compile(r"<\|(?:video|prod|ad|living)_begin\|>")
                doms = sid_re.findall(bt)
                partial_a = "<s_a_" in bt or "s_a_" in bt
                if not doms and not partial_a:
                    cats["no_sid_no_domain"] += 1
                elif doms and not full:
                    cats["domain_no_full_sid"] += 1
                elif not doms and partial_a:
                    cats["no_domain_but_s_a"] += 1
                else:
                    cats["other"] += 1
                if len(examples) < 2:
                    examples.append(bt[-80:])
            else:
                cats["valid"] += 1
        print(f"  CoT{ci} (len {len(cot_trim)}): {dict(cats)}")
        for e in examples:
            print("    invalid tail:", repr(e))

    # ---- NoThink: completion length vs first-SID position ----
    print("\n=== NoThink completion analysis ===")
    nrec = [r for r in rows if r["route"] == "no_think"][:1]
    nids = encode_prompt(tokenizer, nrec[0]["prompt"])
    t0 = time.time()
    ntexts = generate_batch(model, tokenizer, [nids], max_new_tokens=128,
                            do_sample=True, temperature=1.0, top_p=1.0,
                            num_return_sequences=8)
    print(f"NoThink gen: 8 samples {time.time()-t0:.1f}s")
    for i, t in enumerate(ntexts):
        sid = final_sid(t)
        m = __import__("re").search(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>", t)
        first_pos = m.start() if m else -1
        print(f"  sample{i}: len={len(t)} first_sid_at={first_pos} "
              f"extra_after_sid={len(t)-first_pos-len(m.group(0)) if m else 'NA'} "
              f"valid={sid is not None}")


if __name__ == "__main__":
    main()
