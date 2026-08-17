# -*- coding: utf-8 -*-
"""Beam32 cbs re-benchmark v2 after TRUE left-padding fix in generate_batch.
1) Minimal deterministic test: short/long single-run vs batch [short,long] and
   [long,short] - candidate token ids must match EXACTLY per candidate.
2) Benchmark FA2 cbs=1 vs cbs=2 on typical + long contexts.
3) If cbs=2 is correct AND faster, probe cbs=4; else freeze cbs=1.
Beam semantics unchanged: 32 beams / 32 returns / 128 new tokens."""
import collections
import gc
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt, generate_batch, render_prompt
from grpo_sid import final_sid, think_reward, parse_sid
from transformers import StoppingCriteria, StoppingCriteriaList

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816


def make_cots(model, tokenizer, prompts, max_new=2048):
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return input_ids[:, -1] == tid

    out_pairs = []
    with torch.inference_mode():
        for p in prompts:
            pids = encode_prompt(tokenizer, p)
            inp = {"input_ids": torch.tensor([pids], device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(1, len(pids), device=model.device, dtype=torch.long)}
            gen = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                                 temperature=0.9, top_p=0.95, num_return_sequences=1,
                                 pad_token_id=tokenizer.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([_Stop()]))[0]
            cids = gen[len(pids):].tolist()
            try:
                pos = cids.index(tid)
                cids = cids[:pos + 1]
            except ValueError:
                pass
            out_pairs.append((pids, cids))
    return out_pairs


def beam_ids(model, tokenizer, contexts, cbs):
    """One generate call over cbs contexts; returns (ids_list, wall, peak)."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _, ids = generate_batch(model, tokenizer, contexts,
                            max_new_tokens=128, num_beams=32,
                            num_return_sequences=32, return_ids=True)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    return ids, wall, peak


def score_ids(ids_list, gold_set, tokenizer):
    texts = [tokenizer.decode(x, skip_special_tokens=False) for x in ids_list]
    sids = [final_sid(t) for t in texts]
    inv = sum(1 for s in sids if s is None)
    r, ec, ac, a2 = think_reward(sids, gold_set)
    return inv, ec, ac, a2, r


def main():
    import random
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)

    model, tokenizer, template = load_model("cuda:0")
    model.eval()

    typ_prompts = [by_group[g]["think"]["prompt"] for g in gids[:4]]
    trows = [r for r in rows if r["route"] == "think"]
    trows.sort(key=lambda r: len(tokenizer.encode(render_prompt(tokenizer, r["prompt"]),
                                                  add_special_tokens=False)), reverse=True)
    long_prompts = [r["prompt"] for r in trows[:4]]

    results = {"deterministic": {}, "bench": {}}
    for tag, prompts in (("typical", typ_prompts), ("long", long_prompts)):
        pairs = make_cots(model, tokenizer, prompts)
        contexts = [p + c for p, c in pairs]
        gold_key = [gids[:4]] if tag == "typical" else [[trows[i]["recommendation_group_id"] for i in range(4)]]
        golds = []
        for g in gold_key[0]:
            gs = set()
            for s in by_group[g]["think"]["all_gold_sids"]:
                t = parse_sid(s)
                if t:
                    gs.add(t)
            golds.append(gs)
        clens = [len(c) for c in contexts]
        print(f"[{tag}] ctx lens: {clens}", flush=True)

        # ---- minimal deterministic test: short=ctx0, long=ctx3 ----
        if tag == "typical":
            si, li = 0, 3
            if clens[si] > clens[li]:
                si, li = li, si
            short, long_ = contexts[si], contexts[li]
            with torch.inference_mode():
                a_ids, _, _ = beam_ids(model, tokenizer, [short], 1)
                b_ids, _, _ = beam_ids(model, tokenizer, [long_], 1)
                ab_ids, _, _ = beam_ids(model, tokenizer, [short, long_], 2)
                ba_ids, _, _ = beam_ids(model, tokenizer, [long_, short], 2)
            dt = {
                "short_len": len(short), "long_len": len(long_),
                "A_32": len(a_ids) == 32, "B_32": len(b_ids) == 32,
                "AB_exact_short": ab_ids[:32] == a_ids,
                "AB_exact_long": ab_ids[32:] == b_ids,
                "BA_exact_long": ba_ids[:32] == b_ids,
                "BA_exact_short": ba_ids[32:] == a_ids,
            }
            results["deterministic"] = dt
            print("deterministic:", json.dumps(dt, ensure_ascii=False), flush=True)
            if not (dt["AB_exact_short"] and dt["AB_exact_long"] and
                    dt["BA_exact_long"] and dt["BA_exact_short"]):
                print("DETERMINISTIC FAIL - batch != single", flush=True)

        # ---- benchmark cbs=1 vs cbs=2 ----
        bench = {}
        for cbs in (1, 2):
            try:
                batches = [contexts[i:i + cbs] for i in range(0, len(contexts), cbs)]
                walls, peaks, all_ids = [], [], []
                with torch.inference_mode():
                    for bctx in batches:
                        ids, wall, peak = beam_ids(model, tokenizer, bctx, cbs)
                        walls.append(wall)
                        peaks.append(peak)
                        all_ids.append(ids)
                wall = sum(walls)
                peak = max(peaks)
                flat = [x for b in all_ids for x in b]
                # regroup by context (ownership): flat order = (batch, ret_idx)
                per_ctx = []
                idx = 0
                for b in all_ids:
                    for c in range(len(b) // 32):
                        per_ctx.append(b[c * 32:(c + 1) * 32])
                # correctness vs cbs=1 baseline
                base = []
                with torch.inference_mode():
                    for ctx in contexts:
                        ids, _, _ = beam_ids(model, tokenizer, [ctx], 1)
                        base.append(ids)
                cand_parity = per_ctx == base  # exact per-candidate ids
                rp = True
                for ci in range(len(contexts)):
                    if score_ids(per_ctx[ci], golds[ci], tokenizer) != score_ids(base[ci], golds[ci], tokenizer):
                        rp = False
                inv = sum(score_ids(per_ctx[ci], golds[ci], tokenizer)[0] for ci in range(len(contexts)))
                bench[cbs] = dict(
                    wall=round(wall, 1), sec_cot=round(wall / len(contexts), 1),
                    peak_allocated_mb=peak, candidate_parity=bool(cand_parity),
                    reward_parity=bool(rp), invalid=inv, oom=False)
                print(f"[{tag}] cbs={cbs}: wall={bench[cbs]['wall']}s "
                      f"sec/cot={bench[cbs]['sec_cot']}s peak={peak}MB "
                      f"cand_parity={bench[cbs]['candidate_parity']} "
                      f"reward_parity={bench[cbs]['reward_parity']} invalid={inv}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                bench[cbs] = {"oom": True}
                print(f"[{tag}] cbs={cbs}: OOM", flush=True)
        results["bench"][tag] = bench

    # ---- cbs=4 probe only if cbs=2 correct and faster ----
    probe4 = {}
    for tag in ("typical", "long"):
        b2 = results["bench"][tag].get("2", {})
        b1 = results["bench"][tag].get("1", {})
        if (not b2.get("oom") and b2.get("candidate_parity") and b2.get("reward_parity")
                and b2.get("sec_cot", 1e9) < b1.get("sec_cot", 0) * 0.9):
            print(f"[{tag}] cbs=2 faster+correct -> probe cbs=4", flush=True)
            contexts = [p + c for p, c in make_cots(model, tokenizer,
                                                    typ_prompts if tag == "typical" else long_prompts)]
            try:
                with torch.inference_mode():
                    ids, wall, peak = beam_ids(model, tokenizer, contexts, 4)
                probe4[tag] = dict(wall=round(wall, 1), sec_cot=round(wall / 4, 1),
                                   peak_allocated_mb=peak, oom=False)
                print(f"[{tag}] cbs=4: wall={probe4[tag]['wall']}s sec/cot={probe4[tag]['sec_cot']}s peak={peak}MB", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                probe4[tag] = {"oom": True}
                print(f"[{tag}] cbs=4: OOM", flush=True)
        else:
            probe4[tag] = {"skipped": True}
    results["probe4"] = probe4

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=1), flush=True)
    with open("/data/GRPO/logs/beam_cbs_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
