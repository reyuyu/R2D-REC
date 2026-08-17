# -*- coding: utf-8 -*-
"""Beam32 context micro-batch benchmark on FINAL smoke real closed Think CoT.
cbs = context_batch_size (number of prompt+CoT contexts per generate call).
Order: cbs=1 -> 2 -> 4 (skip larger if OOM). Beam semantics unchanged:
num_beams=32, num_return_sequences=32, max_new_tokens=128, hierarchical reward.
Correctness vs cbs=1: per-CoT 32 candidates identical (beam is deterministic),
candidate ownership unmixed, exact/AB/A counts, reward, invalid rate all equal."""
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
    """Generate ONE closed CoT per prompt (per-sample </think> stop), trim at
    </think>; returns (prompt_ids, cot_ids) pairs."""
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
                pass  # keep full (no closure, mirrors reward path)
            out_pairs.append((pids, cids))
    return out_pairs


def beam_one(model, tokenizer, contexts, cbs):
    """Run beam32 over cbs contexts in one generate call. Returns (texts, wall, peak)."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    texts = generate_batch(model, tokenizer, contexts,
                           max_new_tokens=128, num_beams=32,
                           num_return_sequences=32)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    return texts, wall, peak


def score(texts, gold_set):
    """(invalid, exact, ab, a, reward) over 32 candidates."""
    sids = [final_sid(t) for t in texts]
    inv = sum(1 for s in sids if s is None)
    r, ec, ac, a2 = think_reward(sids, gold_set)
    return inv, ec, ac, a2, r


def main():
    import collections
    import json as _json
    rows = [_json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = __import__("random").Random(SEED)
    rng.shuffle(gids)

    device = "cuda:0"
    model, tokenizer, template = load_model(device)
    model.eval()

    # ---- context set 1: current typical (final smoke Think chunk0 = gids[:4]) ----
    typ_prompts = [by_group[g]["think"]["prompt"] for g in gids[:4]]
    # ---- context set 2: longer real contexts (longest think prompts) ----
    import operator
    trows = [r for r in rows if r["route"] == "think"]
    trows.sort(key=lambda r: len(tokenizer.encode(render_prompt(tokenizer, r["prompt"]),
                                                  add_special_tokens=False)), reverse=True)
    long_prompts = [r["prompt"] for r in trows[:4]]

    results = {}
    for tag, prompts in (("typical", typ_prompts), ("long", long_prompts)):
        pairs = make_cots(model, tokenizer, prompts)
        contexts = [p + c for p, c in pairs]
        golds = []
        for g in gids[:4] if tag == "typical" else [trows[i]["recommendation_group_id"] for i in range(4)]:
            gs = set()
            for s in by_group[g]["think"]["all_gold_sids"]:
                t = parse_sid(s)
                if t:
                    gs.add(t)
            golds.append(gs)
        plens = [len(p) for p, _ in pairs]
        clens = [len(c) for _, c in pairs]
        print(f"[{tag}] ctx lens: prompt {plens} + cot {clens} = {[a + b for a, b in zip(plens, clens)]}",
              flush=True)
        # cbs=1 baseline
        base = []
        for i, ctx in enumerate(contexts):
            texts, wall, peak = beam_one(model, tokenizer, [ctx], 1)
            base.append((texts, wall, peak))
        # correctness refs from cbs=1
        ref_scores = [score(texts, golds[i]) for i, (texts, _, _) in enumerate(base)]
        ref_sec_cot = statistics.mean(w for _, w, _ in base)

        results[tag] = {}
        for cbs in (1, 2, 4):
            if cbs > 1 and results[tag].get(2 if cbs == 4 else 1, {}).get("oom"):
                print(f"[{tag}] skip cbs={cbs} (previous OOM)", flush=True)
                results[tag][cbs] = {"oom": True}
                continue
            try:
                batches = [contexts[i:i + cbs] for i in range(0, len(contexts), cbs)]
                walls, peaks, all_texts = [], [], []
                for bctx in batches:
                    texts, wall, peak = beam_one(model, tokenizer, bctx, cbs)
                    walls.append(wall)
                    peaks.append(peak)
                    all_texts.append(texts)
                wall = sum(walls)
                peak = max(peaks)
                # correctness vs cbs=1: regroup by input_idx
                ok_32 = True
                ok_own = True
                ok_score = True
                flat = []
                for batch_texts in all_texts:
                    for i, t in enumerate(batch_texts):
                        flat.append((len(flat) // 32, i % 32, t))
                for ctx_i in range(len(contexts)):
                    got = [t for ci, si, t in flat if ci == ctx_i]
                    if len(got) != 32:
                        ok_32 = False
                    ref_t = base[ctx_i][0]
                    if got != ref_t:
                        ok_own = False  # ownership or determinism mismatch
                        # fall back to score comparison
                    gs = score(got, golds[ctx_i])
                    rs = ref_scores[ctx_i]
                    if gs != rs:
                        ok_score = False
                # candidate-identity check: compare set equality per context
                ident_ok = all(sorted([t for ci, si, t in flat if ci == ctx_i]) ==
                               sorted(base[ctx_i][0]) for ctx_i in range(len(contexts)))
                sec_cot = wall / len(contexts)
                speedup = ref_sec_cot / sec_cot if sec_cot > 0 else 0.0
                res = dict(wall=round(wall, 1), sec_cot=round(sec_cot, 1),
                           peak_allocated_mb=peak,
                           speedup=round(speedup, 2),
                           per_cot_32=ok_32, ownership=ident_ok,
                           score_parity=ok_score,
                           ref_scores=[list(s) for s in ref_scores],
                           got_scores=[list(score([t for ci, si, t in flat if ci == ctx_i], golds[ctx_i]))
                                       for ctx_i in range(len(contexts))])
                results[tag][cbs] = res
                print(f"[{tag}] cbs={cbs}: wall={res['wall']}s sec/cot={res['sec_cot']}s "
                      f"peak={res['peak_allocated_mb']}MB speedup={res['speedup']} "
                      f"32/cot={ok_32} ownership={ident_ok} score_parity={ok_score}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[{tag}] cbs={cbs}: OOM", flush=True)
                results[tag][cbs] = {"oom": True}

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=1), flush=True)
    with open("/data/GRPO/logs/beam_cbs_final.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
