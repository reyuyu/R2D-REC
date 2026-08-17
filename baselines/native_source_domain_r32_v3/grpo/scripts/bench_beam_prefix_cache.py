# -*- coding: utf-8 -*-
"""Beam32 shared-prefix KV prefill reuse experiment (NO training).
Baseline: production generate_batch cbs=1 (full context, beam32 32/32/128).
Prefix-cache: prefill prefix once (batch=1), batch_repeat_interleave(32),
then native model.generate(full_context, past_key_values=cache32, beam32).
Forward hook proves prefill batch=1 seq=len-1 and beam first forward batch=32 seq~1.
Stage1: 4 typical + 4 long closed CoTs; Stage2: 32 CoTs (if stage1 reward parity 100%)."""
import collections
import gc
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt, generate_batch
from grpo_sid import think_reward, parse_sid
from transformers import StoppingCriteria, StoppingCriteriaList

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816


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
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return input_ids[:, -1] == tid

    def gen_cot(p):
        pids = encode_prompt(tokenizer, p)
        with torch.inference_mode():
            inp = {"input_ids": torch.tensor([pids], device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(1, len(pids), device=model.device, dtype=torch.long)}
            gen = model.generate(**inp, max_new_tokens=2048, do_sample=True,
                                 temperature=0.9, top_p=0.95, num_return_sequences=1,
                                 pad_token_id=pad_id,
                                 stopping_criteria=StoppingCriteriaList([_Stop()]))[0]
            cids = gen[len(pids):].tolist()
            try:
                pos = cids.index(tid)
                cids = cids[:pos + 1]
            except ValueError:
                pass
        return pids + cids

    # ---- forward instrumentation on the HF base model ----
    fwd_log = []

    def _hook(module, args, kwargs):
        inp = args[0] if args else kwargs.get("input_ids")
        cp = kwargs.get("cache_position")
        fwd_log.append((tuple(inp.shape), None if cp is None else cp.tolist()[:4]))

    handle = model.base_model.model.register_forward_pre_hook(_hook, with_kwargs=True)

    # ---- contexts ----
    typ_prompts = [by_group[g]["think"]["prompt"] for g in gids[:4]]
    trows = [r for r in rows if r["route"] == "think"]
    trows.sort(key=lambda r: len(encode_prompt(tokenizer, r["prompt"])), reverse=True)
    long_prompts = [r["prompt"] for r in trows[:4]]
    contexts = [gen_cot(p) for p in typ_prompts] + [gen_cot(p) for p in long_prompts]
    tags = ["typical"] * 4 + ["long"] * 4
    golds_all = []
    for g in list(gids[:4]) + [trows[i]["recommendation_group_id"] for i in range(4)]:
        gs = set()
        for s in by_group[g]["think"]["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gs.add(t)
        golds_all.append(gs)
    print("ctx lens:", [len(c) for c in contexts], flush=True)

    def baseline_beam(ctx):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _, ids = generate_batch(model, tokenizer, [ctx], max_new_tokens=128,
                                num_beams=32, num_return_sequences=32, return_ids=True)
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        return ids, wall, peak

    def cache_beam(ctx):
        """prefill prefix once -> repeat cache x32 -> native beam32 generate."""
        prefix = ctx[:-1]
        dev = model.device
        fwd_log.clear()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            pre = model(
                input_ids=torch.tensor([prefix], device=dev, dtype=torch.long),
                attention_mask=torch.ones(1, len(prefix), dtype=torch.long, device=dev),
                use_cache=True, return_dict=True)
        prefill_sec = time.time() - t0
        cache = pre.past_key_values
        assert cache.get_seq_length() == len(prefix), (cache.get_seq_length(), len(prefix))
        t0 = time.time()
        cache.batch_repeat_interleave(32)  # in-place
        rep_sec = time.time() - t0
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=torch.tensor([ctx], device=dev, dtype=torch.long),
                attention_mask=torch.ones(1, len(ctx), dtype=torch.long, device=dev),
                past_key_values=cache,
                max_new_tokens=128,
                num_beams=32,
                num_return_sequences=32,
                pad_token_id=pad_id,
            )
        beam_sec = time.time() - t0
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        ids = [out[i, len(ctx):].tolist() for i in range(32)]
        shapes = list(fwd_log)
        return ids, prefill_sec, rep_sec, beam_sec, peak, shapes

    def score(ids):
        texts = [tokenizer.decode(x, skip_special_tokens=False) for x in ids]
        sids = [__import__("grpo_sid", fromlist=["final_sid"]).final_sid(t) for t in texts]
        inv = sum(1 for s in sids if s is None)
        r, ec, ac, a2 = think_reward(sids, None) if False else think_reward(
            [s for s in sids], golds_all[0])  # placeholder, replaced below
        return inv, ec, ac, a2, r, sids

    # ---- stage 1: 8 CoTs ----
    results = {"stage1": [], "fwd_proof": None}
    stage1_ok = True
    for i, ctx in enumerate(contexts):
        gold = golds_all[i]
        b_ids, b_wall, b_peak = baseline_beam(ctx)
        c_ids, p_sec, r_sec, c_beam_sec, c_peak, shapes = cache_beam(ctx)
        b_texts = [tokenizer.decode(x, skip_special_tokens=False) for x in b_ids]
        c_texts = [tokenizer.decode(x, skip_special_tokens=False) for x in c_ids]
        b_sids = [__import__("grpo_sid", fromlist=["final_sid"]).final_sid(t) for t in b_texts]
        c_sids = [__import__("grpo_sid", fromlist=["final_sid"]).final_sid(t) for t in c_texts]
        b_r, b_ec, b_ac, b_a2 = think_reward(b_sids, gold)
        c_r, c_ec, c_ac, c_a2 = think_reward(c_sids, gold)
        b_inv = sum(1 for s in b_sids if s is None)
        c_inv = sum(1 for s in c_sids if s is None)
        cand_parity = b_ids == c_ids
        sid_set_parity = set(b_sids) == set(c_sids)
        reward_parity = (b_r == c_r) and (b_ec, b_ac, b_a2) == (c_ec, c_ac, c_a2) and (b_inv == c_inv)
        if not reward_parity:
            stage1_ok = False
        results["stage1"].append(dict(
            idx=i, tag=tags[i], ctx_len=len(ctx),
            baseline_wall=round(b_wall, 1), baseline_peak_mb=b_peak,
            prefill_sec=round(p_sec, 1), repeat_sec=round(r_sec, 2),
            beam_sec=round(c_beam_sec, 1), cache_total=round(p_sec + c_beam_sec, 1),
            cache_peak_mb=c_peak,
            reward_base=b_r, reward_cache=c_r,
            exact=(b_ec, c_ec), ab=(b_ac, c_ac), a=(b_a2, c_a2),
            invalid=(b_inv, c_inv),
            cand_parity=bool(cand_parity), sid_set_parity=bool(sid_set_parity),
            reward_parity=bool(reward_parity)))
        print(f"[{tags[i]}] ctx{len(ctx)} base={b_wall:.1f}s/{b_peak}MB "
              f"cache={p_sec:.1f}+{c_beam_sec:.1f}={p_sec + c_beam_sec:.1f}s/{c_peak}MB "
              f"reward {b_r:.3f}->{c_r:.3f} exact {b_ec}->{c_ec} invalid {b_inv}->{c_inv} "
              f"cand_parity={cand_parity} sid_set={sid_set_parity} rw_parity={reward_parity}",
              flush=True)
        if shapes:
            results["fwd_proof"] = dict(
                prefill=[s for s in shapes if s[0][0] == 1 and s[0][1] == len(ctx) - 1][:1],
                beam_first=[s for s in shapes if s[0][0] == 32][:3],
                all_shapes=shapes[:8])
    print("fwd_proof:", json.dumps(results["fwd_proof"], ensure_ascii=False), flush=True)

    # ---- stage 2: 32 CoTs (only if stage1 reward parity 100%) ----
    if stage1_ok:
        more_prompts = ([by_group[g]["think"]["prompt"] for g in gids[4:20]]
                        + [r["prompt"] for r in trows[4:20]])
        more_gids = list(gids[4:20]) + [trows[i]["recommendation_group_id"] for i in range(4, 20)]
        ctxs2 = [gen_cot(p) for p in more_prompts]
        golds2 = []
        for g in more_gids:
            gs = set()
            for s in by_group[g]["think"]["all_gold_sids"]:
                t = parse_sid(s)
                if t:
                    gs.add(t)
            golds2.append(gs)
        n_bad = 0
        for ctx, gold in zip(ctxs2, golds2):
            b_ids, _, _ = baseline_beam(ctx)
            c_ids, _, _, _, _, _ = cache_beam(ctx)
            b_texts = [tokenizer.decode(x, skip_special_tokens=False) for x in b_ids]
            c_texts = [tokenizer.decode(x, skip_special_tokens=False) for x in c_ids]
            fs = __import__("grpo_sid", fromlist=["final_sid"]).final_sid
            b_sids = [fs(t) for t in b_texts]
            c_sids = [fs(t) for t in c_texts]
            b_r, b_ec, b_ac, b_a2 = think_reward(b_sids, gold)
            c_r, c_ec, c_ac, c_a2 = think_reward(c_sids, gold)
            b_inv = sum(1 for s in b_sids if s is None)
            c_inv = sum(1 for s in c_sids if s is None)
            ok = (b_r == c_r) and (b_ec, b_ac, b_a2) == (c_ec, c_ac, c_a2) and b_inv == c_inv
            if not ok:
                n_bad += 1
                print(f"[stage2] MISMATCH ctx_len={len(ctx)} reward {b_r}->{c_r} "
                      f"invalid {b_inv}->{c_inv}", flush=True)
        results["stage2"] = dict(n=len(ctxs2), mismatches=n_bad,
                                 reward_parity_100=(n_bad == 0))
        print(f"stage2: {len(ctxs2)} CoTs, mismatches={n_bad}", flush=True)

    # ---- performance summary (stage1 typical/long) ----
    perf = {}
    for tag in ("typical", "long"):
        rows_p = [r for r in results["stage1"] if r["tag"] == tag]
        base_w = statistics.mean(r["baseline_wall"] for r in rows_p)
        cache_w = statistics.mean(r["prefill_sec"] + r["beam_sec"] for r in rows_p)
        perf[tag] = dict(
            baseline_sec_cot=round(base_w, 1),
            cache_total_sec_cot=round(cache_w, 1),
            speedup=round(base_w / cache_w, 3),
            baseline_peak_mb=max(r["baseline_peak_mb"] for r in rows_p),
            cache_peak_mb=max(r["cache_peak_mb"] for r in rows_p))
    results["perf"] = perf
    handle.remove()
    print("perf:", json.dumps(perf, ensure_ascii=False, indent=1), flush=True)
    with open("/data/GRPO/logs/beam_prefix_cache.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
