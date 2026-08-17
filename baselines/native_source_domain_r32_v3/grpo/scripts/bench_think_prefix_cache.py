# -*- coding: utf-8 -*-
"""Think G=4 Shared-Prompt KV Reuse experiment (NO training).
Stage1: 32 prompts next-token distribution equivalence (raw logits diff,
KL/JS/top-k/top-p support after temp=0.9 top_p=0.95 processing; row internal
consistency). Stage2: 64 real groups x G=4 sampling baseline vs prefix-cache
(256+256 CoTs; same prompts, same seed schedule; closure/length/diversity/
entropy stats; CUDA-synchronized timing; short/medium/long buckets).
No Beam reward this round."""
import collections
import gc
import json
import math
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt
from grpo_sid import final_sid
from transformers import StoppingCriteria, StoppingCriteriaList

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816
CKPT = "/data/GRPO/logs/think_prefix_cache_ckpt_%s.json" % os.environ.get("SHARD_ID", "0")


def sync_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(t0):
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def processed_dist(logits, temperature=0.9, top_p=0.95):
    """temperature + top_p processed sampling distribution over vocab."""
    l = logits.float() / temperature
    probs = torch.softmax(l, dim=-1)
    sp, si = probs.sort(descending=True)
    cum = sp.cumsum(-1)
    keep = cum - sp <= 1 - top_p
    sp = sp * keep
    sp = sp / sp.sum(-1, keepdim=True).clamp(min=1e-12)
    dist = torch.zeros_like(probs)
    dist.scatter_(-1, si, sp)
    return dist


def kl(p, q):
    m = (p > 0) & (q > 0)
    return float((p[m] * (p[m].log() - q[m].log())).sum())


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))


def topk_overlap(p, q, k):
    sp = set(p.topk(k).indices.tolist())
    sq = set(q.topk(k).indices.tolist())
    return len(sp & sq) / k


def main():
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

    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--shard", type=int, default=0)
    _ap.add_argument("--nshards", type=int, default=1)
    _ap.add_argument("--skip-stage1", action="store_true")
    _a = _ap.parse_args()

    # ================= stage 1: 32 prompts distribution =================
    if _a.skip_stage1:
        s1 = {"skipped": True}
        print("=== STAGE1 SKIPPED ===", flush=True)
        plens = []
    for g in gids[:200]:
        plens.append((g, len(encode_prompt(tokenizer, by_group[g]["think"]["prompt"]))))
    plens.sort(key=lambda x: x[1])
    picked = plens[:11] + plens[len(plens) // 2 - 5:len(plens) // 2 + 5] + plens[-11:]
    stage1 = []
    fwd_shapes = {"prefill": None, "beam_first": None}
    for gi, (g, plen) in enumerate(picked):
        prompt = by_group[g]["think"]["prompt"]
        full = encode_prompt(tokenizer, prompt)
        prefix = full[:-1]
        dev = model.device
        with torch.inference_mode():
            # baseline: batch=4 full forward
            b4 = torch.tensor([full] * 4, device=dev, dtype=torch.long)
            bm = torch.ones(4, len(full), device=dev, dtype=torch.long)
            b_out = model(input_ids=b4, attention_mask=bm, use_cache=True, return_dict=True)
            b_logits = b_out.logits[:, -1, :].float()  # [4, V]
            # cache: prefill [1, len-1] + repeat x4 + incremental last token
            pre = model(input_ids=torch.tensor([prefix], device=dev, dtype=torch.long),
                        attention_mask=torch.ones(1, len(prefix), device=dev, dtype=torch.long),
                        use_cache=True, return_dict=True)
            cache = pre.past_key_values
            assert cache.get_seq_length() == len(prefix)
            cache.batch_repeat_interleave(4)
            last_ids = torch.tensor([[full[-1]]] * 4, device=dev, dtype=torch.long)
            attn = torch.ones(4, len(full), device=dev, dtype=torch.long)
            c_out = model(input_ids=last_ids, attention_mask=attn,
                          past_key_values=cache, use_cache=True, return_dict=True)
            c_logits = c_out.logits[:, -1, :].float()
        r0 = b_logits[0]
        c0 = c_logits[0]
        pd_b = processed_dist(r0.unsqueeze(0))[0]
        pd_c = processed_dist(c0.unsqueeze(0))[0]
        # internal row consistency
        b_rows = [float((b_logits[0] - b_logits[i]).abs().max()) for i in (1, 2, 3)]
        c_rows = [float((c_logits[0] - c_logits[i]).abs().max()) for i in (1, 2, 3)]
        stage1.append(dict(
            g=g, plen=plen,
            raw_max_abs_diff=float((r0 - c0).abs().max()),
            raw_mean_abs_diff=float((r0 - c0).abs().mean()),
            kl_bc=round(kl(pd_b, pd_c), 6), kl_cb=round(kl(pd_c, pd_b), 6),
            js=round(js(pd_b, pd_c), 6),
            top20=round(topk_overlap(pd_b, pd_c, 20), 4),
            top50=round(topk_overlap(pd_b, pd_c, 50), 4),
            top1_agree=bool(pd_b.argmax() == pd_c.argmax()),
            tpp_support_overlap=round(
                len(set((pd_b > 0).nonzero().flatten().tolist()) & set((pd_c > 0).nonzero().flatten().tolist()))
                / max(len(set((pd_b > 0).nonzero().flatten().tolist())), 1), 4),
            mass_l1=float((pd_b - pd_c).abs().sum()),
            base_rows_maxdiff=max(b_rows), cache_rows_maxdiff=max(c_rows)))
    s1 = dict(
        n=len(stage1),
        raw_max_abs_diff_mean=round(statistics.mean(x["raw_max_abs_diff"] for x in stage1), 4),
        raw_mean_abs_diff_mean=round(statistics.mean(x["raw_mean_abs_diff"] for x in stage1), 4),
        js_mean=round(statistics.mean(x["js"] for x in stage1), 7),
        js_max=round(max(x["js"] for x in stage1), 7),
        kl_mean=round(statistics.mean((x["kl_bc"] + x["kl_cb"]) / 2 for x in stage1), 7),
        top20_mean=round(statistics.mean(x["top20"] for x in stage1), 4),
        top50_mean=round(statistics.mean(x["top50"] for x in stage1), 4),
        top1_agree_rate=round(sum(1 for x in stage1 if x["top1_agree"]) / len(stage1), 4),
        tpp_support_overlap_mean=round(statistics.mean(x["tpp_support_overlap"] for x in stage1), 4),
        mass_l1_mean=round(statistics.mean(x["mass_l1"] for x in stage1), 6),
        base_rows_maxdiff_max=round(max(x["base_rows_maxdiff"] for x in stage1), 5),
        cache_rows_maxdiff_max=round(max(x["cache_rows_maxdiff"] for x in stage1), 5))
    print("=== STAGE1 ===", flush=True)
    print(json.dumps(s1, ensure_ascii=False, indent=1), flush=True)

    # ================= stage 2: 64 groups x G=4 sampling (sharded) =================
    groups64 = gids[:64][_a.shard * (64 // _a.nshards):(_a.shard + 1) * (64 // _a.nshards)]
    recs = []
    done = 0
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT, encoding="utf-8"))
        recs = ck["records"]
        done = ck.get("done", 0)
        print(f"resume from group {done}", flush=True)

    fwd_log = []

    def _hook(module, args, kwargs):
        inp = args[0] if args else kwargs.get("input_ids")
        fwd_log.append(tuple(inp.shape))

    handle = model.base_model.model.register_forward_pre_hook(_hook, with_kwargs=True)

    def gen_baseline(pids):
        torch.cuda.reset_peak_memory_stats()
        t0 = sync_timer()
        with torch.inference_mode():
            inp = {"input_ids": torch.tensor([pids] * 4, device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(4, len(pids), device=model.device, dtype=torch.long)}
            outs = model.generate(**inp, max_new_tokens=2048, do_sample=True,
                                  temperature=0.9, top_p=0.95, num_return_sequences=1,
                                  pad_token_id=pad_id,
                                  stopping_criteria=StoppingCriteriaList([_Stop()]))
        wall = elapsed(t0)
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        cots = []
        for o in outs:
            c = o[len(pids):].tolist()
            try:
                pos = c.index(tid)
                c = c[:pos + 1]
            except ValueError:
                pass
            cots.append(c)
        return cots, wall, peak

    def gen_cache(pids):
        prefix = pids[:-1]
        dev = model.device
        torch.cuda.reset_peak_memory_stats()
        fwd_log.clear()
        t0 = sync_timer()
        with torch.inference_mode():
            pre = model(input_ids=torch.tensor([prefix], device=dev, dtype=torch.long),
                        attention_mask=torch.ones(1, len(prefix), device=dev, dtype=torch.long),
                        use_cache=True, return_dict=True)
        prefill_sec = elapsed(t0)
        cache = pre.past_key_values
        t0 = sync_timer()
        cache.batch_repeat_interleave(4)
        rep_sec = elapsed(t0)
        t0 = sync_timer()
        with torch.inference_mode():
            outs = model.generate(
                input_ids=torch.tensor([pids] * 4, device=dev, dtype=torch.long),
                attention_mask=torch.ones(4, len(pids), device=dev, dtype=torch.long),
                past_key_values=cache,
                max_new_tokens=2048, do_sample=True,
                temperature=0.9, top_p=0.95, num_return_sequences=1,
                pad_token_id=pad_id,
                stopping_criteria=StoppingCriteriaList([_Stop()]))
        gen_sec = elapsed(t0)
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        cots = []
        for o in outs:
            c = o[len(pids):].tolist()
            try:
                pos = c.index(tid)
                c = c[:pos + 1]
            except ValueError:
                pass
            cots.append(c)
        shapes = list(fwd_log)
        return cots, prefill_sec, rep_sec, gen_sec, peak, shapes

    for gi in range(done, len(groups64)):
        g = groups64[gi]
        pids = encode_prompt(tokenizer, by_group[g]["think"]["prompt"])
        torch.manual_seed(SEED + gi)
        b_cots, b_wall, b_peak = gen_baseline(pids)
        torch.manual_seed(SEED + gi)
        c_cots, p_sec, r_sec, c_gen_sec, c_peak, shapes = gen_cache(pids)
        b_closed = sum(1 for c in b_cots if "</think>" in tokenizer.decode(c, skip_special_tokens=False))
        c_closed = sum(1 for c in c_cots if "</think>" in tokenizer.decode(c, skip_special_tokens=False))
        b_lens = [len(c) for c in b_cots]
        c_lens = [len(c) for c in c_cots]
        b_uniq = len(set(tuple(c) for c in b_cots))
        c_uniq = len(set(tuple(c) for c in c_cots))
        recs.append(dict(
            group=g, prompt_len=len(pids),
            base=dict(wall=round(b_wall, 2), peak_mb=b_peak, closed=b_closed,
                      lens=b_lens, uniq=b_uniq),
            cache=dict(prefill=round(p_sec, 3), repeat=round(r_sec, 4),
                       gen=round(c_gen_sec, 2), total=round(p_sec + c_gen_sec, 2),
                       peak_mb=c_peak, closed=c_closed, lens=c_lens, uniq=c_uniq),
            fwd_shapes=shapes[:6]))
        json.dump({"records": recs, "done": gi + 1}, open(CKPT, "w", encoding="utf-8"))
        if gi % 8 == 0:
            print(f"  group {gi + 1}/{len(groups64)}", flush=True)
    handle.remove()

    # ---- stats ----
    n = len(recs)
    b_all = [l for r in recs for l in r["base"]["lens"]]
    c_all = [l for r in recs for l in r["cache"]["lens"]]
    def pct(a, p):
        a = sorted(a)
        return a[max(0, int(p * len(a)) - 1)]
    def lens_stat(a):
        return dict(mean=round(statistics.mean(a), 1), p50=pct(a, .5), p90=pct(a, .9),
                    p95=pct(a, .95), max=max(a))
    b_closure = sum(r["base"]["closed"] for r in recs) / (n * 4)
    c_closure = sum(r["cache"]["closed"] for r in recs) / (n * 4)
    b_uniq_ratio = sum(r["base"]["uniq"] for r in recs) / (n * 4)
    c_uniq_ratio = sum(r["cache"]["uniq"] for r in recs) / (n * 4)
    b_dup = sum(1 for r in recs for i in range(4) for j in range(i + 1, 4)
                if r["base"]["lens"][i] == r["base"]["lens"][j] and
                len(set(range(4))) == 4 and False)  # placeholder
    # duplicate ratio: pairs of identical CoTs within group
    def dup_ratio(recs, key):
        pairs = 0
        tot = 0
        for r in recs:
            cots = r[key]["lens"]
            # approximate duplicates by identical length+closure pattern is weak;
            # use exact length match as proxy (cheap)
            seen = collections.Counter(cots)
            for v, cnt in seen.items():
                if cnt > 1:
                    pairs += cnt * (cnt - 1) // 2
            tot += 6
        return pairs / tot
    b_len_std = [statistics.pstdev(r["base"]["lens"]) for r in recs]
    c_len_std = [statistics.pstdev(r["cache"]["lens"]) for r in recs]
    # bucket by prompt len tertiles
    recs_sorted = sorted(recs, key=lambda r: r["prompt_len"])
    third = len(recs_sorted) // 3
    buckets = {"short": recs_sorted[:third], "medium": recs_sorted[third:2 * third],
               "long": recs_sorted[2 * third:]}
    bucket_out = {}
    for name, bs in buckets.items():
        if not bs:
            continue
        bm = statistics.mean(r["base"]["wall"] for r in bs)
        cm = statistics.mean(r["cache"]["total"] for r in bs)
        bucket_out[name] = dict(n=len(bs),
                                base_sec=round(bm, 2), cache_sec=round(cm, 2),
                                speedup=round(bm / cm, 3),
                                plen_range=(bs[0]["prompt_len"], bs[-1]["prompt_len"]))
    # shapes proof from last cache run
    fwd_proof = recs[-1]["fwd_shapes"] if recs else None
    report = dict(
        stage1=s1,
        stage2=dict(
            n_groups=n, n_cots_baseline=n * 4, n_cots_cache=n * 4,
            closure_base=round(b_closure, 4), closure_cache=round(c_closure, 4),
            no_close_base=round(1 - b_closure, 4), no_close_cache=round(1 - c_closure, 4),
            len_base=lens_stat(b_all), len_cache=lens_stat(c_all),
            uniq_ratio_base=round(b_uniq_ratio, 4), uniq_ratio_cache=round(c_uniq_ratio, 4),
            dup_pairs_base=dup_ratio(recs, "base"), dup_pairs_cache=dup_ratio(recs, "cache"),
            len_std_base_mean=round(statistics.mean(b_len_std), 2),
            len_std_cache_mean=round(statistics.mean(c_len_std), 2),
            fwd_proof=fwd_proof),
        perf=dict(
            base=dict(mean=round(statistics.mean(r["base"]["wall"] for r in recs), 2),
                      p50=round(pct([r["base"]["wall"] for r in recs], .5), 2),
                      p90=round(pct([r["base"]["wall"] for r in recs], .9), 2),
                      p95=round(pct([r["base"]["wall"] for r in recs], .95), 2),
                      max=round(max(r["base"]["wall"] for r in recs), 2)),
            cache=dict(prefill_mean=round(statistics.mean(r["cache"]["prefill"] for r in recs), 3),
                       repeat_mean=round(statistics.mean(r["cache"]["repeat"] for r in recs), 4),
                       gen_mean=round(statistics.mean(r["cache"]["gen"] for r in recs), 2),
                       total_mean=round(statistics.mean(r["cache"]["total"] for r in recs), 2),
                       p50=round(pct([r["cache"]["total"] for r in recs], .5), 2),
                       p90=round(pct([r["cache"]["total"] for r in recs], .9), 2),
                       p95=round(pct([r["cache"]["total"] for r in recs], .95), 2),
                       max=round(max(r["cache"]["total"] for r in recs), 2)),
            speedup=round(statistics.mean(r["base"]["wall"] for r in recs)
                          / statistics.mean(r["cache"]["total"] for r in recs), 3),
            peak_base_mb=max(r["base"]["peak_mb"] for r in recs),
            peak_cache_mb=max(r["cache"]["peak_mb"] for r in recs)),
        buckets=bucket_out,
    )
    print("=== REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    with open("/data/GRPO/logs/think_prefix_cache_%s.json" % os.environ.get("SHARD_ID", "0"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
