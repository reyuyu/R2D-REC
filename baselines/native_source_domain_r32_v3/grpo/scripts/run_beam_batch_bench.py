# -*- coding: utf-8 -*-
"""Think reward Beam32 micro-batch benchmark.
Fixed 1 Think batch (4 prompts x G=4 = 16 CoTs). No backward / no training.
Tests context_batch_size = 1 / 2 / 4; correctness vs batch=1; OOM handling.
Usage: CUDA_VISIBLE_DEVICES=0 python run_beam_batch_bench.py"""
import collections
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_sid import final_sid, think_reward, parse_sid
from grpo_model import load_model, encode_prompt, generate_batch

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816


def load_think_batch(path, n_prompts=4):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = __import__("random").Random(SEED)
    rng.shuffle(gids)
    prompts, gold_sets = [], []
    for gid in gids[:n_prompts]:
        rec = by_group[gid]["think"]
        prompts.append(rec["prompt"])
        gs = set()
        for s in rec["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gs.add(t)
        gold_sets.append(gs)
    return prompts, gold_sets


def sample_cots(model, tokenizer, prompts, seed, M=4):
    """4 prompts x 4 = 16 CoTs in ONE batched call (4 inputs x 4 returns);
    trimmed at </think>. Returns list of cot strings in [p0x4, p1x4, ...] order."""
    torch.manual_seed(seed)
    inputs = [encode_prompt(tokenizer, p) for p in prompts]
    texts = generate_batch(model, tokenizer, inputs, max_new_tokens=2048,
                           do_sample=True, temperature=0.9, top_p=0.95,
                           num_return_sequences=M)
    cots = []
    for t in texts:
        idx = t.find("</think>")
        cot = t[: idx + len("</think>")] if idx >= 0 else t
        cots.append(cot)
    return cots


def beam32_all(model, tokenizer, prompts, cots, context_batch_size):
    """Beam32 over all CoTs (cots in [p0x4, p1x4, ...] order), batched in
    groups of context_batch_size. Returns per-cot candidate lists (32 each)."""
    inputs = []
    for i, cot in enumerate(cots):
        prompt_ids = encode_prompt(tokenizer, prompts[i // 4])
        cot_ids = tokenizer.encode(cot, add_special_tokens=False)
        inputs.append(prompt_ids + cot_ids)
    results = []
    for i in range(0, len(inputs), context_batch_size):
        chunk = inputs[i:i + context_batch_size]
        texts = generate_batch(model, tokenizer, chunk, max_new_tokens=128,
                               num_beams=32, num_return_sequences=32)
        per_cot = [[] for _ in chunk]
        for j, t in enumerate(texts):
            per_cot[j // 32].append(final_sid(t))
        results.extend(per_cot)
    return results


def reward_stats(cands_per_cot, gold_sets):
    """Per-cot (group) hierarchical reward; each cot belongs to its group gold set."""
    rewards = []
    hits = []
    # cots are grouped per prompt: prompts = 4, M = 4
    for idx, cands in enumerate(cands_per_cot):
        gid = idx // 4  # cot -> prompt index (4 cots per prompt)
        gs = gold_sets[gid]
        r, ec, ac, a2 = think_reward(cands, gs)
        rewards.append(r)
        hits.append((ec, ac, a2))
    return rewards, hits


def main():
    torch.manual_seed(SEED)
    model, tokenizer, template = load_model("cuda:0")
    prompts, gold_sets = load_think_batch(DATA)
    print(f"prompts={len(prompts)} gold sizes={[len(g) for g in gold_sets]}", flush=True)

    # CoT sampling (once, fixed)
    t0 = time.time()
    cots = sample_cots(model, tokenizer, prompts, SEED)
    cot_time = time.time() - t0
    lens = [len(c) for c in cots]
    print(f"CoT sampling: {cot_time:.1f}s, mean_len={statistics.mean(lens):.0f} "
          f"p95={sorted(lens)[int(0.95*len(lens))]}", flush=True)

    results = {}
    baseline = None
    for cbs in [1, 2, 4]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            cands = beam32_all(model, tokenizer, prompts, cots, context_batch_size=cbs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[cbs] = dict(status="OOM", beam_time=None, total=None,
                                peak_mb=None, candidates=None)
            print(f"[cbs={cbs}] OOM", flush=True)
            continue
        beam_time = time.time() - t0
        t1 = time.time()
        rewards, hits = reward_stats(cands, gold_sets)
        reward_time = time.time() - t1
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        counts = [len(c) for c in cands]
        exact = sum(h[0] for h in hits)
        ab = sum(h[1] for h in hits)
        a = sum(h[2] for h in hits)
        results[cbs] = dict(
            status="OK", beam_time=beam_time, reward_time=reward_time,
            total=cot_time + beam_time + reward_time,
            peak_mb=peak, candidates=counts, rewards=rewards, hits=hits,
            exact=exact, ab=ab, a=a,
        )
        per_cot = beam_time / len(cots)
        print(f"[cbs={cbs}] beam={beam_time:.1f}s ({per_cot:.1f}s/cot) "
              f"reward={reward_time:.1f}s peak={peak}MB cand_sizes={set(counts)} "
              f"exact={exact} ab={ab} a={a}", flush=True)
        if cbs == 1:
            baseline = results[1]
        else:
            b = baseline
            ok_cands = b["candidates"] == counts
            ok_rewards = b["rewards"] == rewards
            ok_hits = b["hits"] == hits
            print(f"[cbs={cbs}] correctness: cands={ok_cands} rewards={ok_rewards} "
                  f"hits={ok_hits}", flush=True)
            results[cbs]["correctness_match"] = ok_cands and ok_rewards and ok_hits
            if not (ok_cands and ok_rewards and ok_hits):
                results[cbs]["status"] = "MISMATCH"

    # summary table
    print("\n=== SUMMARY ===")
    print(f"{'ctx batch':>9} | {'Beam32 time':>11} | {'Think total':>11} | {'peak MB':>7} | {'speedup':>7} | status")
    t1_total = results[1]["total"]
    for cbs in [1, 2, 4]:
        r = results[cbs]
        if r["status"] == "OK":
            sp = t1_total / r["total"]
            print(f"{cbs:>9} | {r['beam_time']:>9.1f}s | {r['total']:>9.1f}s | {r['peak_mb']:>7} | {sp:>6.2f}x | {r['status']}")
        else:
            print(f"{cbs:>9} | {'-':>11} | {'-':>11} | {'-':>7} | {'-':>7} | {r['status']}")
    with open("/data/GRPO/logs/beam_batch_bench.json", "w", encoding="utf-8") as f:
        json.dump({str(k): {kk: (vv if not isinstance(vv, list) else vv[:4] if kk in ("rewards",) else vv)
                             for kk, vv in v.items()} for k, v in results.items()},
                  f, ensure_ascii=False, indent=2)
    print("saved /data/GRPO/logs/beam_batch_bench.json")


if __name__ == "__main__":
    main()
