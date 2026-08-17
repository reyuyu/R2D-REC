# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 smoke runner (inference-only, batched).
Usage: python run_smoke.py --groups 64 --device cuda:0 --seed 20260816 --out ..."""
import argparse
import collections
import json
import os
import random
import statistics
import time
import torch

from grpo_sid import parse_sid, final_sid, q_reward, think_reward, raw_w
from grpo_model import load_model, encode_prompt, generate_batch

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20)]
THINK_CLOSE = "</think>"


def load_data(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    return by_group


def gold_set(rec):
    out = set()
    for s in rec.get("all_gold_sids", []):
        t = parse_sid(s)
        if t:
            out.add(t)
    return out


def select_groups(by_group, n, seed):
    rng = random.Random(seed)
    def bucket_of(k):
        for lo, hi in BUCKETS:
            if lo <= k <= hi:
                return (lo, hi)
        return BUCKETS[-1]
    strata = collections.defaultdict(list)
    for gid, recs in by_group.items():
        r = recs.get("think") or recs.get("no_think")
        k = len(r.get("all_gold_sids", []))
        strata[(r.get("target_domain", "?"), bucket_of(k))].append(gid)
    picked, seen = [], set()
    keys = sorted(strata.keys())
    rng.shuffle(keys)
    while len(picked) < n:
        any_left = False
        for key in keys:
            pool = [g for g in strata[key] if g not in seen]
            if not pool:
                continue
            any_left = True
            g = rng.choice(pool)
            seen.add(g)
            picked.append(g)
            if len(picked) >= n:
                break
        if not any_left:
            break
    return [by_group[g] for g in picked]


def rollout_group(model, tokenizer, rec_think, rec_nothink, golds):
    out = {}
    # ---------- NoThink: 16 samples in one batched call ----------
    nt_ids = encode_prompt(tokenizer, rec_nothink["prompt"])
    texts = generate_batch(model, tokenizer, [nt_ids], max_new_tokens=128,
                           do_sample=True, temperature=1.0, top_p=1.0,
                           num_return_sequences=16)
    no_rewards, no_sids = [], []
    for t in texts:
        sid = final_sid(t)
        no_sids.append(sid)
        no_rewards.append(q_reward(sid, golds))
    out["no_rewards"] = no_rewards
    out["no_sids"] = no_sids

    # NoThink Beam32 monitor (one call)
    beam_texts = generate_batch(model, tokenizer, [nt_ids], max_new_tokens=128,
                                num_beams=32, num_return_sequences=32)
    out["nt_beam_sids"] = [final_sid(t) for t in beam_texts]

    # ---------- Think: 4 CoT samples in one batched call ----------
    th_ids = encode_prompt(tokenizer, rec_think["prompt"])
    cot_texts = generate_batch(model, tokenizer, [th_ids], max_new_tokens=2048,
                               do_sample=True, temperature=0.9, top_p=0.95,
                               num_return_sequences=4)
    think_rewards, cot_lens, trunc = [], [], 0
    per_cot = []
    for cot in cot_texts:
        # manual stop at </think> (keep it)
        idx = cot.find(THINK_CLOSE)
        if idx >= 0:
            cot = cot[:idx + len(THINK_CLOSE)]
        else:
            trunc += 1
        cot_lens.append(len(cot))
        cot_ids = tokenizer.encode(cot, add_special_tokens=False)
        beam_texts = generate_batch(model, tokenizer, [th_ids + cot_ids],
                                    max_new_tokens=128, num_beams=32,
                                    num_return_sequences=32)
        beam_sids = [final_sid(t) for t in beam_texts]
        r, ec, ac, a2 = think_reward(beam_sids, golds)
        think_rewards.append(r)
        per_cot.append({"beam_sids": beam_sids, "exact": ec, "ab": ac, "a": a2,
                        "cot_truncated": "</think>" not in cot})
    out["think_rewards"] = think_rewards
    out["cot_lens"] = cot_lens
    out["cot_truncation"] = trunc
    out["per_cot"] = per_cot
    return out


def summarize(groups_out):
    n_groups = len(groups_out)
    th_r = [r for g in groups_out for r in g["think_rewards"]]
    th_groups = [g["think_rewards"] for g in groups_out]
    cot_lens = [l for g in groups_out for l in g["cot_lens"]]
    trunc_rate = sum(g["cot_truncation"] for g in groups_out) / max(1, 4 * n_groups)
    beam_sids_all = [s for g in groups_out for p in g["per_cot"] for s in p["beam_sids"]]
    invalid_beam = sum(1 for s in beam_sids_all if s is None) / max(1, len(beam_sids_all))
    mean_exact = statistics.mean(p["exact"] for g in groups_out for p in g["per_cot"])
    mean_ab = statistics.mean(p["ab"] for g in groups_out for p in g["per_cot"])
    mean_a = statistics.mean(p["a"] for g in groups_out for p in g["per_cot"])
    # CoT unique ratio: can't cheaply hash texts (not stored); approximate via length+reward signature
    cot_sig = [tuple(g["think_rewards"]) + (g["cot_lens"][0] if g["cot_lens"] else 0,) for g in groups_out]
    unique_cot = len(set(cot_sig)) / max(1, len(cot_sig))

    th_std_groups = [statistics.pstdev(gr) for gr in th_groups]
    zero_std = sum(1 for s in th_std_groups if s == 0.0) / n_groups
    all_zero = sum(1 for gr in th_groups if all(x == 0 for x in gr)) / n_groups
    mixed = sum(1 for gr in th_groups if len(set(gr)) > 1) / n_groups
    unique_rew = statistics.mean(len(set(gr)) for gr in th_groups)

    no_r = [r for g in groups_out for r in g["no_rewards"]]
    no_sids = [s for g in groups_out for s in g["no_sids"]]
    valid_no = sum(1 for s in no_sids if s is not None) / max(1, len(no_sids))
    uniq_sids = set(s for s in no_sids if s)
    unique_sid = len(uniq_sids) / max(1, sum(1 for s in no_sids if s))
    mean_uniq_a = len(set(s[1] for s in no_sids if s)) / n_groups
    mean_uniq_ab = len(set((s[0], s[1], s[2]) for s in no_sids if s)) / n_groups
    level_dist = collections.Counter(no_r)
    n_no = len(no_r)
    exact_rate = level_dist.get(8.0, 0) / n_no
    ab_only = level_dist.get(2.0, 0) / n_no
    a_only = level_dist.get(0.5, 0) / n_no
    no_grp = [g["no_rewards"] for g in groups_out]
    no_std_groups = [statistics.pstdev(gr) for gr in no_grp]
    no_zero_std = sum(1 for s in no_std_groups if s == 0.0) / n_groups
    no_all_zero = sum(1 for gr in no_grp if all(x == 0 for x in gr)) / n_groups
    no_all_exact = sum(1 for gr in no_grp if all(x == 8.0 for x in gr)) / n_groups
    no_mixed = sum(1 for gr in no_grp if len(set(gr)) > 1) / n_groups

    th_hit = nt_hit = union_hit = both = th_only = nt_only = both_miss = 0
    for g in groups_out:
        golds = g["golds"]
        nt32 = {s for s in g["nt_beam_sids"] if s}
        nt_ok = bool(nt32 & golds)
        for p in g["per_cot"]:
            th32 = {s for s in p["beam_sids"] if s}
            ok = bool(th32 & golds)
            union_hit += bool((th32 | nt32) & golds)
            both += int(ok and nt_ok)
            th_only += int(ok and not nt_ok)
            nt_only += int(not ok and nt_ok)
            both_miss += int(not ok and not nt_ok)
        th_hit += sum(bool({s for s in p['beam_sids'] if s} & golds) for p in g["per_cot"]) / 4
        nt_hit += nt_ok

    srt = sorted(th_r)
    return {
        "groups": n_groups,
        "think": {
            "M": 4,
            "reward_mean": statistics.mean(th_r),
            "reward_std": statistics.pstdev(th_r) if len(th_r) > 1 else 0.0,
            "reward_median": statistics.median(th_r),
            "reward_p25": srt[int(0.25 * len(srt))],
            "reward_p75": srt[int(0.75 * len(srt))],
            "reward_p90": srt[int(0.9 * len(srt))],
            "zero_std_group_ratio": zero_std,
            "all_zero_group_ratio": all_zero,
            "mixed_reward_group_ratio": mixed,
            "mean_unique_reward_count": unique_rew,
            "cot_unique_ratio": unique_cot,
            "cot_mean_len": statistics.mean(cot_lens),
            "cot_p95_len": sorted(cot_lens)[int(0.95 * len(cot_lens))],
            "cot_truncation_rate": trunc_rate,
            "beam_invalid_sid_rate": invalid_beam,
            "exact_hit_mean": mean_exact,
            "ab_hit_mean": mean_ab,
            "a_hit_mean": mean_a,
        },
        "nothink": {
            "M": 16,
            "reward_mean": statistics.mean(no_r),
            "reward_std": statistics.pstdev(no_r) if len(no_r) > 1 else 0.0,
            "reward_median": statistics.median(no_r),
            "valid_sid_rate": valid_no,
            "sampled_sid_unique_ratio": unique_sid,
            "mean_unique_a_count": mean_uniq_a,
            "mean_unique_ab_count": mean_uniq_ab,
            "zero_std_group_ratio": no_zero_std,
            "all_zero_group_ratio": no_all_zero,
            "all_exact_group_ratio": no_all_exact,
            "mixed_reward_group_ratio": no_mixed,
            "reward_level_dist": {str(k): v for k, v in sorted(level_dist.items())},
            "exact_sample_rate": exact_rate,
            "ab_only_sample_rate": ab_only,
            "a_only_sample_rate": a_only,
        },
        "beam_monitor": {
            "think_exact_hit32": th_hit / n_groups,
            "nothink_exact_hit32": nt_hit / n_groups,
            "union_sid_hit64_surrogate": union_hit / (n_groups * 4),
            "both_hit_ratio": both / (n_groups * 4),
            "think_only_ratio": th_only / (n_groups * 4),
            "nothink_only_ratio": nt_only / (n_groups * 4),
            "both_miss_ratio": both_miss / (n_groups * 4),
        },
    }


def bucket_summary(groups_out):
    out = {}
    for lo, hi in BUCKETS:
        gg = [g for g in groups_out if lo <= g["gold_count"] <= hi]
        if not gg:
            continue
        th_r = [x for g in gg for x in g["think_rewards"]]
        no_r = [x for g in gg for x in g["no_rewards"]]
        out[f"{lo}-{hi}"] = {
            "groups": len(gg),
            "think_reward_mean": statistics.mean(th_r),
            "nothink_reward_mean": statistics.mean(no_r),
            "think_mixed_ratio": sum(1 for g in gg if len(set(g["think_rewards"])) > 1) / len(gg),
            "nothink_mixed_ratio": sum(1 for g in gg if len(set(g["no_rewards"])) > 1) / len(gg),
            "think_zero_std_ratio": sum(1 for g in gg if statistics.pstdev(g["think_rewards"]) == 0) / len(gg),
            "nothink_zero_std_ratio": sum(1 for g in gg if statistics.pstdev(g["no_rewards"]) == 0) / len(gg),
            "any_exact_ratio": sum(1 for g in gg if 8.0 in g["think_rewards"] or 8.0 in g["no_rewards"]) / len(gg),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", default="/data/GRPO/logs/smoke_v1.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    by_group = load_data(DATA)
    groups = select_groups(by_group, args.groups, args.seed)
    if args.limit:
        groups = groups[args.offset: args.offset + args.limit]
    print(f"selected {len(groups)} groups (offset={args.offset})", flush=True)

    model, tokenizer, template = load_model(args.device)
    t0 = time.time()
    groups_out = []
    for idx, recs in enumerate(groups):
        rec_think, rec_nothink = recs.get("think"), recs.get("no_think")
        if not rec_think or not rec_nothink:
            continue
        golds = gold_set(rec_think)
        r = rollout_group(model, tokenizer, rec_think, rec_nothink, golds)
        r["golds"] = golds
        r["gold_count"] = len(golds)
        r["group_id"] = rec_think["recommendation_group_id"]
        r["domain"] = rec_think["target_domain"]
        groups_out.append(r)
        if (idx + 1) % 4 == 0 or idx == len(groups) - 1:
            el = time.time() - t0
            print(f"[{idx+1}/{len(groups)}] elapsed={el:.0f}s rate={el/(idx+1):.1f}s/group", flush=True)
    total = time.time() - t0

    summary = summarize(groups_out)
    summary["buckets"] = bucket_summary(groups_out)
    all_raw = []
    for recs in by_group.values():
        r = recs.get("think") or recs.get("no_think")
        if r:
            all_raw.append(raw_w(len(r.get("all_gold_sids", []))))
    summary["runtime"] = {
        "total_sec": total,
        "sec_per_group": total / max(1, len(groups_out)),
        "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else "cpu",
        "peak_allocated_mb": torch.cuda.max_memory_allocated(args.device) // (1024 * 1024) if torch.cuda.is_available() else 0,
        "raw_w_smoke_mean": statistics.mean(raw_w(g["gold_count"]) for g in groups_out),
        "raw_w_global_mean": statistics.mean(all_raw),
        "raw_w_global_norm_mean": 1.0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
