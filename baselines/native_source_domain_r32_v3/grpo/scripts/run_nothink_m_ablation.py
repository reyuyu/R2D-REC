# -*- coding: utf-8 -*-
"""NoThink M ablation: M=4/8/16 on the SAME 32 groups (seed 20260816).
Usage: python run_nothink_m_ablation.py --m 16 --device cuda:0 --out ..."""
import argparse
import collections
import json
import os
import statistics
import time
import torch

from grpo_sid import final_sid, q_reward
from grpo_model import load_model, encode_prompt, generate_batch

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20)]
SEED = 20260816


def load_data(path):
    import json as _j
    rows = [_j.loads(l) for l in open(path, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    return by_group


def gold_set(rec):
    from grpo_sid import parse_sid
    out = set()
    for s in rec.get("all_gold_sids", []):
        t = parse_sid(s)
        if t:
            out.add(t)
    return out


def select_groups(by_group, n, seed):
    import random
    rng = random.Random(seed)

    def bucket_of(k):
        for lo, hi in BUCKETS:
            if lo <= k <= hi:
                return (lo, hi)
        return BUCKETS[-1]
    strata = collections.defaultdict(list)
    for gid, recs in by_group.items():
        r = recs.get("think") or recs.get("no_think")
        strata[(r.get("target_domain", "?"), bucket_of(len(r.get("all_gold_sids", []))))].append(gid)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/data/GRPO/logs/nothink_m_ablation_v1.json")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    by_group = load_data(DATA)
    groups = select_groups(by_group, args.groups, SEED)
    if args.limit:
        groups = groups[args.offset: args.offset + args.limit]

    model, tokenizer, template = load_model(args.device)
    t0 = time.time()
    per_group = []
    for idx, recs in enumerate(groups):
        rec_nothink = recs["no_think"]
        golds = gold_set(rec_nothink)
        prompt_ids = encode_prompt(tokenizer, rec_nothink["prompt"])
        texts = generate_batch(model, tokenizer, [prompt_ids], max_new_tokens=128,
                               do_sample=True, temperature=1.0, top_p=1.0,
                               num_return_sequences=args.m)
        rewards, sids = [], []
        for t in texts:
            sid = final_sid(t)
            sids.append(sid)
            rewards.append(q_reward(sid, golds))
        per_group.append(dict(
            group_id=recs["think"]["recommendation_group_id"],
            domain=recs["think"]["target_domain"],
            gold_count=len(golds),
            rewards=rewards,
            sids=[None if s is None else list(s) for s in sids],
        ))
        if (idx + 1) % 8 == 0 or idx == len(groups) - 1:
            el = time.time() - t0
            print(f"[{idx+1}/{len(groups)}] elapsed={el:.0f}s", flush=True)
    total = time.time() - t0

    # ---------------- aggregate stats ----------------
    M = args.m
    n_g = len(per_group)
    all_r = [r for g in per_group for r in g["rewards"]]
    all_s = [s for g in per_group for s in g["sids"]]
    level_dist = collections.Counter(all_r)
    n_total = len(all_r)
    valid = [s for s in all_s if s is not None]
    uniq_sid = set(tuple(s) for s in valid)
    uniq_a = set(s[1] for s in valid)
    uniq_ab = set((s[0], s[1], s[2]) for s in valid)

    zero_std = sum(1 for g in per_group if statistics.pstdev(g["rewards"]) == 0) / n_g
    mixed = sum(1 for g in per_group if len(set(g["rewards"])) > 1) / n_g
    mean_grp_std = statistics.mean(statistics.pstdev(g["rewards"]) for g in per_group)
    mean_uniq_levels = statistics.mean(len(set(g["rewards"])) for g in per_group)
    mean_uniq_sid = statistics.mean(len(set(tuple(s) for s in g["sids"] if s)) for g in per_group)
    mean_uniq_a = statistics.mean(len(set(s[1] for s in g["sids"] if s)) for g in per_group)
    mean_uniq_ab = statistics.mean(len(set((s[0], s[1], s[2]) for s in g["sids"] if s)) for g in per_group)

    # group classification
    n_all_bad = n_partial = n_has_exact = 0
    for g in per_group:
        has_exact = 8.0 in g["rewards"]
        has_partial = any(r in (2.0, 0.5) for r in g["rewards"])
        all_bad = all(r <= 0 for r in g["rewards"])
        if has_exact:
            n_has_exact += 1
        elif has_partial:
            n_partial += 1
        elif all_bad:
            n_all_bad += 1

    # buckets
    buckets = {}
    for lo, hi in BUCKETS:
        gg = [g for g in per_group if lo <= g["gold_count"] <= hi]
        if not gg:
            continue
        buckets[f"{lo}-{hi}"] = dict(
            groups=len(gg),
            mixed_ratio=sum(1 for g in gg if len(set(g["rewards"])) > 1) / len(gg),
            zero_std_ratio=sum(1 for g in gg if statistics.pstdev(g["rewards"]) == 0) / len(gg),
            has_exact_ratio=sum(1 for g in gg if 8.0 in g["rewards"]) / len(gg),
            mean_unique_a=statistics.mean(len(set(s[1] for s in g["sids"] if s)) for g in gg),
            reward_std=statistics.pstdev([r for g in gg for r in g["rewards"]]),
        )

    summary = dict(
        M=M,
        groups=n_g,
        total_actions=n_total,
        valid_sid_rate=len(valid) / n_total,
        sampled_sid_unique_ratio=len(uniq_sid) / max(1, len(valid)),
        mean_unique_sid_count=mean_uniq_sid,
        mean_unique_a_count=mean_uniq_a,
        mean_unique_ab_count=mean_uniq_ab,
        reward_mean=statistics.mean(all_r),
        reward_std=statistics.pstdev(all_r),
        mean_within_group_reward_std=mean_grp_std,
        zero_std_group_ratio=zero_std,
        mixed_reward_group_ratio=mixed,
        mean_unique_reward_levels=mean_uniq_levels,
        reward_level_dist={str(k): v for k, v in sorted(level_dist.items())},
        exact_sample_rate=level_dist.get(8.0, 0) / n_total,
        ab_only_sample_rate=level_dist.get(2.0, 0) / n_total,
        a_only_sample_rate=level_dist.get(0.5, 0) / n_total,
        wrong_a_rate=level_dist.get(0.0, 0) / n_total,
        wrong_domain_rate=level_dist.get(-0.25, 0) / n_total,
        invalid_rate=level_dist.get(-1.0, 0) / n_total,
        group_classification=dict(
            has_exact=n_has_exact / n_g,
            partial_only=n_partial / n_g,
            all_bad=n_all_bad / n_g,
        ),
        buckets=buckets,
        runtime=dict(
            total_sec=total,
            sec_per_group=total / n_g,
            gpu=torch.cuda.get_device_name(args.device),
            peak_allocated_mb=torch.cuda.max_memory_allocated(args.device) // (1024 * 1024),
        ),
        group_ids=[g["group_id"] for g in per_group],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
