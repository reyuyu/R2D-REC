# -*- coding: utf-8 -*-
"""Merge per-GPU smoke summaries into one aggregate summary."""
import json
import statistics
import sys

FILES = [f"/data/GRPO/logs/smoke_v2_gpu{i}.json" for i in range(4)]
OUT = "/data/GRPO/logs/smoke_v2.json"


def weighted_mean(parts, n):
    return sum(p * n_i for p, n_i in zip(parts, n)) / sum(n)


def pooled_std(stds, means, n):
    total = sum(n)
    m = weighted_mean(means, n)
    var = sum(n_i * (s_i * s_i + (m_i - m) ** 2) for s_i, m_i, n_i in zip(stds, means, n)) / total
    return var ** 0.5


def main():
    data = []
    for f in FILES:
        try:
            data.append(json.load(open(f, encoding="utf-8")))
        except Exception as e:
            print(f"missing {f}: {e}")
    if len(data) < 4:
        print("BLOCKED: not all GPU summaries ready")
        sys.exit(1)

    n_groups = sum(d["groups"] for d in data)
    n_arr = [d["groups"] for d in data]

    # ---- Think ----
    th = [d["think"] for d in data]
    th_mean = weighted_mean([t["reward_mean"] for t in th], n_arr)
    th_std = pooled_std([t["reward_std"] for t in th], [t["reward_mean"] for t in th], n_arr)
    merged = {
        "groups": n_groups,
        "think": {
            "M": 4,
            "reward_mean": th_mean,
            "reward_std": th_std,
            "reward_median": weighted_mean([t["reward_median"] for t in th], n_arr),
            "reward_p25": weighted_mean([t["reward_p25"] for t in th], n_arr),
            "reward_p75": weighted_mean([t["reward_p75"] for t in th], n_arr),
            "reward_p90": weighted_mean([t["reward_p90"] for t in th], n_arr),
            "zero_std_group_ratio": weighted_mean([t["zero_std_group_ratio"] for t in th], n_arr),
            "all_zero_group_ratio": weighted_mean([t["all_zero_group_ratio"] for t in th], n_arr),
            "mixed_reward_group_ratio": weighted_mean([t["mixed_reward_group_ratio"] for t in th], n_arr),
            "mean_unique_reward_count": weighted_mean([t["mean_unique_reward_count"] for t in th], n_arr),
            "cot_unique_ratio": weighted_mean([t["cot_unique_ratio"] for t in th], n_arr),
            "cot_mean_len": weighted_mean([t["cot_mean_len"] for t in th], n_arr),
            "cot_p95_len": weighted_mean([t["cot_p95_len"] for t in th], n_arr),
            "cot_truncation_rate": weighted_mean([t["cot_truncation_rate"] for t in th], n_arr),
            "beam_invalid_sid_rate": weighted_mean([t["beam_invalid_sid_rate"] for t in th], n_arr),
            "exact_hit_mean": weighted_mean([t["exact_hit_mean"] for t in th], n_arr),
            "ab_hit_mean": weighted_mean([t["ab_hit_mean"] for t in th], n_arr),
            "a_hit_mean": weighted_mean([t["a_hit_mean"] for t in th], n_arr),
        },
        "nothink": {},
        "beam_monitor": {},
        "buckets": {},
        "runtime": {
            "total_sec": sum(d["runtime"]["total_sec"] for d in data),
            "sec_per_group_avg": sum(d["runtime"]["total_sec"] for d in data) / n_groups,
            "gpu_per_partition": [d["runtime"]["gpu"] for d in data],
            "peak_allocated_mb": max(d["runtime"]["peak_allocated_mb"] for d in data),
            "raw_w_global_mean": data[0]["runtime"]["raw_w_global_mean"],
        },
    }
    no = [d["nothink"] for d in data]
    merged["nothink"] = {
        "M": 16,
        "reward_mean": weighted_mean([t["reward_mean"] for t in no], n_arr),
        "reward_std": pooled_std([t["reward_std"] for t in no], [t["reward_mean"] for t in no], n_arr),
        "reward_median": weighted_mean([t["reward_median"] for t in no], n_arr),
        "valid_sid_rate": weighted_mean([t["valid_sid_rate"] for t in no], n_arr),
        "sampled_sid_unique_ratio": weighted_mean([t["sampled_sid_unique_ratio"] for t in no], n_arr),
        "mean_unique_a_count": weighted_mean([t["mean_unique_a_count"] for t in no], n_arr),
        "mean_unique_ab_count": weighted_mean([t["mean_unique_ab_count"] for t in no], n_arr),
        "zero_std_group_ratio": weighted_mean([t["zero_std_group_ratio"] for t in no], n_arr),
        "all_zero_group_ratio": weighted_mean([t["all_zero_group_ratio"] for t in no], n_arr),
        "all_exact_group_ratio": weighted_mean([t["all_exact_group_ratio"] for t in no], n_arr),
        "mixed_reward_group_ratio": weighted_mean([t["mixed_reward_group_ratio"] for t in no], n_arr),
        "exact_sample_rate": weighted_mean([t["exact_sample_rate"] for t in no], n_arr),
        "ab_only_sample_rate": weighted_mean([t["ab_only_sample_rate"] for t in no], n_arr),
        "a_only_sample_rate": weighted_mean([t["a_only_sample_rate"] for t in no], n_arr),
    }
    # reward level dist: sum counts
    level = {}
    for t in no:
        for k, v in t["reward_level_dist"].items():
            level[k] = level.get(k, 0) + v
    merged["nothink"]["reward_level_dist"] = level

    bm = [d["beam_monitor"] for d in data]
    merged["beam_monitor"] = {
        "think_exact_hit32": weighted_mean([t["think_exact_hit32"] for t in bm], n_arr),
        "nothink_exact_hit32": weighted_mean([t["nothink_exact_hit32"] for t in bm], n_arr),
        "union_sid_hit64_surrogate": weighted_mean([t["union_sid_hit64_surrogate"] for t in bm], n_arr),
        "both_hit_ratio": weighted_mean([t["both_hit_ratio"] for t in bm], n_arr),
        "think_only_ratio": weighted_mean([t["think_only_ratio"] for t in bm], n_arr),
        "nothink_only_ratio": weighted_mean([t["nothink_only_ratio"] for t in bm], n_arr),
        "both_miss_ratio": weighted_mean([t["both_miss_ratio"] for t in bm], n_arr),
    }
    # buckets: merge per-key
    keys = set()
    for d in data:
        keys.update(d["buckets"].keys())
    for k in sorted(keys):
        parts = [d["buckets"].get(k) for d in data]
        parts = [p for p in parts if p]
        if not parts:
            continue
        n = [p["groups"] for p in parts]
        merged["buckets"][k] = {
            "groups": sum(n),
            "think_reward_mean": weighted_mean([p["think_reward_mean"] for p in parts], n),
            "nothink_reward_mean": weighted_mean([p["nothink_reward_mean"] for p in parts], n),
            "think_mixed_ratio": weighted_mean([p["think_mixed_ratio"] for p in parts], n),
            "nothink_mixed_ratio": weighted_mean([p["nothink_mixed_ratio"] for p in parts], n),
            "think_zero_std_ratio": weighted_mean([p["think_zero_std_ratio"] for p in parts], n),
            "nothink_zero_std_ratio": weighted_mean([p["nothink_zero_std_ratio"] for p in parts], n),
            "any_exact_ratio": weighted_mean([p["any_exact_ratio"] for p in parts], n),
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(json.dumps(merged, ensure_ascii=False, indent=2)[:3000])
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
