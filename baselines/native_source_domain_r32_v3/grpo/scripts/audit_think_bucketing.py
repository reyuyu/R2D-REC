# -*- coding: utf-8 -*-
"""Static analysis: Think 4-group block length imbalance, random vs length-bucket."""
import collections
import json
import random
import statistics
import sys

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import render_prompt
from transformers import AutoTokenizer

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816
tok = AutoTokenizer.from_pretrained("/data/models/onereason-8b-pretrain-competition",
                                    trust_remote_code=True)

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
by_group = collections.defaultdict(dict)
for r in rows:
    by_group[r["recommendation_group_id"]][r["route"]] = r
gids = sorted(by_group.keys())
rng = random.Random(SEED)
rng.shuffle(gids)
gids = gids[:1549]

lens = []
for g in gids:
    t = len(tok.encode(render_prompt(tok, by_group[g]["think"]["prompt"]),
                       add_special_tokens=False))
    lens.append((g, t))

def block_stats(lens4):
    vs = [x for _, x in lens4]
    mn, mx = min(vs), max(vs)
    return dict(min=mn, max=mx, range=mx - mn,
                max_over_mean=round(mx / statistics.mean(vs), 3),
                std=round(statistics.pstdev(vs), 1))

def summarize(blocks):
    rngs = [b["range"] for b in blocks]
    mom = [b["max_over_mean"] for b in blocks]
    stds = [b["std"] for b in blocks]
    def pct(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, int(p * len(a)))]
    return dict(
        n_blocks=len(blocks),
        mean_range=round(statistics.mean(rngs), 1),
        p50_range=pct(rngs, 0.5), p90_range=pct(rngs, 0.9),
        p95_range=pct(rngs, 0.95), max_range=max(rngs),
        mean_max_over_mean=round(statistics.mean(mom), 3),
        p95_max_over_mean=pct(mom, 0.95),
        mean_std=round(statistics.mean(stds), 1),
        p95_std=pct(stds, 0.95),
    )

# random grouping: current gids order (seed shuffle), every 4 = 1 block
rand_blocks = []
for i in range(0, len(lens) - 3, 4):
    rand_blocks.append(block_stats(lens[i:i + 4]))
rand_sum = summarize(rand_blocks)

# bucket grouping: sort by length, every 4 adjacent = 1 block, then shuffle blocks
sorted_lens = sorted(lens, key=lambda x: x[1])
bucket_blocks_raw = []
for i in range(0, len(sorted_lens) - 3, 4):
    bucket_blocks_raw.append(sorted_lens[i:i + 4])
bucket_blocks = [block_stats(b) for b in bucket_blocks_raw]
bucket_sum = summarize(bucket_blocks)
bucket_dropped = len(sorted_lens) - len(bucket_blocks_raw) * 4  # tail group(s)

res = {
    "n_groups": len(lens),
    "random": rand_sum,
    "bucket": bucket_sum,
    "bucket_dropped_tail": bucket_dropped,
    "random_dropped_tail": len(lens) - len(rand_blocks) * 4,
    "improvement_range_mean_pct": round(
        (1 - bucket_sum["mean_range"] / rand_sum["mean_range"]) * 100, 1),
    "improvement_range_p95_pct": round(
        (1 - bucket_sum["p95_range"] / rand_sum["p95_range"]) * 100, 1),
    "improvement_std_mean_pct": round(
        (1 - bucket_sum["mean_std"] / rand_sum["mean_std"]) * 100, 1),
    # first 4 bucket blocks and 4 random blocks for GPU bench (group ids + lens)
    "bench_blocks": {
        "random": [{"groups": [g for g, _ in lens[i:i + 4]],
                    "lens": [x for _, x in lens[i:i + 4]]}
                   for i in range(0, 16, 4)],
        "bucket": [{"groups": [g for g, _ in b],
                    "lens": [x for _, x in b]}
                   for b in bucket_blocks_raw[:4]],
    },
}
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
with open("/data/GRPO/logs/think_bucket_static.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
