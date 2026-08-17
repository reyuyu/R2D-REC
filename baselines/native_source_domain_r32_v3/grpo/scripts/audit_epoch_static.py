# -*- coding: utf-8 -*-
"""Full-epoch sampler static audit over all 1549 groups (CPU)."""
import json
import sys
sys.path.insert(0, "/data/GRPO/scripts")
import trl_import_fix  # noqa
from grpo_trl_trainer import build_route_dataset, RouteAwareRepeatSampler, ROUTE_G, ROUTE_LOSS_W, M_THINK, M_NO

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
ds = build_route_dataset(DATA, n_groups=1549, seed=20260816, chunk=8)
rows = list(ds)
print("records:", len(rows), flush=True)

sam = RouteAwareRepeatSampler(ds, generation_batch_size=16, repeat_count=2, shuffle=False)
chunks = sam._chunks
n_t = sum(1 for r, _ in chunks if r == "think")
n_n = sum(1 for r, _ in chunks if r == "no_think")
t_chunks = [inds for r, inds in chunks if r == "think"]
n_chunks = [inds for r, inds in chunks if r == "no_think"]
t_groups = {rows[i]["recommendation_group_id"] for c in t_chunks for i in c}
n_groups = {rows[i]["recommendation_group_id"] for c in n_chunks for i in c}
all_gids = [r["recommendation_group_id"] for r in rows if r["route"] == "think"]
dropped = [g for g in all_gids if g not in t_groups]
dropped_n = [g for g in all_gids if g not in n_groups]
steps = len(sam) // 16
res = {
    "records": len(rows),
    "think_chunks": n_t, "nothink_chunks": n_n,
    "think_groups_covered": len(t_groups), "nothink_groups_covered": len(n_groups),
    "total_groups": len(all_gids),
    "dropped_think": dropped, "dropped_nothink": dropped_n,
    "dropped_think_count": len(dropped), "dropped_nothink_count": len(dropped_n),
    "think_rollouts": n_t, "nothink_rollouts": n_n,
    "optimizer_steps": steps,
    "route_effective_weight": {
        "think": n_t * 16 * ROUTE_LOSS_W["think"],
        "nothink": n_n * 16 * ROUTE_LOSS_W["no_think"],
    },
    "sampler_len": len(sam),
    "expected": {"think_groups": 1548, "nothink_groups": 1548,
                 "think_rollouts": 387, "nothink_rollouts": 774, "steps": 2322},
}
print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
with open("/data/GRPO/logs/epoch_static_audit.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
