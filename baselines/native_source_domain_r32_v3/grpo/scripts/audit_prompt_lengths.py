# -*- coding: utf-8 -*-
"""Full prompt length audit: render_prompt over all 3098 records."""
import json, statistics, sys
sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import render_prompt
from transformers import AutoTokenizer

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
tok = AutoTokenizer.from_pretrained("/data/models/onereason-8b-pretrain-competition",
                                    trust_remote_code=True)
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
lens = []
over8192 = []
for r in rows:
    n = len(tok.encode(render_prompt(tok, r["prompt"]), add_special_tokens=False))
    lens.append(n)
    if n > 8192:
        over8192.append((r["recommendation_group_id"], r["route"], n))
lens.sort()
def pct(p):
    return lens[min(len(lens) - 1, int(p * len(lens)))]
stats = {
    "n": len(lens),
    "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99),
    "max": lens[-1], "min": lens[0],
    "mean": round(statistics.mean(lens), 1),
    "n_over_8192": len(over8192),
    "over8192_samples": over8192[:10],
    "tokenizer_model_max_length": tok.model_max_length,
}
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
with open("/data/GRPO/logs/prompt_length_audit.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
