# -*- coding: utf-8 -*-
"""Prompt token parity audit v2: A=SFT llamafactory encode_multiturn user ids,
B=TRL rollout (shared render_prompt), C=Beam32 encode_prompt. Requires A==B==C."""
import json, random, sys
sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import encode_prompt, render_prompt
from llamafactory.data.template import TEMPLATES

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
rng = random.Random(20260816)
rng.shuffle(rows)
sample = rows[:3]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/models/onereason-8b-pretrain-competition",
                                    trust_remote_code=True)
tpl = TEMPLATES["qwen3_nothink"]

results = []
all_ok = True
for i, r in enumerate(sample):
    prompt = r["prompt"]
    # A: SFT actual rendering (user msg incl. assistant head inside format_user)
    a_ids = tpl.encode_multiturn(
        tok, [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
        system=None)[0][0]
    # B: TRL rollout after fix (shared renderer)
    b_ids = tok.encode(render_prompt(tok, prompt), add_special_tokens=False)
    # C: Beam32
    c_ids = encode_prompt(tok, prompt)
    ok_ab = a_ids == b_ids
    ok_ac = a_ids == c_ids
    ok_bc = b_ids == c_ids
    all_ok = all_ok and ok_ab and ok_ac and ok_bc
    results.append({
        "idx": i, "route": r["route"],
        "len_A": len(a_ids), "len_B": len(b_ids), "len_C": len(c_ids),
        "A==B": ok_ab, "A==C": ok_ac, "B==C": ok_bc,
        "A_head": a_ids[:8], "B_head": b_ids[:8], "C_head": c_ids[:8],
    })
for res in results:
    print(json.dumps(res, ensure_ascii=False), flush=True)
print("PARITY:", "ALL EQUAL" if all_ok else "MISMATCH FOUND", flush=True)
with open("/data/GRPO/logs/prompt_parity_audit.json", "w", encoding="utf-8") as f:
    json.dump({"all_equal": all_ok, "results": results}, f, ensure_ascii=False, indent=2)
