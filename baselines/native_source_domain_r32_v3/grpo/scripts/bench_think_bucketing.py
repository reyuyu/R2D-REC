# -*- coding: utf-8 -*-
"""Think length-bucketing GPU rollout-only benchmark (NO training).
Same 16 groups under two groupings:
  random : existing seed-shuffle order, every 4 groups = 1 block (4 ranks)
  bucket : same 16 groups sorted by Think prompt length, every 4 = 1 block
Per block (4 ranks simulated on 1 GPU): per-rank CoT gen (G=4, per-sample stop,
T=0.9/top_p=0.95), per-rank Beam32 (cbs=1, 32/32/128), rank wall = gen+beam,
global wall = max(4 ranks), straggler = argmax. Beam reward unchanged."""
import collections
import json
import random
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
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)
    # take the 16 groups of the first 4 random blocks (same set as static bench)
    groups16 = gids[:16]
    prompts = [by_group[g]["think"]["prompt"] for g in groups16]
    golds = []
    for g in groups16:
        gs = set()
        for s in by_group[g]["think"]["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gs.add(t)
        golds.append(gs)

    model, tokenizer, template = load_model("cuda:0")
    model.eval()
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return input_ids[:, -1] == tid

    def gen_cot4(pids):
        """One rank's CoT generation: G=4 samples, per-sample stop at </think>.
        Returns (wall_sec, lens, peak_mb, cot_ids_list)."""
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cots = []
        with torch.inference_mode():
            inp = {"input_ids": torch.tensor([pids] * 4, device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(4, len(pids), device=model.device, dtype=torch.long)}
            outs = model.generate(**inp, max_new_tokens=2048, do_sample=True,
                                  temperature=0.9, top_p=0.95, num_return_sequences=1,
                                  pad_token_id=tokenizer.pad_token_id,
                                  stopping_criteria=StoppingCriteriaList([_Stop()]))
            for o in outs:
                c = o[len(pids):].tolist()
                try:
                    pos = c.index(tid)
                    c = c[:pos + 1]
                except ValueError:
                    pass
                cots.append(c)
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        return wall, [len(c) for c in cots], peak, cots

    def beam_cots(pids, cots):
        """One rank's Beam32 over its 4 cots (cbs=1 serial). Returns (wall, peak)."""
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            for cot in cots:
                generate_batch(model, tokenizer, [pids + cot],
                               max_new_tokens=128, num_beams=32,
                               num_return_sequences=32)
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        return wall, peak

    def run_block(blk_groups):
        """blk_groups: 4 group ids. Returns dict."""
        pids_list = [encode_prompt(tokenizer, by_group[g]["think"]["prompt"]) for g in blk_groups]
        plens = [len(p) for p in pids_list]
        rows_r = []
        for i, pids in enumerate(pids_list):
            gw, clens, gpeak, cots = gen_cot4(pids)
            bw, bpeak = beam_cots(pids, cots)
            rows_r.append(dict(
                rank=i, plen=plens[i], gen_sec=round(gw, 1),
                comp_len_mean=round(sum(clens) / len(clens), 1),
                comp_len_max=max(clens), beam_sec=round(bw, 1),
                rank_wall=round(gw + bw, 1), gen_peak_mb=gpeak, beam_peak_mb=bpeak))
        walls = [r["rank_wall"] for r in rows_r]
        straggler = walls.index(max(walls))
        global_wall = max(walls)
        peak = max(max(r["gen_peak_mb"] for r in rows_r),
                   max(r["beam_peak_mb"] for r in rows_r))
        return dict(groups=blk_groups, plens=plens,
                    ranks=rows_r, global_wall=round(global_wall, 1),
                    straggler_rank=straggler, peak_mb=peak)

    # groupings over the SAME 16 groups
    random_blocks = [groups16[i:i + 4] for i in range(0, 16, 4)]
    by_len = sorted(range(16), key=lambda i: len(encode_prompt(tokenizer, prompts[i])))
    bucket_blocks = [[groups16[by_len[i + j]] for j in range(4)] for i in range(0, 16, 4)]

    results = {"random": [], "bucket": []}
    for tag, blocks in (("random", random_blocks), ("bucket", bucket_blocks)):
        for blk in blocks:
            r = run_block(blk)
            results[tag].append(r)
            print(f"[{tag}] plens={r['plens']} global_wall={r['global_wall']}s "
                  f"straggler={r['straggler_rank']} peak={r['peak_mb']}MB "
                  f"ranks={[(x['rank'], x['gen_sec'], x['beam_sec'], x['rank_wall']) for x in r['ranks']]}",
                  flush=True)

    mean_r = sum(b["global_wall"] for b in results["random"]) / len(results["random"])
    mean_b = sum(b["global_wall"] for b in results["bucket"]) / len(results["bucket"])
    summary = dict(
        mean_random_wall=round(mean_r, 1),
        mean_bucket_wall=round(mean_b, 1),
        speedup=round(mean_r / mean_b, 3),
        improvement_pct=round((1 - mean_b / mean_r) * 100, 1),
    )
    results["summary"] = summary
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    with open("/data/GRPO/logs/think_bucket_gpu.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
