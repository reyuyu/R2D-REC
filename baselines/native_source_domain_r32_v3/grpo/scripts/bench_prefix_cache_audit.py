# -*- coding: utf-8 -*-
"""Beam32 Prefix-KV Group-Level Equivalence Audit.
32 real Think groups x G=4 = 128 CoTs; the SAME 128 CoTs go through BOTH
A) production baseline generate_batch cbs=1 and B) prefix-KV (prefill once +
cache.batch_repeat_interleave(32) + native generate). Canonical id comparison
(slice + trailing pad strip on BOTH sides), CUDA-synchronized timing.
All beam semantics frozen: 32/32/128, do_sample=False, same eos/pad defaults.
No training / no backward / no optimizer."""
import collections
import gc
import json
import math
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import load_model, encode_prompt, generate_batch
from grpo_sid import think_reward, parse_sid, final_sid
from transformers import StoppingCriteria, StoppingCriteriaList

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
SEED = 20260816
N_GROUPS = 32
G = 4
PAD_STRIP = True
REPORT_PATH = "/data/GRPO/logs/prefix_cache_audit.json"
DETAILS_PATH = "/data/GRPO/logs/prefix_cache_audit_details.json"
RECORDS_PATH = "/data/GRPO/logs/prefix_cache_audit_records.jsonl"
COTS_PATH = "/data/GRPO/logs/prefix_cache_audit_cots.json"


def sync_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(t0):
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def canon(ids, pad_id):
    """slice already applied; strip trailing pad tokens (unified canonicalization)."""
    out = list(ids)
    while out and out[-1] == pad_id:
        out.pop()
    return out


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(SEED)
    rng.shuffle(gids)
    groups = gids[:N_GROUPS]

    model, tokenizer, template = load_model("cuda:0")
    model.eval()
    tid = tokenizer.encode("</think>", add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return input_ids[:, -1] == tid

    # ---------- 1. generate 128 CoTs (per group batch of G=4, fixed seed) ----------
    torch.manual_seed(SEED)
    cot_data = []  # per group: list of (pids, cids)
    cot_generation = []
    print("generating CoTs...", flush=True)
    for gi, g in enumerate(groups):
        prompt = by_group[g]["think"]["prompt"]
        pids = encode_prompt(tokenizer, prompt)
        t0 = sync_timer()
        with torch.inference_mode():
            inp = {"input_ids": torch.tensor([pids] * G, device=model.device, dtype=torch.long),
                   "attention_mask": torch.ones(G, len(pids), device=model.device, dtype=torch.long)}
            outs = model.generate(**inp, max_new_tokens=2048, do_sample=True,
                                  temperature=0.9, top_p=0.95, num_return_sequences=1,
                                  pad_token_id=pad_id,
                                  stopping_criteria=StoppingCriteriaList([_Stop()]))
            cots = []
            for o in outs:
                c = o[len(pids):].tolist()
                try:
                    pos = c.index(tid)
                    c = c[:pos + 1]
                except ValueError:
                    pass
                cots.append(c)
        cot_wall = elapsed(t0)
        cot_data.append([(pids, c) for c in cots])
        cot_generation.append(dict(
            group=g,
            wall=round(cot_wall, 4),
            prompt_len=len(pids),
            cot_lens=[len(c) for c in cots],
        ))
        if gi % 8 == 0:
            print(f"  group {gi}/{N_GROUPS}", flush=True)
    cot_samples = [
        dict(group=groups[gi], trajectory=ti, prompt_ids=pids, cot_ids=cids)
        for gi, samples in enumerate(cot_data)
        for ti, (pids, cids) in enumerate(samples)
    ]
    with open(COTS_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(seed=SEED, samples=cot_samples), f, ensure_ascii=False)
    print("CoT gen done.", flush=True)

    # ---------- 2. beam helpers ----------
    def baseline_beam(ctx):
        t0 = sync_timer()
        _, ids = generate_batch(model, tokenizer, [ctx], max_new_tokens=128,
                                num_beams=32, num_return_sequences=32, return_ids=True)
        wall = elapsed(t0)
        return ids, wall

    def cache_beam(ctx):
        prefix = ctx[:-1]
        dev = model.device
        t0 = sync_timer()
        with torch.inference_mode():
            pre = model(input_ids=torch.tensor([prefix], device=dev, dtype=torch.long),
                        attention_mask=torch.ones(1, len(prefix), dtype=torch.long, device=dev),
                        use_cache=True, return_dict=True)
        prefill_sec = elapsed(t0)
        cache = pre.past_key_values
        assert cache.get_seq_length() == len(prefix)
        t0 = sync_timer()
        cache.batch_repeat_interleave(32)
        rep_sec = elapsed(t0)
        t0 = sync_timer()
        with torch.inference_mode():
            out = model.generate(input_ids=torch.tensor([ctx], device=dev, dtype=torch.long),
                                 attention_mask=torch.ones(1, len(ctx), dtype=torch.long, device=dev),
                                 past_key_values=cache,
                                 max_new_tokens=128, num_beams=32, num_return_sequences=32,
                                 pad_token_id=pad_id)
        beam_sec = elapsed(t0)
        t0 = time.perf_counter()
        ids = [canon(out[i, len(ctx):].tolist(), pad_id) for i in range(32)]
        # Match production generate_batch post-processing in the timed total.
        [tokenizer.decode(x, skip_special_tokens=False) for x in ids]
        post_sec = time.perf_counter() - t0
        return ids, prefill_sec, rep_sec, beam_sec, post_sec

    def score_ids(ids, gold):
        sids = [final_sid(tokenizer.decode(x, skip_special_tokens=False)) for x in ids]
        inv = sum(1 for s in sids if s is None)
        r, ec, ac, a2 = think_reward(sids, gold)
        return sids, inv, ec, ac, a2, r

    # ---------- 3. run both paths on the SAME 128 CoTs ----------
    records = []
    open(RECORDS_PATH, "w", encoding="utf-8").close()
    for gi, g in enumerate(groups):
        gold = set()
        for s in by_group[g]["think"]["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gold.add(t)
        for ti, (pids, cids) in enumerate(cot_data[gi]):
            ctx = pids + cids
            torch.cuda.reset_peak_memory_stats()
            b_ids, b_wall = baseline_beam(ctx)
            b_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
            b_reserved = torch.cuda.max_memory_reserved() // (1024 * 1024)
            b_ids = [canon(x, pad_id) for x in b_ids]
            b_sids, b_inv, b_ec, b_ac, b_a2, b_r = score_ids(b_ids, gold)
            torch.cuda.reset_peak_memory_stats()
            c_ids, p_sec, r_sec, cb_sec, post_sec = cache_beam(ctx)
            c_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
            c_reserved = torch.cuda.max_memory_reserved() // (1024 * 1024)
            c_sids, c_inv, c_ec, c_ac, c_a2, c_r = score_ids(c_ids, gold)
            rec = dict(
                group=g, traj=ti,
                prompt_len=len(pids), cot_len=len(cids), context_len=len(ctx),
                base=dict(ids=b_ids, sids=b_sids, invalid=b_inv, exact=b_ec, ab=b_ac, a=b_a2, reward=b_r,
                          wall=round(b_wall, 4), peak_mb=b_peak, peak_reserved_mb=b_reserved),
                cache=dict(ids=c_ids, sids=c_sids, invalid=c_inv, exact=c_ec, ab=c_ac, a=c_a2, reward=c_r,
                           prefill=round(p_sec, 4), repeat=round(r_sec, 4), beam=round(cb_sec, 4),
                           post=round(post_sec, 4),
                           total=round(p_sec + r_sec + cb_sec + post_sec, 4),
                           peak_mb=c_peak, peak_reserved_mb=c_reserved),
                cand_parity=b_ids == c_ids,
                sid_set_parity=set(b_sids) == set(c_sids),
            )
            records.append(rec)
            with open(RECORDS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if gi % 8 == 0:
            print(f"  beam {gi}/{N_GROUPS}", flush=True)

    # ---------- 4. determinism check (first 8 CoTs, both paths rerun) ----------
    det = []
    ctx_store = []
    for gi in range(N_GROUPS):
        for pids, cids in cot_data[gi]:
            ctx_store.append(pids + cids)
    for i in range(8):
        ctx = ctx_store[i]
        b2_ids, _ = baseline_beam(ctx)
        b2_ids = [canon(x, pad_id) for x in b2_ids]
        c2_ids, _, _, _, _ = cache_beam(ctx)
        g = records[i]["group"]
        gold = set()
        for s in by_group[g]["think"]["all_gold_sids"]:
            t = parse_sid(s)
            if t:
                gold.add(t)
        b2_score = score_ids(b2_ids, gold)
        c2_score = score_ids(c2_ids, gold)
        det.append(dict(
            cot=i,
            base_det=b2_ids == records[i]["base"]["ids"],
            base_sids_det=b2_score[0] == records[i]["base"]["sids"],
            base_reward_det=b2_score[5] == records[i]["base"]["reward"],
            cache_det=c2_ids == records[i]["cache"]["ids"],
            cache_sids_det=c2_score[0] == records[i]["cache"]["sids"],
            cache_reward_det=c2_score[5] == records[i]["cache"]["reward"],
        ))

    # ---------- 5. group-level metrics ----------
    def population_adv(rewards):
        r = torch.tensor(rewards, dtype=torch.float32)
        m = r.mean()
        s = r.std(unbiased=False)
        if s == 0:
            return torch.zeros_like(r)
        return (r - m) / (s + 1e-4)

    grp = collections.defaultdict(list)
    for rec in records:
        grp[rec["group"]].append(rec)

    adv_base, adv_cache = [], []
    zero_flip = {"base0_cache_mixed": 0, "base_mixed_cache0": 0,
                 "both_zero": 0, "both_mixed": 0}
    argmax_agree = argmin_agree = rank_agree = top2_agree = 0
    cosines = []
    sign_flip_pos_neg = 0
    sign_flip_neg_pos = 0
    sign_agree_total = 0
    sign_agree_denom = 0
    adv_exact_total = 0
    adv_close_total = 0
    group_details = []

    def tie_rank_agree(ra, rc):
        # rank by descending value; ties share rank -> compare equivalence classes
        def ranks(xs):
            order = sorted(range(len(xs)), key=lambda i: -xs[i])
            rr = [0] * len(xs)
            cur = 0
            prev = None
            for k, i in enumerate(order):
                if prev is None or abs(xs[i] - prev) > 1e-9:
                    cur = k
                rr[i] = cur
                prev = xs[i]
            return tuple(rr)
        return ranks(ra) == ranks(rc)

    def tie_aware_topk(xs, k=2):
        """Include every trajectory tied with the kth-highest reward."""
        threshold = sorted(xs, reverse=True)[min(k, len(xs)) - 1]
        return {i for i, value in enumerate(xs) if value >= threshold - 1e-9}

    for g, recs in grp.items():
        ra = [r["base"]["reward"] for r in recs]
        rc = [r["cache"]["reward"] for r in recs]
        ab = population_adv(ra)
        ac = population_adv(rc)
        adv_base.append(ab)
        adv_cache.append(ac)
        bz = torch.tensor(ra).std(unbiased=False).item() <= 1e-12
        cz = torch.tensor(rc).std(unbiased=False).item() <= 1e-12
        if bz and cz:
            zero_flip["both_zero"] += 1
        elif bz and not cz:
            zero_flip["base0_cache_mixed"] += 1
        elif not bz and cz:
            zero_flip["base_mixed_cache0"] += 1
        else:
            zero_flip["both_mixed"] += 1
        if not bz and not cz:
            cos = torch.nn.functional.cosine_similarity(ab, ac, dim=0).item()
            cosines.append(cos)
        # ranking
        mx_b = [i for i, v in enumerate(ra) if v == max(ra)]
        mx_c = [i for i, v in enumerate(rc) if v == max(rc)]
        if set(mx_b) == set(mx_c):
            argmax_agree += 1
        mn_b = [i for i, v in enumerate(ra) if v == min(ra)]
        mn_c = [i for i, v in enumerate(rc) if v == min(rc)]
        if set(mn_b) == set(mn_c):
            argmin_agree += 1
        if tie_rank_agree(ra, rc):
            rank_agree += 1
        top2_b = tie_aware_topk(ra)
        top2_c = tie_aware_topk(rc)
        top2_ok = top2_b == top2_c
        if top2_ok:
            top2_agree += 1
        # sign agreement
        for a, c in zip(ab.tolist(), ac.tolist()):
            if a == c:
                adv_exact_total += 1
            if math.isclose(a, c, rel_tol=1e-5, abs_tol=1e-6):
                adv_close_total += 1
            sa = 1 if a > 1e-4 else (-1 if a < -1e-4 else 0)
            sc = 1 if c > 1e-4 else (-1 if c < -1e-4 else 0)
            sign_agree_denom += 1
            if sa == sc:
                sign_agree_total += 1
            elif sa == 1 and sc == -1:
                sign_flip_pos_neg += 1
            elif sa == -1 and sc == 1:
                sign_flip_neg_pos += 1
        group_details.append(dict(
            group=g,
            baseline_rewards=ra,
            cache_rewards=rc,
            baseline_advantages=ab.tolist(),
            cache_advantages=ac.tolist(),
            baseline_zero_std=bz,
            cache_zero_std=cz,
            argmax_agree=set(mx_b) == set(mx_c),
            argmin_agree=set(mn_b) == set(mn_c),
            full_rank_agree=tie_rank_agree(ra, rc),
            baseline_top2=sorted(top2_b),
            cache_top2=sorted(top2_c),
            top2_agree=top2_ok,
            advantage_cosine=(
                torch.nn.functional.cosine_similarity(ab, ac, dim=0).item()
                if not bz and not cz else None
            ),
        ))

    n_grp = len(grp)
    # raw reward stats
    diffs = [r["cache"]["reward"] - r["base"]["reward"] for r in records]
    n_mismatch = sum(1 for d in diffs if abs(d) > 1e-9)
    mean_bias = statistics.mean(diffs)
    mae = statistics.mean(abs(d) for d in diffs)
    max_diff = max(abs(d) for d in diffs)
    # aggregate exact/ab/a
    def aggregate_pair(key):
        base = sum(r["base"][key] for r in records)
        cache = sum(r["cache"][key] for r in records)
        return dict(baseline=base, cache=cache, delta=cache - base)

    agg = {key: aggregate_pair(key) for key in ("exact", "ab", "a", "invalid")}
    cos_sorted = sorted(cosines)
    adv_cos = dict(
        mean=round(statistics.mean(cosines), 6) if cosines else None,
        p50=round(cos_sorted[len(cos_sorted) // 2], 6) if cosines else None,
        p10=round(cos_sorted[max(0, int(0.1 * len(cos_sorted)) - 1)], 6) if cosines else None,
        min=round(cos_sorted[0], 6) if cosines else None,
        n=len(cosines))
    # advantage diff stats
    adv_abs_diffs = torch.stack([(a - c).abs() for a, c in zip(adv_base, adv_cache)])
    sign_agree_rate = sign_agree_total / sign_agree_denom if sign_agree_denom else None
    # pearson surrogate
    flat_b = torch.cat([a for a in adv_base]).tolist()
    flat_c = torch.cat([a for a in adv_cache]).tolist()
    mb, mc = statistics.mean(flat_b), statistics.mean(flat_c)
    cov = sum((x - mb) * (y - mc) for x, y in zip(flat_b, flat_c))
    vb = math.sqrt(sum((x - mb) ** 2 for x in flat_b))
    vc = math.sqrt(sum((y - mc) ** 2 for y in flat_c))
    pearson = cov / (vb * vc) if vb * vc > 0 else None
    distortion = float((adv_abs_diffs.sum()))

    # timing
    perf = collections.defaultdict(list)
    for r in records:
        perf["all"].append(r)
        perf[r["context_len"] < 3000 and "typical" or "long"].append(r)

    def pct(values, p):
        return values[max(0, math.ceil(p * len(values)) - 1)]

    perf_out = {}
    for bucket in ("all", "typical", "long"):
        recs = perf[bucket]
        if not recs:
            perf_out[bucket] = dict(n=0)
            continue
        bwalls = sorted(r["base"]["wall"] for r in recs)
        cwalls = sorted(r["cache"]["total"] for r in recs)
        perf_out[bucket] = dict(
            n=len(recs),
            base=dict(mean=round(statistics.mean(bwalls), 2), p50=round(pct(bwalls, .5), 2),
                      p90=round(pct(bwalls, .9), 2), p95=round(pct(bwalls, .95), 2), max=round(bwalls[-1], 2)),
            cache=dict(prefill_mean=round(statistics.mean(r["cache"]["prefill"] for r in recs), 3),
                       repeat_mean=round(statistics.mean(r["cache"]["repeat"] for r in recs), 4),
                       beam_mean=round(statistics.mean(r["cache"]["beam"] for r in recs), 2),
                       post_mean=round(statistics.mean(r["cache"]["post"] for r in recs), 4),
                       total_mean=round(statistics.mean(cwalls), 2),
                       p50=round(pct(cwalls, .5), 2), p90=round(pct(cwalls, .9), 2),
                       p95=round(pct(cwalls, .95), 2), max=round(cwalls[-1], 2)),
            speedup=round(statistics.mean(bwalls) / statistics.mean(cwalls), 3),
            peak_base_mb=max(r["base"]["peak_mb"] for r in recs),
            peak_cache_mb=max(r["cache"]["peak_mb"] for r in recs),
            peak_reserved_base_mb=max(r["base"]["peak_reserved_mb"] for r in recs),
            peak_reserved_cache_mb=max(r["cache"]["peak_reserved_mb"] for r in recs))

    cot_walls = sorted(item["wall"] for item in cot_generation)
    cot_generation_perf = dict(
        n_groups=len(cot_generation),
        mean=round(statistics.mean(cot_walls), 2),
        p50=round(pct(cot_walls, .5), 2),
        p90=round(pct(cot_walls, .9), 2),
        p95=round(pct(cot_walls, .95), 2),
        max=round(cot_walls[-1], 2),
    )

    report = dict(
        n_groups=n_grp, n_cots=len(records), seed=SEED,
        cand_parity_rate=round(sum(1 for r in records if r["cand_parity"]) / len(records), 4),
        sid_set_parity_rate=round(sum(1 for r in records if r["sid_set_parity"]) / len(records), 4),
        raw_reward_parity_rate=round(1 - n_mismatch / len(records), 4),
        reward_mismatch=n_mismatch,
        reward_mean_bias=round(mean_bias, 4),
        reward_mae=round(mae, 4),
        reward_max_abs_diff=round(max_diff, 4),
        agg=agg,
        argmax_agree=round(argmax_agree / n_grp, 4),
        argmin_agree=round(argmin_agree / n_grp, 4),
        full_rank_agree=round(rank_agree / n_grp, 4),
        top2_agree=round(top2_agree / n_grp, 4),
        adv_sign_agree=round(sign_agree_rate, 6) if sign_agree_rate is not None else None,
        adv_exact_agree=round(adv_exact_total / sign_agree_denom, 6),
        adv_close_agree=round(adv_close_total / sign_agree_denom, 6),
        adv_cosine=adv_cos,
        adv_mean_abs_diff=round(float(adv_abs_diffs.mean()), 5),
        adv_max_diff=round(float(adv_abs_diffs.max()), 5),
        zero_std=zero_flip,
        sign_flip_pos_neg=sign_flip_pos_neg,
        sign_flip_neg_pos=sign_flip_neg_pos,
        pearson_surrogate=round(pearson, 6) if pearson is not None else None,
        learning_signal_distortion=round(distortion, 4),
        determinism=dict(
            base_det=sum(1 for d in det if d["base_det"]),
            base_sids_det=sum(1 for d in det if d["base_sids_det"]),
            base_reward_det=sum(1 for d in det if d["base_reward_det"]),
            cache_det=sum(1 for d in det if d["cache_det"]),
            cache_sids_det=sum(1 for d in det if d["cache_sids_det"]),
            cache_reward_det=sum(1 for d in det if d["cache_reward_det"]), n=len(det)),
        cot_generation=cot_generation_perf,
        perf=perf_out,
    )
    details = dict(
        seed=SEED,
        groups=groups,
        cot_generation=cot_generation,
        records=records,
        group_metrics=group_details,
        determinism=det,
        cot_samples_path=COTS_PATH,
        records_jsonl_path=RECORDS_PATH,
    )
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    print("=== REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
