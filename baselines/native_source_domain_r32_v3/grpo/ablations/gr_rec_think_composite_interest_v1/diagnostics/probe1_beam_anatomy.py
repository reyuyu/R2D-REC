"""Inference-only, four-GPU anatomy for Composite Interest Probe1 Beam search."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

GRPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))
from grpo_model import ADAPTER, BASE, encode_prompt, generate_batch, load_model
from grpo_sid import think_reward

GROUP_ID = "662e21149595c33240d7ec0baed2282d687f71e1a0983252f28edc9c18855cbc"
OLD_PROBE = Path("/data/GRPO/runs/GR-REC-THINK-COMPOSITE-INTEREST-V1-FORMAL716-20260823/probes.jsonl")
FIXED_PROBE = Path("/data/GRPO/runs/GR-REC-THINK-COMPOSITE-INTEREST-V1-ABC3-FORMAL716-20260823/probes.jsonl")
DEFAULT_PARTS = Path("/data/GRPO/results/probe1_beam_anatomy_20260823_parts")
DEFAULT_OUTPUT = GRPO_ROOT / "results/gr_rec_think_composite_interest_v1_probe1_beam_anatomy_20260823.json"
HISTORICAL_EXACT = (("video", 2406, 3727, 5563), ("video", 5739, 2965, 670))
DOMAINS = {"video": "<|video_begin|>", "prod": "<|prod_begin|>", "ad": "<|ad_begin|>", "living": "<|living_begin|>"}
ABC = tuple(re.compile(rf"<s_{stage}_(\d+)>").fullmatch for stage in "abc")
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def load_probe(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("step") == 0 and row.get("group_id") == GROUP_ID:
                return row
    raise RuntimeError(f"Probe1 step0 missing: {path}")


def completion_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def closed_cot(text):
    end = text.find("</think>")
    if end < 0:
        raise RuntimeError("Probe CoT is not closed")
    return text[: end + len("</think>")]


def tokens(tokenizer, ids):
    value = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return [value] if isinstance(value, str) else list(value)


def parse_gold(text):
    match = SID_RE.fullmatch(text)
    if not match:
        raise RuntimeError(f"Invalid gold SID: {text}")
    return (match.group(1), *(int(match.group(i)) for i in range(2, 5)))


def parse_abc3(tokenizer, ids):
    ts = tokens(tokenizer, ids)
    matches = [pattern(token) for pattern, token in zip(ABC, ts)]
    if len(ids) != 3 or len(ts) != 3 or any(match is None for match in matches):
        return None
    return ("video", *(int(match.group(1)) for match in matches))


def parse_dabc4(tokenizer, ids):
    ts = tokens(tokenizer, ids)
    domain = {value: key for key, value in DOMAINS.items()}.get(ts[0]) if ts else None
    matches = [pattern(token) for pattern, token in zip(ABC, ts[1:])]
    if len(ids) != 4 or len(ts) != 4 or domain is None or any(match is None for match in matches):
        return None
    return (domain, *(int(match.group(1)) for match in matches))


def relation(sid, gold):
    if sid is None:
        return "INVALID"
    if sid in gold:
        return "EXACT"
    if any(sid[:3] == item[:3] for item in gold):
        return "AB"
    if any(sid[:2] == item[:2] for item in gold):
        return "A"
    return "VALID_NO_HIT"


def summary(sids, gold):
    reward, exact, ab, a = think_reward(sids, gold)
    return {"beam_raw": float(reward), "exact": int(exact), "ab": int(ab), "a": int(a), "invalid": sum(x is None for x in sids)}


def hf_beam(model, tokenizer, context, width, count, parser, gold):
    texts, ids = generate_batch(model, tokenizer, [context], min_new_tokens=count,
        max_new_tokens=count, do_sample=False, num_beams=width,
        num_return_sequences=width, return_ids=True)
    sids = [parser(tokenizer, row) for row in ids]
    beams = [{"beam_index": i, "generated_text": text, "generated_token_ids": row,
        "generated_tokens": tokens(tokenizer, row), "parsed_sid": list(sid) if sid else None,
        "relation_to_gold": relation(sid, gold)}
        for i, (text, row, sid) in enumerate(zip(texts, ids, sids))]
    return {"width": width, "implementation": "hf_generate",
        "summary": summary(sids, gold), "beams": beams}, ids


def token_rank(logits, token_id):
    return int(torch.count_nonzero(logits > logits[token_id]).item()) + 1


def logprob_rank(logits, token_id):
    logits = logits.float()
    return float(torch.log_softmax(logits, -1)[token_id].item()), token_rank(logits, token_id)


def prefill(model, ids):
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
    return output.logits[0, -1].float(), output.past_key_values


def decode_one(model, token_ids, cache, length):
    if token_ids.ndim == 1:
        token_ids = token_ids[:, None]
    attention = torch.ones((token_ids.shape[0], length), dtype=torch.long, device=model.device)
    return model(input_ids=token_ids, attention_mask=attention, past_key_values=cache, use_cache=True)


def score_sequences(model, tokenizer, context, sequences):
    """Teacher-force ABC while branching a mutable DynamicCache via crop."""
    a_logits, cache = prefill(model, context)
    base = len(context)
    by_a = {}
    for index, ids in enumerate(sequences):
        if len(ids) != 3:
            raise ValueError("ABC scoring requires three tokens")
        by_a.setdefault(ids[0], []).append((index, ids))
    result = [None] * len(sequences)
    for a_id, a_rows in by_a.items():
        log_a, rank_a = logprob_rank(a_logits, a_id)
        out_a = decode_one(model, torch.tensor([a_id], device=model.device), cache, base + 1)
        b_logits = out_a.logits[0, -1].float()
        by_b = {}
        for index, ids in a_rows:
            by_b.setdefault(ids[1], []).append((index, ids))
        for b_id, b_rows in by_b.items():
            log_b, rank_b = logprob_rank(b_logits, b_id)
            out_b = decode_one(model, torch.tensor([b_id], device=model.device), cache, base + 2)
            c_logits = out_b.logits[0, -1].float()
            for index, ids in b_rows:
                log_c, rank_c = logprob_rank(c_logits, ids[2])
                result[index] = {"token_ids": ids, "tokens": tokens(tokenizer, ids),
                    "a_logprob": log_a, "a_rank": rank_a, "b_logprob": log_b,
                    "b_rank": rank_b, "c_logprob": log_c, "c_rank": rank_c,
                    "abc_conditional_logprob": log_a + log_b + log_c}
            cache.crop(base + 1)
        cache.crop(base)
    return result


def eos_ids(model):
    value = model.generation_config.eos_token_id
    if value is None:
        return []
    return [value] if isinstance(value, int) else list(value)


def mask_eos(values, ids):
    if ids:
        values[..., ids] = -math.inf
    return values


def frontier_rows(tokenizer, ids, scores):
    return [{"rank": i + 1, "token_ids": row.tolist(), "tokens": tokens(tokenizer, row.tolist()),
        "cumulative_logprob": float(score.item())}
        for i, (row, score) in enumerate(zip(ids.cpu(), scores.cpu()))]


def manual_beam32(model, tokenizer, context):
    width = 32
    logits, cache = prefill(model, context)
    eos = eos_ids(model)
    lp_a = mask_eos(torch.log_softmax(logits, -1), eos)
    score_a, id_a = torch.topk(lp_a, width, sorted=True)
    seq_a = id_a[:, None]
    cache.batch_repeat_interleave(width)
    out_b = decode_one(model, id_a, cache, len(context) + 1)
    lp_b = mask_eos(torch.log_softmax(out_b.logits[:, -1].float(), -1), eos)
    flat_b = (score_a[:, None] + lp_b).flatten()
    score_b, flat_id_b = torch.topk(flat_b, width, sorted=True)
    vocab = lp_b.shape[-1]
    parent_b = torch.div(flat_id_b, vocab, rounding_mode="floor")
    id_b = flat_id_b.remainder(vocab)
    seq_b = torch.cat((seq_a[parent_b], id_b[:, None]), 1)
    cache.batch_select_indices(parent_b)
    out_c = decode_one(model, id_b, cache, len(context) + 2)
    lp_c = mask_eos(torch.log_softmax(out_c.logits[:, -1].float(), -1), eos)
    flat_c = (score_b[:, None] + lp_c).flatten()
    score_c, flat_id_c = torch.topk(flat_c, width, sorted=True)
    parent_c = torch.div(flat_id_c, vocab, rounding_mode="floor")
    id_c = flat_id_c.remainder(vocab)
    seq_c = torch.cat((seq_b[parent_c], id_c[:, None]), 1)
    return {"step_a": frontier_rows(tokenizer, seq_a, score_a),
        "step_b": frontier_rows(tokenizer, seq_b, score_b),
        "step_c": frontier_rows(tokenizer, seq_c, score_c)}


def manual_beam_sweep(model, tokenizer, context, width, gold):
    """Exact three-step Beam-K with sequential cache branches for bounded memory."""
    logits, cache = prefill(model, context)
    base, eos = len(context), eos_ids(model)
    lp_a = mask_eos(torch.log_softmax(logits, -1), eos)
    score_a, id_a = torch.topk(lp_a, width, sorted=True)
    step_a = [(float(score), [int(token)]) for score, token in zip(score_a, id_a)]
    b_candidates = []
    for a_score, (a_id,) in step_a:
        out = decode_one(model, torch.tensor([a_id], device=model.device), cache, base + 1)
        lp = mask_eos(torch.log_softmax(out.logits[0, -1].float(), -1), eos)
        scores, ids = torch.topk(lp + a_score, width, sorted=True)
        b_candidates.extend((float(score), [a_id, int(b_id)]) for score, b_id in zip(scores, ids))
        cache.crop(base)
    step_b = sorted(b_candidates, key=lambda item: item[0], reverse=True)[:width]
    c_candidates = []
    by_a = {}
    for score, ids in step_b:
        by_a.setdefault(ids[0], []).append((score, ids[1]))
    for a_id, rows in by_a.items():
        decode_one(model, torch.tensor([a_id], device=model.device), cache, base + 1)
        for ab_score, b_id in rows:
            out = decode_one(model, torch.tensor([b_id], device=model.device), cache, base + 2)
            lp = mask_eos(torch.log_softmax(out.logits[0, -1].float(), -1), eos)
            scores, ids = torch.topk(lp + ab_score, width, sorted=True)
            c_candidates.extend((float(score), [a_id, b_id, int(c_id)]) for score, c_id in zip(scores, ids))
            cache.crop(base + 1)
        cache.crop(base)
    step_c = sorted(c_candidates, key=lambda item: item[0], reverse=True)[:width]
    generated = [ids for _, ids in step_c]
    sids = [parse_abc3(tokenizer, ids) for ids in generated]
    beams = [{"beam_index": i, "generated_text": tokenizer.decode(ids, skip_special_tokens=False),
        "generated_token_ids": ids, "generated_tokens": tokens(tokenizer, ids),
        "parsed_sid": list(sid) if sid else None, "relation_to_gold": relation(sid, gold),
        "cumulative_logprob": score} for i, ((score, ids), sid) in enumerate(zip(step_c, sids))]
    return {"width": width, "implementation": "manual_frontier_sequential_kv",
        "summary": summary(sids, gold), "beams": beams}, generated


def pruned_at(ids, manual):
    sets = [{tuple(row["token_ids"]) for row in manual[key]} for key in ("step_a", "step_b", "step_c")]
    if tuple(ids[:1]) not in sets[0]: return "PRUNED_AT_A"
    if tuple(ids[:2]) not in sets[1]: return "PRUNED_AT_AB"
    if tuple(ids) not in sets[2]: return "PRUNED_AT_C"
    return "SURVIVED_TO_FINAL"


def overlap(left, right):
    return sum((Counter(map(tuple, left)) & Counter(map(tuple, right))).values())


def prefill_kv(model, free_context, domain_ids, gold_a):
    if len(domain_ids) != 1:
        raise RuntimeError(f"Video prefix must be one token: {domain_ids}")
    direct, _ = prefill(model, free_context + domain_ids)
    _, cache = prefill(model, free_context)
    out = decode_one(model, torch.tensor(domain_ids, device=model.device), cache, len(free_context) + 1)
    cached = out.logits[0, -1].float()
    diff = (direct - cached).abs()
    direct_top = torch.topk(direct, 32).indices.tolist()
    cached_top = torch.topk(cached, 32).indices.tolist()
    return {"max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
        "top32_overlap": len(set(direct_top) & set(cached_top)),
        "gold_a_rank_diff": {str(x): token_rank(direct, x) - token_rank(cached, x) for x in sorted(set(gold_a))}}


def gold_ids(tokenizer, sid):
    values = [tokenizer.convert_tokens_to_ids(f"<s_{stage}_{sid[i]}>") for i, stage in enumerate("abc", 1)]
    if any(value is None or value == tokenizer.unk_token_id for value in values):
        raise RuntimeError(f"Unknown gold token: {sid}")
    return [int(value) for value in values]


def run_rank(rank, parts_dir):
    old, fixed = load_probe(OLD_PROBE), load_probe(FIXED_PROBE)
    old_c, fixed_c = old["think"]["candidates"], fixed["think"]["candidates"]
    sha_pairs = [{"candidate_id": i, "old": a["completion_sha256"], "fixed": b["completion_sha256"],
        "match": a["completion_sha256"] == b["completion_sha256"]} for i, (a, b) in enumerate(zip(old_c, fixed_c))]
    if len(sha_pairs) != 4 or sum(x["match"] for x in sha_pairs) != 4:
        raise RuntimeError("COT_SHA_MISMATCH")
    torch.cuda.set_device(rank)
    model, tokenizer, _ = load_model(device=f"cuda:{rank}")
    candidate = fixed_c[rank]
    full_retokenized_ids = tokenizer.encode(candidate["completion"], add_special_tokens=False)
    full_retokenized_sha = completion_sha(full_retokenized_ids)
    cot_ids = tokenizer.encode(closed_cot(candidate["completion"]), add_special_tokens=False)
    closed_cot_sha = completion_sha(cot_ids)
    prompt_ids = encode_prompt(tokenizer, fixed["think_prompt"])
    free = prompt_ids + cot_ids
    domain_ids = tokenizer.encode(DOMAINS["video"], add_special_tokens=False)
    context = free + domain_ids
    gold = [parse_gold(x) for x in fixed["gold_sids"]]
    gold_set = set(gold)
    g_ids = [gold_ids(tokenizer, sid) for sid in gold]
    started = time.perf_counter()
    with torch.inference_mode():
        dabc, _ = hf_beam(model, tokenizer, free, 32, 4, parse_dabc4, gold_set)
        fixed32, ids32 = hf_beam(model, tokenizer, context, 32, 3, parse_abc3, gold_set)
        fixed64, ids64 = manual_beam_sweep(model, tokenizer, context, 64, gold_set)
        fixed128, ids128 = manual_beam_sweep(model, tokenizer, context, 128, gold_set)
        torch.cuda.empty_cache()
        scored = score_sequences(model, tokenizer, context, g_ids + ids32)
        scored_gold, scored_fixed = scored[:len(g_ids)], scored[len(g_ids):]
        rescored = sorted(scored_fixed, key=lambda x: x["abc_conditional_logprob"], reverse=True)
        cutoff = rescored[31]["abc_conditional_logprob"]
        manual = manual_beam32(model, tokenizer, context)
        parity_count = overlap([x["token_ids"] for x in manual["step_c"]], ids32)
        parity_ok = parity_count == 32
        parity = prefill_kv(model, free, domain_ids, [x[0] for x in g_ids])
    width_sets = {32: set(map(tuple, ids32)), 64: set(map(tuple, ids64)), 128: set(map(tuple, ids128))}
    gold_rows = []
    for sid, ids, score in zip(gold, g_ids, scored_gold):
        first = next((width for width in (32, 64, 128) if tuple(ids) in width_sets[width]), "NOT_FOUND")
        found32 = tuple(ids) in width_sets[32]
        flag = None if found32 else ("BEAM_SEARCH_PRUNING_ERROR" if score["abc_conditional_logprob"] > cutoff else "FULL_SEQUENCE_SCORE_BELOW_TOP32")
        gold_rows.append({"gold_sid": list(sid), **score, "fixed_beam32_cutoff_logprob": cutoff,
            "diagnostic_flag": flag, "pruned_at": pruned_at(ids, manual) if parity_ok else "NOT_APPLICABLE",
            "abc3_32_found": found32, "abc3_64_found": tuple(ids) in width_sets[64],
            "abc3_128_found": tuple(ids) in width_sets[128], "first_found_beam_width": first})
    for row, score in zip(fixed32["beams"], scored_fixed):
        row["teacher_forced_abc_logprob"] = score["abc_conditional_logprob"]
    historical = [{"sid": list(sid),
        "dabc4_found": any(x["parsed_sid"] == list(sid) for x in dabc["beams"]),
        "abc3_32_found": tuple(gold_ids(tokenizer, sid)) in width_sets[32],
        "abc3_64_found": tuple(gold_ids(tokenizer, sid)) in width_sets[64],
        "abc3_128_found": tuple(gold_ids(tokenizer, sid)) in width_sets[128]} for sid in HISTORICAL_EXACT]
    result = {"rank": rank, "device": f"cuda:{rank}", "model_parent": {"base": BASE, "adapter": ADAPTER,
        "fresh_original_bata": True}, "candidate_id": rank, "cot_sha_pairs": sha_pairs,
        "completion_sha256": candidate["completion_sha256"],
        "full_text_retokenized_sha256": full_retokenized_sha,
        "full_text_retokenized_sha_match": full_retokenized_sha == candidate["completion_sha256"],
        "production_closed_cot_sha256": closed_cot_sha,
        "prompt_token_count": len(prompt_ids),
        "cot_token_count": len(cot_ids), "strict_dabc4": dabc, "fixed_abc3_beam32": fixed32,
        "fixed_abc3_beam64": fixed64, "fixed_abc3_beam128": fixed128,
        "fixed_beam32_rescored_order": rescored, "fixed_beam32_cutoff_logprob": cutoff,
        "gold_teacher_forced": gold_rows, "manual_beam32": manual,
        "manual_beam_parity_count": parity_count, "manual_beam_parity": "32/32" if parity_ok else "FAIL",
        "prefill_vs_kv": parity, "historical_exact": historical,
        "elapsed_sec": time.perf_counter() - started, "training_started": False, "optimizer_created": False}
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"rank{rank}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANATOMY_RANK_DONE rank={rank} dabc={dabc['summary']} fixed={fixed32['summary']} parity={result['manual_beam_parity']} elapsed={result['elapsed_sec']:.1f}s", flush=True)


def aggregate(parts, key):
    rows = [x[key]["summary"] for x in parts]
    return {"beam_raw_mean": sum(x["beam_raw"] for x in rows) / 4,
        **{field: sum(x[field] for x in rows) for field in ("exact", "ab", "a", "invalid")}}


def merge(parts_dir, output):
    parts = [json.loads((parts_dir / f"rank{i}.json").read_text(encoding="utf-8")) for i in range(4)]
    strict, fixed = aggregate(parts, "strict_dabc4"), aggregate(parts, "fixed_abc3_beam32")
    expected = {"beam_raw_mean": 1.125, "exact": 0, "ab": 1, "a": 8, "invalid": 0}
    reproduced = all(math.isclose(fixed[k], v, abs_tol=1e-9) if isinstance(v, float) else fixed[k] == v for k, v in expected.items())
    historical = []
    for sid in HISTORICAL_EXACT:
        rows = [next(x for x in part["historical_exact"] if tuple(x["sid"]) == sid) for part in parts]
        historical.append({"sid": list(sid), **{field: any(x[field] for x in rows)
            for field in ("dabc4_found", "abc3_32_found", "abc3_64_found", "abc3_128_found")}, "per_candidate": rows})
    all_gold = [row for part in parts for row in part["gold_teacher_forced"]]
    flags = sorted({x["diagnostic_flag"] for x in all_gold if x["diagnostic_flag"]})
    max_diff = max(x["prefill_vs_kv"]["max_abs_logit_diff"] for x in parts)
    min_overlap = min(x["prefill_vs_kv"]["top32_overlap"] for x in parts)
    causes = []
    if not any(x["dabc4_found"] for x in historical): causes.append("OLD_PARSE_ARTIFACT")
    if "BEAM_SEARCH_PRUNING_ERROR" in flags: causes.append("BEAM_FRONTIER_PRUNING")
    if "FULL_SEQUENCE_SCORE_BELOW_TOP32" in flags: causes.append("FULL_SEQUENCE_BELOW_TOP32")
    if max_diff > 0.05 or min_overlap < 30: causes.append("NUMERICAL_PREFILL_KV_DIFFERENCE")
    root = causes[0] if len(causes) == 1 else "MIXED"
    sha_match = sum(x["match"] for x in parts[0]["cot_sha_pairs"])
    result = {"type": "probe1_beam_search_anatomy", "generated_at_unix": time.time(),
        "anatomy_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=GRPO_ROOT, text=True).strip(),
        "model_parent": parts[0]["model_parent"], "recommendation_group_id": GROUP_ID,
        "cot_sha_match": f"{sha_match}/4", "strict_dabc4": strict, "fixed_abc3": fixed,
        "fixed_abc3_expected": expected, "fixed_abc3_reproduced": reproduced,
        "fixed_beam64_exact": sum(x["fixed_abc3_beam64"]["summary"]["exact"] for x in parts),
        "fixed_beam128_exact": sum(x["fixed_abc3_beam128"]["summary"]["exact"] for x in parts),
        "historical_exact": historical, "manual_beam_parity": "32/32" if all(x["manual_beam_parity"] == "32/32" for x in parts) else "FAIL",
        "prefill_vs_kv_summary": {"max_abs_logit_diff": max_diff,
            "mean_abs_logit_diff_mean": sum(x["prefill_vs_kv"]["mean_abs_logit_diff"] for x in parts) / 4,
            "top32_overlap_min": min_overlap, "per_candidate": [x["prefill_vs_kv"] for x in parts]},
        "diagnostic_flags": flags, "root_cause": root, "root_cause_components": causes,
        "parts": parts, "gpu_used": 4, "training_started": False, "production_code_changed": False}
    if sha_match != 4: raise RuntimeError("COT_SHA_MATCH is not 4/4")
    if not reproduced: raise RuntimeError(f"FIXED_ABC3_REPRODUCTION_FAILED: {fixed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("cot_sha_match", "strict_dabc4", "fixed_abc3", "fixed_beam64_exact", "fixed_beam128_exact", "manual_beam_parity", "prefill_vs_kv_summary", "root_cause", "root_cause_components")}, ensure_ascii=False, indent=2))
    print(f"ANATOMY_RESULT={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args.parts_dir, args.output)
        return
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    if world != 4 or rank not in range(4):
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    run_rank(rank, args.parts_dir)


if __name__ == "__main__":
    main()
