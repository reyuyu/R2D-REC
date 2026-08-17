# -*- coding: utf-8 -*-
"""Paired real-model benchmark for call-local Beam prompt ID caching."""
import hashlib
import json
import statistics
import time

import torch

from grpo_model import encode_prompt, generate_batch, load_model
from grpo_sid import final_sid, parse_sid, think_reward

SAMPLES = "/data/GRPO/logs/beam_direct_ids_samples.json"
RESULT = "/data/GRPO/logs/beam_prompt_cache_benchmark.json"
PREPROCESS_REPEATS = 100
PAIRED_REPEATS = 2


def old_cot_ids(tokenizer, completion_ids):
    cot = tokenizer.decode(completion_ids, skip_special_tokens=False)
    close = cot.find("</think>")
    cot_trim = cot[: close + len("</think>")] if close >= 0 else cot
    return tokenizer.encode(cot_trim, add_special_tokens=False)


def direct_cot_ids(completion_ids, think_token_id):
    try:
        return completion_ids[: completion_ids.index(think_token_id) + 1]
    except ValueError:
        return completion_ids


def build_contexts(tokenizer, samples, cached):
    prompt_ids_cache = {}
    contexts = []
    metadata = []
    for sample in samples:
        prompt = sample["prompt"]
        gold_set = {
            parsed for sid in sample["all_gold_sids"]
            if (parsed := parse_sid(sid)) is not None
        }
        for completion_ids in sample["completion_ids"]:
            if cached:
                prompt_ids = prompt_ids_cache.get(prompt)
                if prompt_ids is None:
                    prompt_ids = encode_prompt(tokenizer, prompt)
                    prompt_ids_cache[prompt] = prompt_ids
            else:
                prompt_ids = encode_prompt(tokenizer, prompt)
            contexts.append(prompt_ids + old_cot_ids(tokenizer, completion_ids))
            metadata.append(gold_set)
    return contexts, metadata


def hash_ids(nested_ids):
    digest = hashlib.sha256()
    for sequence in nested_ids:
        digest.update(",".join(str(token_id) for token_id in sequence).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def evaluate_outputs(texts, metadata):
    rewards = []
    exact = []
    ab = []
    a = []
    invalid = []
    for index, gold_set in enumerate(metadata):
        group_texts = texts[index * 32:(index + 1) * 32]
        sids = [final_sid(text) for text in group_texts]
        reward, n_exact, n_ab, n_a = think_reward(sids, gold_set)
        rewards.append(reward)
        exact.append(n_exact)
        ab.append(n_ab)
        a.append(n_a)
        invalid.append(sum(sid is None for sid in sids))
    return {
        "reward": rewards,
        "exact": exact,
        "ab": ab,
        "a": a,
        "invalid": invalid,
    }


def run_reward(model, tokenizer, samples, cached):
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    preprocess_start = time.perf_counter()
    contexts, metadata = build_contexts(tokenizer, samples, cached)
    preprocess_sec = time.perf_counter() - preprocess_start

    all_texts = []
    all_ids = []
    torch.cuda.synchronize()
    beam_start = time.perf_counter()
    for context in contexts:
        texts, ids = generate_batch(
            model, tokenizer, [context], max_new_tokens=128,
            num_beams=32, num_return_sequences=32, return_ids=True,
        )
        all_texts.extend(texts)
        all_ids.extend(ids)
    torch.cuda.synchronize()
    beam_sec = time.perf_counter() - beam_start
    metrics = evaluate_outputs(all_texts, metadata)
    torch.cuda.synchronize()
    total_sec = time.perf_counter() - total_start
    return {
        "preprocess_sec": preprocess_sec,
        "beam_sec": beam_sec,
        "total_sec": total_sec,
        "output_ids": all_ids,
        "output_hash": hash_ids(all_ids),
        "metrics": metrics,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
    }


def summarize(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def main():
    data = json.load(open(SAMPLES, encoding="utf-8"))
    samples = data["samples"]
    torch.cuda.set_device(0)
    model, tokenizer, _ = load_model("cuda:0")
    model.eval()
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) != 1:
        raise RuntimeError(f"</think> must be one token, got {think_tokens}")
    think_token_id = think_tokens[0]

    old_contexts, _ = build_contexts(tokenizer, samples, cached=False)
    cached_contexts, _ = build_contexts(tokenizer, samples, cached=True)
    cache_context_equal = [old == new for old, new in zip(old_contexts, cached_contexts)]

    direct_closed_equal = []
    direct_noclose_equal = []
    direct_differences = []
    for group_index, sample in enumerate(samples):
        prompt_ids = encode_prompt(tokenizer, sample["prompt"])
        for cot_index, completion_ids in enumerate(sample["completion_ids"]):
            old_ids = old_cot_ids(tokenizer, completion_ids)
            direct_ids = direct_cot_ids(completion_ids, think_token_id)
            equal = old_ids == direct_ids
            direct_closed_equal.append(equal)
            if not equal:
                first = next(
                    (i for i, (left, right) in enumerate(zip(old_ids, direct_ids)) if left != right),
                    min(len(old_ids), len(direct_ids)),
                )
                direct_differences.append({
                    "group": group_index,
                    "cot": cot_index,
                    "old_length": len(old_ids),
                    "direct_length": len(direct_ids),
                    "first_diff": first,
                    "decoded_text_equal": tokenizer.decode(old_ids, skip_special_tokens=False)
                    == tokenizer.decode(direct_ids, skip_special_tokens=False),
                })
        no_close = sample["completion_ids"][0]
        no_close = no_close[: no_close.index(think_token_id)]
        direct_noclose_equal.append(
            old_cot_ids(tokenizer, no_close) == direct_cot_ids(no_close, think_token_id)
        )

    # CPU preprocessing microbenchmark; alternate order to limit drift.
    preprocess_times = {"baseline": [], "cached": []}
    for repeat in range(PREPROCESS_REPEATS):
        order = (("baseline", False), ("cached", True))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, cached in order:
            started = time.perf_counter()
            contexts, _ = build_contexts(tokenizer, samples, cached)
            preprocess_times[name].append(time.perf_counter() - started)
            if contexts != old_contexts:
                raise RuntimeError(f"{name} preprocessing changed Beam contexts")

    # Warm Beam32 with one production-shape context, excluded from timings.
    generate_batch(
        model, tokenizer, [old_contexts[0]], max_new_tokens=128,
        num_beams=32, num_return_sequences=32, return_ids=True,
    )
    torch.cuda.synchronize()

    runs = {"baseline": [], "cached": []}
    parity = []
    for repeat in range(PAIRED_REPEATS):
        order = (("baseline", False), ("cached", True))
        if repeat % 2:
            order = tuple(reversed(order))
        pair = {}
        for name, cached in order:
            torch.cuda.reset_peak_memory_stats()
            pair[name] = run_reward(model, tokenizer, samples, cached)
            runs[name].append(pair[name])
            print(
                f"pair {repeat + 1}/{PAIRED_REPEATS} {name}: "
                f"prep={pair[name]['preprocess_sec']:.4f}s "
                f"beam={pair[name]['beam_sec']:.3f}s total={pair[name]['total_sec']:.3f}s",
                flush=True,
            )
        parity.append({
            "output_ids_equal": pair["baseline"]["output_ids"] == pair["cached"]["output_ids"],
            "sid_reward_metrics_equal": pair["baseline"]["metrics"] == pair["cached"]["metrics"],
            "baseline_output_hash": pair["baseline"]["output_hash"],
            "cached_output_hash": pair["cached"]["output_hash"],
        })

    result = {
        "groups": len(samples),
        "cots": sum(len(sample["completion_ids"]) for sample in samples),
        "think_token_ids": think_tokens,
        "direct_id_candidate": {
            "adopted": False,
            "closed_input_ids_equal": [sum(direct_closed_equal), len(direct_closed_equal)],
            "synthetic_noclose_input_ids_equal": [sum(direct_noclose_equal), len(direct_noclose_equal)],
            "differences": direct_differences,
        },
        "prompt_cache_candidate": {
            "context_ids_equal": [sum(cache_context_equal), len(cache_context_equal)],
            "preprocess_microbenchmark_repeats": PREPROCESS_REPEATS,
            "preprocess_microbenchmark": {
                name: summarize(values) for name, values in preprocess_times.items()
            },
            "paired_repeats": PAIRED_REPEATS,
            "parity": parity,
            "timing": {
                name: {
                    metric: summarize([run[metric] for run in mode_runs])
                    for metric in ("preprocess_sec", "beam_sec", "total_sec")
                }
                for name, mode_runs in runs.items()
            },
            "metrics": {
                name: [run["metrics"] for run in mode_runs] for name, mode_runs in runs.items()
            },
            "memory": {
                name: [{
                    "peak_allocated_mb": run["peak_allocated_mb"],
                    "peak_reserved_mb": run["peak_reserved_mb"],
                } for run in mode_runs]
                for name, mode_runs in runs.items()
            },
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    with open(RESULT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if not all(cache_context_equal):
        raise SystemExit("prompt cache changed Beam input IDs")
    if not all(item["output_ids_equal"] and item["sid_reward_metrics_equal"] for item in parity):
        raise SystemExit("prompt cache Beam parity failed")
    print("PROMPT CACHE PARITY PASSED", flush=True)


if __name__ == "__main__":
    main()
