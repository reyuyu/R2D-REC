# -*- coding: utf-8 -*-
"""Synchronized long-context timing supplement for the Prefix-KV audit."""
import collections
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import encode_prompt, generate_batch, load_model
from grpo_sid import final_sid, parse_sid, think_reward
from transformers import StoppingCriteria, StoppingCriteriaList


DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
OUTPUT = "/data/GRPO/logs/prefix_cache_long_sync.json"
SEED = 20260816
N_LONG = 4


def sync_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(start):
    torch.cuda.synchronize()
    return time.perf_counter() - start


def canonicalize(ids, pad_id):
    ids = list(ids)
    while ids and ids[-1] == pad_id:
        ids.pop()
    return ids


def main():
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for row in rows:
        by_group[row["recommendation_group_id"]][row["route"]] = row

    model, tokenizer, _ = load_model("cuda:0")
    model.eval()
    pad_id = tokenizer.pad_token_id
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    class StopAtThinkEnd(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids[:, -1] == think_end_id

    think_rows = [row for row in rows if row["route"] == "think"]
    for row in think_rows:
        row["_prompt_ids"] = encode_prompt(tokenizer, row["prompt"])
    think_rows.sort(key=lambda row: len(row["_prompt_ids"]), reverse=True)
    selected = think_rows[:N_LONG]

    torch.manual_seed(SEED)
    contexts = []
    for row in selected:
        prompt_ids = row["_prompt_ids"]
        start = sync_timer()
        with torch.inference_mode():
            output = model.generate(
                input_ids=torch.tensor([prompt_ids], device=model.device, dtype=torch.long),
                attention_mask=torch.ones(1, len(prompt_ids), device=model.device, dtype=torch.long),
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=pad_id,
                stopping_criteria=StoppingCriteriaList([StopAtThinkEnd()]),
            )[0]
        generation_wall = elapsed(start)
        cot_ids = output[len(prompt_ids):].tolist()
        try:
            end = cot_ids.index(think_end_id)
            cot_ids = cot_ids[:end + 1]
        except ValueError:
            pass
        contexts.append(dict(
            group=row["recommendation_group_id"],
            prompt_ids=prompt_ids,
            cot_ids=cot_ids,
            context=prompt_ids + cot_ids,
            generation_wall=round(generation_wall, 4),
        ))

    def baseline_beam(context):
        torch.cuda.reset_peak_memory_stats()
        start = sync_timer()
        _, ids = generate_batch(
            model,
            tokenizer,
            [context],
            max_new_tokens=128,
            num_beams=32,
            num_return_sequences=32,
            return_ids=True,
        )
        wall = elapsed(start)
        return (
            [canonicalize(item, pad_id) for item in ids],
            wall,
            torch.cuda.max_memory_allocated() // (1024 * 1024),
            torch.cuda.max_memory_reserved() // (1024 * 1024),
        )

    def cache_beam(context):
        prefix = context[:-1]
        device = model.device
        torch.cuda.reset_peak_memory_stats()
        start = sync_timer()
        with torch.inference_mode():
            result = model(
                input_ids=torch.tensor([prefix], device=device, dtype=torch.long),
                attention_mask=torch.ones(1, len(prefix), device=device, dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
        prefill = elapsed(start)
        cache = result.past_key_values
        assert cache.get_seq_length() == len(prefix)

        start = sync_timer()
        cache.batch_repeat_interleave(32)
        repeat = elapsed(start)

        start = sync_timer()
        with torch.inference_mode():
            output = model.generate(
                input_ids=torch.tensor([context], device=device, dtype=torch.long),
                attention_mask=torch.ones(1, len(context), device=device, dtype=torch.long),
                past_key_values=cache,
                max_new_tokens=128,
                num_beams=32,
                num_return_sequences=32,
                pad_token_id=pad_id,
            )
        beam = elapsed(start)

        start = time.perf_counter()
        ids = [canonicalize(output[i, len(context):].tolist(), pad_id) for i in range(32)]
        [tokenizer.decode(item, skip_special_tokens=False) for item in ids]
        post = time.perf_counter() - start
        return (
            ids,
            prefill,
            repeat,
            beam,
            post,
            torch.cuda.max_memory_allocated() // (1024 * 1024),
            torch.cuda.max_memory_reserved() // (1024 * 1024),
        )

    def score(ids, gold):
        sids = [final_sid(tokenizer.decode(item, skip_special_tokens=False)) for item in ids]
        invalid = sum(sid is None for sid in sids)
        reward, exact, ab, a_count = think_reward(sids, gold)
        return dict(sids=sids, invalid=invalid, exact=exact, ab=ab, a=a_count, reward=reward)

    records = []
    for index, sample in enumerate(contexts):
        gold = {
            parsed
            for sid in by_group[sample["group"]]["think"]["all_gold_sids"]
            if (parsed := parse_sid(sid))
        }
        baseline_ids, baseline_wall, baseline_peak, baseline_reserved = baseline_beam(sample["context"])
        cache_ids, prefill, repeat, beam, post, cache_peak, cache_reserved = cache_beam(sample["context"])
        baseline_score = score(baseline_ids, gold)
        cache_score = score(cache_ids, gold)
        total = prefill + repeat + beam + post
        records.append(dict(
            index=index,
            group=sample["group"],
            prompt_len=len(sample["prompt_ids"]),
            cot_len=len(sample["cot_ids"]),
            context_len=len(sample["context"]),
            generation_wall=sample["generation_wall"],
            baseline=dict(wall=round(baseline_wall, 4), peak_mb=baseline_peak,
                          peak_reserved_mb=baseline_reserved, **baseline_score),
            cache=dict(prefill=round(prefill, 4), repeat=round(repeat, 4),
                       beam=round(beam, 4), post=round(post, 4), total=round(total, 4),
                       peak_mb=cache_peak, peak_reserved_mb=cache_reserved, **cache_score),
            candidate_parity=baseline_ids == cache_ids,
            sid_set_parity=set(baseline_score["sids"]) == set(cache_score["sids"]),
            reward_parity=baseline_score["reward"] == cache_score["reward"],
        ))
        print(f"long {index + 1}/{N_LONG} context={len(sample['context'])}", flush=True)

    baseline_walls = [record["baseline"]["wall"] for record in records]
    cache_walls = [record["cache"]["total"] for record in records]
    report = dict(
        n=len(records),
        seed=SEED,
        context_lengths=[record["context_len"] for record in records],
        baseline_mean=statistics.mean(baseline_walls),
        cache_mean=statistics.mean(cache_walls),
        speedup=statistics.mean(baseline_walls) / statistics.mean(cache_walls),
        baseline_peak_mb=max(record["baseline"]["peak_mb"] for record in records),
        cache_peak_mb=max(record["cache"]["peak_mb"] for record in records),
        reward_parity=sum(record["reward_parity"] for record in records),
        records=records,
    )
    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=1)
    print("=== REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
