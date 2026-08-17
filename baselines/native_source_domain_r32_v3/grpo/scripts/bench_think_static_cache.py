# -*- coding: utf-8 -*-
"""Benchmark DynamicCache vs StaticCache on the production Think path.

This is a diagnostic script only. It does not modify trainer configuration.
"""
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from collections import Counter

import torch
import peft
import transformers
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from grpo_model import load_model, render_prompt
from grpo_sid import parse_sid
from run_grpo_trl_smoke import make_beam32_fn


DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
RESULT = "/data/GRPO/logs/think_static_cache.json"
SEED = 20260820
G = 4
MAX_COMPLETION = 2048
PAIRED_REPEATS = 2
TARGET_LENGTHS = {"short": 700, "medium": 1250, "long": 1850}
COMPILE_SEQUENCE = ["short", "short", "medium", "medium", "long", "long", "short"]


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


def generation_config(tokenizer, cache_implementation, max_new_tokens=MAX_COMPLETION):
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_p=0.95,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        cache_implementation=cache_implementation,
        stop_strings=None,
    )


def completion_sha256(token_ids):
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def postprocess(output, prompt_width, think_token_id, eos_token_id):
    completions = []
    closures = []
    for ids in output[:, prompt_width:].tolist():
        try:
            closure = ids.index(think_token_id)
            ids = ids[: closure + 1]
        except ValueError:
            closure = None
        if eos_token_id in ids:
            ids = ids[: ids.index(eos_token_id) + 1]
        completions.append(ids)
        closures.append(closure)
    return completions, closures


def select_samples(tokenizer):
    rows = []
    seen = set()
    with open(DATA, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group_id = row["recommendation_group_id"]
            if row["route"] != "think" or group_id in seen:
                continue
            seen.add(group_id)
            prompt_ids = tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False)
            rows.append((len(prompt_ids), row))

    selected = {}
    used = set()
    for tier, target in TARGET_LENGTHS.items():
        candidates = [item for item in rows if item[1]["recommendation_group_id"] not in used]
        prompt_tokens, row = min(candidates, key=lambda item: abs(item[0] - target))
        used.add(row["recommendation_group_id"])
        selected[tier] = {
            "tier": tier,
            "target_tokens": target,
            "prompt_tokens": prompt_tokens,
            "recommendation_group_id": row["recommendation_group_id"],
            "prompt": row["prompt"],
            "all_gold_sids": row["all_gold_sids"],
        }
    return selected


def prepare_inputs(tokenizer, prompt):
    rendered = [render_prompt(tokenizer, prompt)] * G
    encoded = tokenizer(
        text=rendered,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        max_length=8192,
        truncation=True,
        add_special_tokens=False,
    )
    return {key: value.to("cuda:0") for key, value in encoded.items()}


def counter_snapshot():
    snapshot = {}
    for category, counter in torch._dynamo.utils.counters.items():
        values = {str(key): value for key, value in counter.items() if value}
        if values:
            snapshot[str(category)] = values
    guard_failures = getattr(torch._dynamo.utils, "guard_failures", {})
    failures = []
    for code, reasons in guard_failures.items():
        for reason in reasons:
            failures.append({"code": str(code), "reason": str(reason)})
    snapshot["guard_failures"] = failures[:50]
    snapshot["guard_failure_count"] = len(failures)
    return snapshot


def run_generate(model, tokenizer, inputs, think_token_id, *, tier, mode, seed,
                 cache_implementation, disable_compile):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    criteria = StoppingCriteriaList([ThinkTokenStop(think_token_id)])
    config = generation_config(tokenizer, cache_implementation)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated() // (1024 * 1024)
    reserved_before = torch.cuda.memory_reserved() // (1024 * 1024)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=config,
            disable_compile=disable_compile,
            tokenizer=tokenizer,
            stopping_criteria=criteria,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    completions, closures = postprocess(
        output, inputs["input_ids"].size(1), think_token_id, tokenizer.eos_token_id,
    )
    return {
        "tier": tier,
        "mode": mode,
        "seed": seed,
        "cache_implementation": cache_implementation,
        "disable_compile": disable_compile,
        "completion_ids": completions,
        "completion_sha256": [completion_sha256(ids) for ids in completions],
        "completion_lengths": [len(ids) for ids in completions],
        "closure_positions": closures,
        "wall_sec": wall,
        "useful_tokens_per_sec": sum(map(len, completions)) / wall,
        "allocated_before_mb": allocated_before,
        "reserved_before_mb": reserved_before,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
        "counters": counter_snapshot(),
    }


def compare_tokens(reference, candidate):
    differences = []
    for index, (expected, actual) in enumerate(zip(reference["completion_ids"], candidate["completion_ids"])):
        first_difference = next(
            (position for position, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
            min(len(expected), len(actual)) if len(expected) != len(actual) else None,
        )
        if first_difference is not None:
            differences.append({
                "sample": index,
                "first_differing_token": first_difference,
                "baseline_token": expected[first_difference] if first_difference < len(expected) else None,
                "candidate_token": actual[first_difference] if first_difference < len(actual) else None,
                "baseline_length": len(expected),
                "candidate_length": len(actual),
            })
    return {
        "equal": not differences,
        "equal_samples": G - len(differences),
        "different_samples": len(differences),
        "sha256_equal": reference["completion_sha256"] == candidate["completion_sha256"],
        "closure_equal": reference["closure_positions"] == candidate["closure_positions"],
        "differences": differences,
    }


def stats(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def save(result):
    temporary = RESULT + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, RESULT)


def parsed_gold(sample):
    return {
        parsed for sid in sample["all_gold_sids"]
        if (parsed := parse_sid(sid)) is not None
    }


def reward_for_run(model, tokenizer, sample, run):
    texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in run["completion_ids"]]
    beam_fn = make_beam32_fn(model, tokenizer)
    torch.cuda.synchronize()
    started = time.perf_counter()
    rewards = beam_fn(
        [sample["prompt"]] * G,
        texts,
        run["completion_ids"],
        [parsed_gold(sample)] * G,
    )
    torch.cuda.synchronize()
    return rewards, time.perf_counter() - started


def main():
    torch.cuda.set_device(0)
    model, tokenizer, _ = load_model("cuda:0")
    model.eval()
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) != 1:
        raise RuntimeError(f"</think> must be one token, got {think_tokens}")
    think_token_id = think_tokens[0]

    samples = select_samples(tokenizer)
    inputs = {tier: prepare_inputs(tokenizer, sample["prompt"]) for tier, sample in samples.items()}
    public_samples = {
        tier: {key: value for key, value in sample.items() if key != "prompt"}
        for tier, sample in samples.items()
    }
    result = {
        "seed": SEED,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
        },
        "g": G,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_completion_length": MAX_COMPLETION,
        "think_token_ids": think_tokens,
        "paired_repeats": PAIRED_REPEATS,
        "compile_sequence": COMPILE_SEQUENCE,
        "model_use_cache": getattr(model.config, "use_cache", None),
        "samples": public_samples,
        "runs": {"dynamic": [], "static_no_compile": [], "static_compile": []},
        "parity": {},
        "rewards": {},
        "errors": [],
    }
    save(result)
    print("selected prompts:", {key: value["prompt_tokens"] for key, value in samples.items()}, flush=True)

    warm_config = generation_config(tokenizer, None, max_new_tokens=1)
    with torch.inference_mode():
        model.generate(
            **inputs["short"], generation_config=warm_config,
            disable_compile=True, tokenizer=tokenizer,
        )
    torch.cuda.synchronize()

    modes = {
        "dynamic": (None, True),
        "static_no_compile": ("static", True),
    }
    for tier_index, tier in enumerate(TARGET_LENGTHS):
        for repeat in range(PAIRED_REPEATS):
            order = tuple(modes) if repeat == 0 else tuple(reversed(modes))
            for mode in order:
                cache_implementation, disable_compile = modes[mode]
                try:
                    run = run_generate(
                        model, tokenizer, inputs[tier], think_token_id,
                        tier=tier,
                        mode=mode,
                        seed=SEED + tier_index,
                        cache_implementation=cache_implementation,
                        disable_compile=disable_compile,
                    )
                    result["runs"][mode].append(run)
                    print(
                        f"paired tier={tier} repeat={repeat + 1} mode={mode} "
                        f"wall={run['wall_sec']:.3f}s lengths={run['completion_lengths']} "
                        f"peak={run['peak_allocated_mb']}/{run['peak_reserved_mb']}MB",
                        flush=True,
                    )
                except Exception as error:
                    result["errors"].append({
                        "phase": "paired", "tier": tier, "mode": mode,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    print(f"paired failed tier={tier} mode={mode}: {type(error).__name__}: {error}", flush=True)
                save(result)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    guard_failures = getattr(torch._dynamo.utils, "guard_failures", None)
    if guard_failures is not None:
        guard_failures.clear()
    for sequence_index, tier in enumerate(COMPILE_SEQUENCE):
        tier_index = list(TARGET_LENGTHS).index(tier)
        compile_started = time.perf_counter()
        try:
            run = run_generate(
                model, tokenizer, inputs[tier], think_token_id,
                tier=tier,
                mode="static_compile",
                seed=SEED + tier_index,
                cache_implementation="static",
                disable_compile=False,
            )
            run["sequence_index"] = sequence_index
            result["runs"]["static_compile"].append(run)
            print(
                f"compile index={sequence_index} tier={tier} wall={run['wall_sec']:.3f}s "
                f"lengths={run['completion_lengths']} counters={run['counters']}",
                flush=True,
            )
        except Exception as error:
            torch.cuda.synchronize()
            result["errors"].append({
                "phase": "compile", "tier": tier, "sequence_index": sequence_index,
                "error": f"{type(error).__name__}: {error}",
                "failure_wall_sec": time.perf_counter() - compile_started,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
                "counters": counter_snapshot(),
                "traceback": traceback.format_exc(),
            })
            print(f"compile failed index={sequence_index} tier={tier}: {type(error).__name__}: {error}", flush=True)
            save(result)
            break
        save(result)

    references = {}
    for tier in TARGET_LENGTHS:
        references[tier] = next(
            (run for run in result["runs"]["dynamic"] if run["tier"] == tier), None,
        )
    for mode, runs in result["runs"].items():
        result["parity"][mode] = []
        for run in runs:
            reference = references[run["tier"]]
            comparison = compare_tokens(reference, run) if reference else {"equal": False, "reason": "missing baseline"}
            result["parity"][mode].append({
                "tier": run["tier"],
                "sequence_index": run.get("sequence_index"),
                **comparison,
            })

    reward_representatives = {
        "dynamic": {tier: references[tier] for tier in TARGET_LENGTHS},
        "static_no_compile": {
            tier: next((run for run in result["runs"]["static_no_compile"] if run["tier"] == tier), None)
            for tier in TARGET_LENGTHS
        },
        "static_compile": {
            tier: next((run for run in result["runs"]["static_compile"] if run["tier"] == tier), None)
            for tier in TARGET_LENGTHS
        },
    }
    for mode, tier_runs in reward_representatives.items():
        result["rewards"][mode] = {}
        for tier, run in tier_runs.items():
            if run is None:
                continue
            rewards, wall = reward_for_run(model, tokenizer, samples[tier], run)
            result["rewards"][mode][tier] = {"values": rewards, "beam_wall_sec": wall}
            print(f"reward mode={mode} tier={tier} values={rewards} wall={wall:.3f}s", flush=True)
            save(result)

    timing = {}
    for mode in ("dynamic", "static_no_compile"):
        timing[mode] = {}
        for tier in TARGET_LENGTHS:
            values = [run["wall_sec"] for run in result["runs"][mode] if run["tier"] == tier]
            if values:
                timing[mode][tier] = stats(values)
    compile_runs = result["runs"]["static_compile"]
    if compile_runs:
        timing["static_compile"] = {
            "first_call_sec": compile_runs[0]["wall_sec"],
            "sequence": [{"tier": run["tier"], "wall_sec": run["wall_sec"]} for run in compile_runs],
        }
        repeated = []
        seen_counts = Counter()
        for run in compile_runs:
            seen_counts[run["tier"]] += 1
            if seen_counts[run["tier"]] > 1:
                repeated.append(run["wall_sec"])
        if repeated:
            timing["static_compile"]["repeated_shape"] = stats(repeated)
        short_steady = [run["wall_sec"] for run in compile_runs if run["tier"] == "short"][1:]
        if short_steady:
            first_compile_overhead = compile_runs[0]["wall_sec"] - statistics.mean(short_steady)
            baseline_savings = timing["dynamic"]["short"]["mean"] - statistics.mean(short_steady)
            timing["static_compile"]["first_compile_overhead_sec"] = first_compile_overhead
            timing["static_compile"]["short_steady_saving_sec"] = baseline_savings
            timing["static_compile"]["break_even_short_rollouts"] = (
                math.ceil(first_compile_overhead / baseline_savings) if baseline_savings > 0 else None
            )
    result["timing"] = timing
    result["final_counters"] = counter_snapshot()
    result["reward_parity"] = {
        mode: {
            tier: result["rewards"].get(mode, {}).get(tier, {}).get("values")
            == result["rewards"].get("dynamic", {}).get(tier, {}).get("values")
            for tier in TARGET_LENGTHS
            if tier in result["rewards"].get(mode, {})
        }
        for mode in ("static_no_compile", "static_compile")
    }
    save(result)

    parity_counts = {
        mode: f"{sum(item.get('equal', False) * G for item in items)}/{len(items) * G}"
        for mode, items in result["parity"].items()
    }
    print(json.dumps({
        "timing": timing,
        "parity_counts": parity_counts,
        "reward_parity": result["reward_parity"],
        "errors": result["errors"],
        "final_counters": result["final_counters"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
