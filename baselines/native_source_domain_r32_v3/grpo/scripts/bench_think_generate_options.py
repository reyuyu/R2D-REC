# -*- coding: utf-8 -*-
"""Real-model paired benchmark for low-risk Think generate options."""
import json
import statistics
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from grpo_model import load_model, render_prompt
from grpo_sid import parse_sid
from run_grpo_trl_smoke import make_beam32_fn

SAMPLES = "/data/GRPO/logs/beam_direct_ids_samples.json"
RESULT = "/data/GRPO/logs/think_generate_options.json"
SEED = 20260819
G = 4
PAIRED_REPEATS = 2


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


def generation_config(tokenizer, max_new_tokens=2048):
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
        cache_implementation=None,
        stop_strings=None,
    )


def postprocess(output, prompt_width, think_token_id, eos_token_id):
    completions = []
    closures = []
    for ids in output[:, prompt_width:].tolist():
        try:
            close = ids.index(think_token_id)
            ids = ids[: close + 1]
        except ValueError:
            close = None
        if eos_token_id in ids:
            ids = ids[: ids.index(eos_token_id) + 1]
        completions.append(ids)
        closures.append(close)
    return completions, closures


def run_generate(model, tokenizer, inputs, think_token_id, mode, disable_compile=True):
    use_inference_mode = mode == "inference_mode"
    pass_tokenizer = mode != "no_tokenizer"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = generation_config(tokenizer)
    criteria = StoppingCriteriaList([ThinkTokenStop(think_token_id)])
    kwargs = dict(
        **inputs,
        generation_config=config,
        disable_compile=disable_compile,
        stopping_criteria=criteria,
    )
    if pass_tokenizer:
        kwargs["tokenizer"] = tokenizer

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with context:
        output = model.generate(**kwargs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    completions, closures = postprocess(
        output, inputs["input_ids"].size(1), think_token_id, tokenizer.eos_token_id,
    )
    # Production returns Python lists; reconstructed policy inputs are ordinary
    # tensors, so inference tensors cannot leak into backward.
    reconstructed = [torch.tensor(ids, device="cpu") for ids in completions]
    return {
        "completion_ids": completions,
        "closure_positions": closures,
        "wall_sec": wall,
        "useful_tokens_per_sec": sum(map(len, completions)) / wall,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
        "reconstructed_is_inference": any(torch.is_inference(item) for item in reconstructed),
    }


def summary(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def compile_counters():
    counters = torch._dynamo.utils.counters
    return {
        "frames_total": counters["frames"]["total"],
        "unique_graphs": counters["stats"]["unique_graphs"],
        "fxgraph_cache_hit": counters["inductor"]["fxgraph_cache_hit"],
        "fxgraph_cache_miss": counters["inductor"]["fxgraph_cache_miss"],
    }


def main():
    fixture = json.load(open(SAMPLES, encoding="utf-8"))
    sample = sorted(fixture["samples"], key=lambda row: row["prompt_tokens"])[len(fixture["samples"]) // 2]
    prompt = sample["prompt"]
    gold_set = {
        parsed for sid in sample["all_gold_sids"]
        if (parsed := parse_sid(sid)) is not None
    }

    torch.cuda.set_device(0)
    model, tokenizer, _ = load_model("cuda:0")
    model.eval()
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) != 1:
        raise RuntimeError(f"</think> must be one token, got {think_tokens}")
    think_token_id = think_tokens[0]
    rendered = [render_prompt(tokenizer, prompt)] * G
    inputs = tokenizer(
        text=rendered, return_tensors="pt", padding=True, padding_side="left",
        max_length=8192, truncation=True, add_special_tokens=False,
    )
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}

    warm_config = generation_config(tokenizer, max_new_tokens=1)
    with torch.no_grad():
        model.generate(**inputs, generation_config=warm_config, disable_compile=True)
    torch.cuda.synchronize()

    modes = ("baseline", "inference_mode", "no_tokenizer")
    runs = {mode: [] for mode in modes}
    for repeat in range(PAIRED_REPEATS):
        order = modes if repeat == 0 else tuple(reversed(modes))
        for mode in order:
            result = run_generate(model, tokenizer, inputs, think_token_id, mode)
            runs[mode].append(result)
            print(
                f"pair={repeat + 1}/{PAIRED_REPEATS} mode={mode} "
                f"wall={result['wall_sec']:.3f}s lengths={list(map(len, result['completion_ids']))}",
                flush=True,
            )

    baseline_ids = runs["baseline"][0]["completion_ids"]
    baseline_closures = runs["baseline"][0]["closure_positions"]
    parity = {}
    for mode in modes:
        parity[mode] = {
            "token_equal": [run["completion_ids"] == baseline_ids for run in runs[mode]],
            "closure_equal": [run["closure_positions"] == baseline_closures for run in runs[mode]],
            "reconstructed_is_inference": [run["reconstructed_is_inference"] for run in runs[mode]],
        }

    # Reward parity uses the exact fixed completions from the first paired run.
    rewards = {}
    reward_wall = {}
    for mode in modes:
        ids = runs[mode][0]["completion_ids"]
        texts = [tokenizer.decode(item, skip_special_tokens=False) for item in ids]
        beam_fn = make_beam32_fn(model, tokenizer)
        torch.cuda.synchronize()
        started = time.perf_counter()
        rewards[mode] = beam_fn([prompt] * G, texts, ids, [gold_set] * G)
        torch.cuda.synchronize()
        reward_wall[mode] = time.perf_counter() - started
        print(f"reward mode={mode} wall={reward_wall[mode]:.3f}s values={rewards[mode]}", flush=True)

    # Diagnostic only: current cache configuration with compilation allowed.
    compile_runs = []
    counters_before = compile_counters()
    try:
        for index in range(2):
            compile_result = run_generate(
                model, tokenizer, inputs, think_token_id, "baseline", disable_compile=False,
            )
            compile_result["compile_counters"] = compile_counters()
            compile_runs.append(compile_result)
            print(f"compile diagnostic {index + 1}/2 wall={compile_result['wall_sec']:.3f}s", flush=True)
        compile_error = None
    except Exception as error:
        compile_error = f"{type(error).__name__}: {error}"
        print(f"compile diagnostic failed: {compile_error}", flush=True)

    output = {
        "seed": SEED,
        "prompt_tokens": int(inputs["attention_mask"][0].sum()),
        "think_token_ids": think_tokens,
        "paired_repeats": PAIRED_REPEATS,
        "baseline_completion_lengths": list(map(len, baseline_ids)),
        "baseline_closure_positions": baseline_closures,
        "model_config_use_cache": getattr(model.config, "use_cache", None),
        "model_generation_config_use_cache": getattr(model.generation_config, "use_cache", None),
        "benchmark_cache_implementation": generation_config(tokenizer).cache_implementation,
        "parity": parity,
        "timing": {
            mode: {
                "wall_sec": summary([run["wall_sec"] for run in mode_runs]),
                "useful_tokens_per_sec": summary([run["useful_tokens_per_sec"] for run in mode_runs]),
                "peak_allocated_mb": [run["peak_allocated_mb"] for run in mode_runs],
                "peak_reserved_mb": [run["peak_reserved_mb"] for run in mode_runs],
            }
            for mode, mode_runs in runs.items()
        },
        "reward": rewards,
        "reward_wall_sec": reward_wall,
        "compile_diagnostic": {
            "disable_compile": False,
            "error": compile_error,
            "counters_before": counters_before,
            "runs": [{
                "wall_sec": run["wall_sec"],
                "token_equal": run["completion_ids"] == baseline_ids,
                "closure_equal": run["closure_positions"] == baseline_closures,
                "peak_allocated_mb": run["peak_allocated_mb"],
                "peak_reserved_mb": run["peak_reserved_mb"],
                "compile_counters": run["compile_counters"],
            } for run in compile_runs],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    with open(RESULT, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    for mode in modes:
        if not all(parity[mode]["token_equal"] + parity[mode]["closure_equal"]):
            raise SystemExit(f"{mode} token parity failed")
    if any(rewards[mode] != rewards["baseline"] for mode in modes):
        raise SystemExit("reward parity failed")
    print("GENERATE OPTION PARITY PASSED", flush=True)


if __name__ == "__main__":
    main()
