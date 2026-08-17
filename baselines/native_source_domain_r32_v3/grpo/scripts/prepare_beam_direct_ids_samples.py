# -*- coding: utf-8 -*-
"""Sample fixed real-model Think completions for Beam context parity."""
import json
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from grpo_model import load_model, render_prompt
from grpo_trl_trainer import build_route_dataset
from run_grpo_trl_smoke import DATA

SEED = 20260818
N_GROUPS = 4
G = 4
OUTPUT = "/data/GRPO/logs/beam_direct_ids_samples.json"


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


def config(tokenizer, max_new_tokens=2048):
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


def sample(model, tokenizer, prompt, seed, think_token_id):
    rendered = [render_prompt(tokenizer, prompt)] * G
    inputs = tokenizer(
        text=rendered,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        max_length=8192,
        truncation=True,
        add_special_tokens=False,
    )
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    criteria = StoppingCriteriaList([ThinkTokenStop(think_token_id)])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=config(tokenizer),
            disable_compile=True,
            tokenizer=tokenizer,
            stopping_criteria=criteria,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    prompt_width = inputs["input_ids"].size(1)
    completions = []
    closures = []
    for ids in output[:, prompt_width:].tolist():
        try:
            close_pos = ids.index(think_token_id)
            ids = ids[: close_pos + 1]
        except ValueError:
            close_pos = None
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
        completions.append(ids)
        closures.append(close_pos)
    return completions, closures, wall, int(inputs["attention_mask"][0].sum())


def main():
    torch.cuda.set_device(0)
    model, tokenizer, _ = load_model("cuda:0")
    model.eval()
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) != 1:
        raise RuntimeError(f"</think> must be one token, got {think_tokens}")
    think_token_id = think_tokens[0]

    dataset = build_route_dataset(DATA, n_groups=N_GROUPS, seed=SEED, chunk=N_GROUPS)
    rows = [row for row in dataset if row["route"] == "think"]

    warm_inputs = tokenizer(
        text=[render_prompt(tokenizer, rows[0]["prompt"])] * G,
        return_tensors="pt", padding=True, padding_side="left",
        max_length=8192, truncation=True, add_special_tokens=False,
    )
    warm_inputs = {key: value.to("cuda:0") for key, value in warm_inputs.items()}
    with torch.inference_mode():
        model.generate(**warm_inputs, generation_config=config(tokenizer, 1), disable_compile=True)
    torch.cuda.synchronize()

    samples = []
    total_wall = 0.0
    for index, row in enumerate(rows):
        completions, closures, wall, prompt_tokens = sample(
            model, tokenizer, row["prompt"], SEED + index, think_token_id,
        )
        total_wall += wall
        samples.append({
            "recommendation_group_id": row["recommendation_group_id"],
            "prompt": row["prompt"],
            "all_gold_sids": row["all_gold_sids"],
            "prompt_tokens": prompt_tokens,
            "completion_ids": completions,
            "closure_positions": closures,
            "generation_wall_sec": wall,
        })
        print(f"group {index + 1}/{N_GROUPS}: wall={wall:.3f}s closures={closures}", flush=True)

    result = {
        "seed": SEED,
        "think_token_id": think_token_id,
        "groups": N_GROUPS,
        "completions": N_GROUPS * G,
        "generation_wall_sec": total_wall,
        "samples": samples,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
