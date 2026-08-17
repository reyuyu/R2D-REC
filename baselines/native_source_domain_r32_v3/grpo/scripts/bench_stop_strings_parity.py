# -*- coding: utf-8 -*-
"""Fixed-seed real-model parity for redundant Think stop_strings."""
import hashlib
import json
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from grpo_model import load_model, render_prompt
from grpo_sid import parse_sid
from run_grpo_trl_smoke import DATA, make_beam32_fn

SEED = 20260817
BATCH_SIZE = 4
RESULT = "/data/GRPO/logs/stop_strings_parity.json"


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


def generation_config(tokenizer, stop_strings):
    return GenerationConfig(
        max_new_tokens=2048,
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
        stop_strings=stop_strings,
    )


def postprocess(output, prompt_length, think_token_id, eos_token_id):
    completions = []
    closure_positions = []
    for row in output[:, prompt_length:].tolist():
        try:
            close_pos = row.index(think_token_id)
            row = row[: close_pos + 1]
        except ValueError:
            close_pos = None
        if eos_token_id in row:
            row = row[: row.index(eos_token_id) + 1]
        completions.append(row)
        closure_positions.append(close_pos)
    return completions, closure_positions


def rollout(model, tokenizer, generate_inputs, think_token_id, stop_strings):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = generation_config(tokenizer, stop_strings)
    criteria = StoppingCriteriaList([ThinkTokenStop(think_token_id)])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **generate_inputs,
            generation_config=config,
            disable_compile=True,
            tokenizer=tokenizer,
            stopping_criteria=criteria,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    completions, closures = postprocess(
        output, generate_inputs["input_ids"].size(1), think_token_id,
        tokenizer.eos_token_id,
    )
    return completions, closures, wall


def digest(ids):
    payload = ",".join(str(token_id) for token_id in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main():
    torch.cuda.set_device(0)
    model, tokenizer, _ = load_model("cuda:0")
    model.eval()

    with open(DATA, encoding="utf-8") as handle:
        row = next(json.loads(line) for line in handle if json.loads(line)["route"] == "think")
    prompts = [row["prompt"]] * BATCH_SIZE
    gold_set = {parsed for sid in row["all_gold_sids"] if (parsed := parse_sid(sid)) is not None}
    rendered = [render_prompt(tokenizer, prompt) for prompt in prompts]
    generate_inputs = tokenizer(
        text=rendered,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        max_length=8192,
        truncation=True,
        add_special_tokens=False,
    )
    generate_inputs = {key: value.to("cuda:0") for key, value in generate_inputs.items()}
    think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # Warm kernels and allocator without consuming the measured RNG stream.
    warm_config = generation_config(tokenizer, None)
    warm_config.max_new_tokens = 1
    with torch.inference_mode():
        model.generate(**generate_inputs, generation_config=warm_config, disable_compile=True)
    torch.cuda.synchronize()

    baseline_ids, baseline_closures, baseline_wall = rollout(
        model, tokenizer, generate_inputs, think_token_id, ["</think>"],
    )
    candidate_ids, candidate_closures, candidate_wall = rollout(
        model, tokenizer, generate_inputs, think_token_id, None,
    )

    ids_equal = baseline_ids == candidate_ids
    closures_equal = baseline_closures == candidate_closures
    baseline_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in baseline_ids]
    candidate_text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in candidate_ids]
    baseline_reward = make_beam32_fn(model, tokenizer)(
        prompts, baseline_text, baseline_ids, [gold_set] * BATCH_SIZE,
    )
    candidate_reward = make_beam32_fn(model, tokenizer)(
        prompts, candidate_text, candidate_ids, [gold_set] * BATCH_SIZE,
    )

    result = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "prompt_tokens": int(generate_inputs["attention_mask"][0].sum()),
        "baseline_stop_strings": ["</think>"],
        "candidate_stop_strings": None,
        "completion_ids_equal": ids_equal,
        "closure_positions_equal": closures_equal,
        "reward_equal": baseline_reward == candidate_reward,
        "baseline": {
            "generation_wall_sec": baseline_wall,
            "completion_lengths": [len(ids) for ids in baseline_ids],
            "completion_sha256": [digest(ids) for ids in baseline_ids],
            "closure_positions": baseline_closures,
            "reward": baseline_reward,
        },
        "candidate": {
            "generation_wall_sec": candidate_wall,
            "completion_lengths": [len(ids) for ids in candidate_ids],
            "completion_sha256": [digest(ids) for ids in candidate_ids],
            "closure_positions": candidate_closures,
            "reward": candidate_reward,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    with open(RESULT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if not (ids_equal and closures_equal and result["reward_equal"]):
        raise SystemExit("PARITY FAILED: do not adopt stop_strings removal")
    print("PARITY PASSED", flush=True)


if __name__ == "__main__":
    main()
