# -*- coding: utf-8 -*-
"""Static safeguards for the call-local Beam prompt cache."""
import inspect

import run_grpo_trl_smoke


source = inspect.getsource(run_grpo_trl_smoke.make_beam32_fn)
beam_task_source = inspect.getsource(run_grpo_trl_smoke.run_beam32_task)

assert "prompt_ids_cache = {}" in source
assert "cached_prompt_ids(tokenizer, prompt, prompt_ids_cache)" in source

calls = []
original_encode_prompt = run_grpo_trl_smoke.encode_prompt
run_grpo_trl_smoke.encode_prompt = lambda tokenizer, prompt: calls.append(prompt) or [len(prompt)]
try:
    first_call_cache = {}
    assert run_grpo_trl_smoke.cached_prompt_ids(None, "same", first_call_cache) == [4]
    assert run_grpo_trl_smoke.cached_prompt_ids(None, "same", first_call_cache) == [4]
    assert run_grpo_trl_smoke.cached_prompt_ids(None, "other", first_call_cache) == [5]
    assert calls == ["same", "other"]

    # A new reward call owns a new dict, so the cache cannot grow across an epoch.
    second_call_cache = {}
    assert run_grpo_trl_smoke.cached_prompt_ids(None, "same", second_call_cache) == [4]
    assert calls == ["same", "other", "same"]
finally:
    run_grpo_trl_smoke.encode_prompt = original_encode_prompt

# CoT preprocessing intentionally remains decode -> text trim -> encode because
# direct sampled IDs failed token parity on the real-model fixture.
assert "tokenizer.decode(cot_ids, skip_special_tokens=False)" in source
assert "cot.find(\"</think>\")" in source
assert "tokenizer.encode(cot_trim, add_special_tokens=False)" in source

# Frozen Beam32 contract.
assert "max_new_tokens=128" in beam_task_source
assert "num_beams=32" in beam_task_source
assert "num_return_sequences=32" in beam_task_source

# Distributed scheduling changes only the executing rank. Its deterministic
# tie-breaker covers every task exactly once and the single-process path stays
# available for tests/diagnostics.
tasks = [
    {"task_id": (0, index), "input_ids": [0] * length}
    for index, length in enumerate([100, 90, 80, 70, 60, 50, 40, 30])
]
assignments, loads = run_grpo_trl_smoke.beam_lpt_assignment(tasks, 4)
assert sorted(task_id for rank_tasks in assignments.values() for task_id in rank_tasks) == [
    task["task_id"] for task in tasks
]
assert loads == [130, 130, 130, 130]
assert run_grpo_trl_smoke.distributed_beam_enabled() is False

print("[PASS] prompt cache is reward-call scoped")
print("[PASS] legacy CoT decode/trim/encode path retained")
print("[PASS] Beam32 generation contract retained")
print("[PASS] deterministic length-LPT scheduler covers every task once")
