# -*- coding: utf-8 -*-
"""Static safeguards for the call-local Beam prompt cache."""
import inspect

import run_grpo_trl_smoke


source = inspect.getsource(run_grpo_trl_smoke.make_beam32_fn)

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
assert "max_new_tokens=128, num_beams=32" in source
assert "num_return_sequences=32" in source

print("[PASS] prompt cache is reward-call scoped")
print("[PASS] legacy CoT decode/trim/encode path retained")
print("[PASS] Beam32 generation contract retained")
