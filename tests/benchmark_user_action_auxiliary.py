import argparse
import json
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from llamafactory.data.action_select import ActionSelectMetadataParser
from llamafactory.train.sft.user_action_auxiliary import UserActionAuxiliaryController


def make_args():
    return SimpleNamespace(
        user_action_history_trie_enabled=True,
        user_action_history_trie_weight=0.06,
        user_action_length_guard_enabled=True,
        user_action_continue_domain_extra=0.75,
        user_action_continue_separator_extra=0.20,
        user_action_no_early_stop_weight=0.02,
        user_action_stop_domain_weight=0.05,
        user_action_stop_tail_extra=1.0,
        user_action_max_stop_tail_positions=4,
        user_action_aux_cap_ratio=0.08,
        user_action_aux_warmup_steps=100,
    )


def synthetic_sample(parser, tokenizer):
    groups = parser.semantic_ids
    domains, a_ids, b_ids, c_ids = (sorted(groups[name]) for name in ("domain", "a", "b", "c"))
    sids = [
        (domains[0], a_ids[0], b_ids[0], c_ids[0]),
        (domains[0], a_ids[0], b_ids[0], c_ids[1]),
        (domains[1], a_ids[1], b_ids[1], c_ids[2]),
    ]
    filler = tokenizer.eos_token_id
    prompt = [filler]
    for sid in sids:
        prompt.extend(sid)
        prompt.append(filler)
    answer = [filler, *sids[0], filler, *sids[1], filler, filler]
    return prompt + answer, [-100] * len(prompt) + answer


def timed_mode(model, controller, metadata, input_ids, labels, enabled, warmup, steps):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if enabled:
            loss = loss + controller.compute(outputs.logits, labels, [metadata], outputs.loss, 100).loss
        loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if enabled:
            loss = loss + controller.compute(outputs.logits, labels, [metadata], outputs.loss, 100).loss
        loss.backward()
    ended.record()
    ended.synchronize()
    return started.elapsed_time(ended) / steps, torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/models/onereason-8b-pretrain-competition")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires one CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    metadata_parser = ActionSelectMetadataParser(tokenizer)
    input_values, label_values = synthetic_sample(metadata_parser, tokenizer)
    parse_started = time.perf_counter()
    for _ in range(1000):
        metadata = metadata_parser.parse(input_values, label_values)
    parse_ms = (time.perf_counter() - parse_started) * 1000.0 / 1000
    assert metadata["parse_valid"]
    device = torch.device("cuda")
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).to(device).eval()
    controller = UserActionAuxiliaryController(tokenizer, make_args())
    input_ids = torch.tensor([input_values], device=device)
    labels = torch.tensor([label_values], device=device)
    off_ms, off_memory = timed_mode(model, controller, metadata, input_ids, labels, False, args.warmup, args.steps)
    on_ms, on_memory = timed_mode(model, controller, metadata, input_ids, labels, True, args.warmup, args.steps)
    result = {
        "cpu_parse_ms_per_sample": parse_ms,
        "microbatch_off_ms": off_ms,
        "microbatch_on_ms": on_ms,
        "aux_increment_ms": on_ms - off_ms,
        "peak_memory_off_bytes": off_memory,
        "peak_memory_on_bytes": on_memory,
        "peak_memory_increment_bytes": max(0, on_memory - off_memory),
        "forward_per_microbatch": 1,
        "backward_per_microbatch": 1,
        "vocab_size": len(tokenizer),
        "sequence_length": len(input_values),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
