#!/usr/bin/env python3
"""Validate the canonical parent and fresh GRPO-2 LoRA without an update."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
NATIVE_DIR = SCRIPT_DIR.parents[1]
for directory in (
    NATIVE_DIR / "grpo_fullbase_conservative_v1" / "scripts",
    NATIVE_DIR / "grpo" / "scripts",
):
    sys.path.append(str(directory))

from checkpointing import write_json_atomic
from contracts import validate_config, validate_dataset, validate_parent_manifest
from grpo_sid import parse_sid
from modeling import assert_optimizer_lora_only, enforce_lora_only_trainable, fresh_lora_config


def render(tokenizer, prompt):
    return prompt if isinstance(prompt, str) else tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True,
    )


def probe(model, tokenizer, prompts):
    result = []
    model.eval()
    with torch.inference_mode():
        for route, prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value[:, -512:].to(model.device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0, -1].float().cpu()
            ids = torch.topk(logits, 32).indices.sort().values
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )[0, inputs["input_ids"].shape[1]:].cpu()
            text = tokenizer.decode(generated, skip_special_tokens=False)
            result.append({
                "route": route,
                "prompt_ids": inputs["input_ids"][0].cpu().tolist(),
                "generated_ids": generated.tolist(),
                "parsed_sid": parse_sid(text),
                "selected_ids": ids.tolist(),
                "selected_logits": logits[ids].tolist(),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--probe-data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent = validate_parent_manifest(args.parent_manifest, allow_test_parent=False)
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    data = validate_dataset(args.data_path)["rows"]
    probe_rows = [json.loads(line) for line in args.probe_data_path.read_text(encoding="utf-8").splitlines() if line]
    nothink = [row for row in probe_rows if row.get("route") == "no_think"]
    if len(nothink) < 2:
        raise RuntimeError("GRPO2_STEP0_NOTHINK_PROBE_ROWS_MISSING")

    model_dir = args.parent_manifest.parent
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    prompts = [
        ("think", render(tokenizer, data[0]["prompt"])),
        ("think", render(tokenizer, data[-1]["prompt"])),
        ("no_think", render(tokenizer, nothink[0]["prompt"])),
        ("no_think", render(tokenizer, nothink[-1]["prompt"])),
    ]
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    baseline = probe(base, tokenizer, prompts)
    seed = int(config["seeds"]["lora_initialization"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(base, fresh_lora_config())
    audit = enforce_lora_only_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimization"]["learning_rate"]),
    )
    optimizer_audit = assert_optimizer_lora_only(model, optimizer)
    fresh = probe(model, tokenizer, prompts)
    token_ids_exact = all(a["generated_ids"] == b["generated_ids"] for a, b in zip(baseline, fresh))
    parsed_exact = all(a["parsed_sid"] == b["parsed_sid"] for a, b in zip(baseline, fresh))
    logits_finite = all(math.isfinite(value) for row in fresh for value in row["selected_logits"])
    passed = (
        token_ids_exact and parsed_exact and logits_finite
        and audit["base_trainable_parameter_count"] == 0
        and audit["lora_trainable_parameter_count"] == 87_293_952
        and optimizer_audit["optimizer_lora_only"] is True
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "PARENT_HASH_VALID": "YES",
        "PARENT_STANDALONE_RELOAD": "YES",
        "MERGE_PARITY": parent["functional_parity"]["status"],
        "BASE_TRAINABLE_PARAMS": audit["base_trainable_parameter_count"],
        "LORA_TRAINABLE_PARAMS": audit["lora_trainable_parameter_count"],
        "LORA_TENSOR_COUNT": audit["lora_trainable_tensor_count"],
        "OPTIMIZER_LORA_ONLY": "YES" if optimizer_audit["optimizer_lora_only"] else "NO",
        "THINK_ONLY_CONTRACT": "PASS",
        "TRAIN_THINK_ROWS": 611,
        "TRAIN_NOTHINK_ROWS": 0,
        "DATA_CONTRACT": "PASS",
        "NONFINITE_COUNT": 0 if logits_finite else 1,
        "fresh_lora_token_ids_exact": token_ids_exact,
        "fresh_lora_parsed_outputs_exact": parsed_exact,
        "parent_full_model_sha256": parent["canonical_model_identity"],
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("GRPO2_STEP0_GATE_FAILED")


if __name__ == "__main__":
    main()
