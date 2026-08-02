#!/usr/bin/env python3
"""Generate checkpoint predictions for the approximate SID/F1 monitor.

This intentionally evaluates a bounded, fixed prefix of the deterministic 2%
dev split. It is designed for checkpoint comparison during training, not as a
replacement for the official competition evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments


TASK_FILES = {
    "recommendation": "onereason_recommendation_cot_dev2.jsonl",
    "action": "onereason_user_action_nocot_dev2.jsonl",
}


def make_messages(row: dict) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in row.get("history") or []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({"role": "user", "content": str(turn[0])})
            messages.append({"role": "assistant", "content": str(turn[1])})
    query = "\n".join(part for part in (row.get("instruction", ""), row.get("input", "")) if part)
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": ""})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, default=Path("/data/lf_data_splits"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="recommendation,action")
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--max-samples-per-task", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=7168)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=args.template))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval().cuda()
    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if unknown := set(requested_tasks) - set(TASK_FILES):
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer, torch.inference_mode():
        for task_name in requested_tasks:
            path = args.dev_dir / TASK_FILES[task_name]
            with path.open(encoding="utf-8") as reader:
                for index, line in enumerate(reader):
                    if args.max_samples_per_task > 0 and index >= args.max_samples_per_task:
                        break
                    row = json.loads(line)
                    prompt_ids, _ = template.encode_oneturn(tokenizer, make_messages(row))
                    prompt_ids = prompt_ids[: args.max_prompt_tokens]
                    input_ids = torch.tensor([prompt_ids], device=model.device)
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    prediction = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=False)
                    writer.write(
                        json.dumps(
                            {
                                "task": task_name,
                                "index": index,
                                "prediction": prediction,
                                "reference": row["output"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    writer.flush()
                    print(f"{task_name} {index + 1}", flush=True)


if __name__ == "__main__":
    main()
