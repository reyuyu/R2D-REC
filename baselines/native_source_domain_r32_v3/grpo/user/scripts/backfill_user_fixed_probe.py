#!/usr/bin/env python3
"""Rollout-only fixed-probe backfill for an existing User GRPO run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from run_user_grpo_smoke import generate_route, read_jsonl
from user_fixed_probe import aggregate_probe_steps, evaluate_user_fixed_probe, validate_probe_rows


EXPECTED_PROBE_SHA = "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lora_checksums(model, limit: int = 10) -> dict[str, str]:
    output = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        value = parameter.detach().float().cpu().contiguous().numpy().tobytes()
        output[name] = hashlib.sha256(value).hexdigest()
        if len(output) == limit:
            break
    if len(output) < limit:
        raise RuntimeError(f"expected at least {limit} LoRA tensors, found {len(output)}")
    return output


def append_events(path: Path, events: list[dict], append: bool) -> None:
    if path.exists() and not append:
        raise FileExistsError(f"probe output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "x", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--expected-sha", default=EXPECTED_PROBE_SHA)
    args = parser.parse_args()

    if file_sha256(args.probe) != args.expected_sha:
        raise RuntimeError("frozen User fixed-probe SHA mismatch")
    rows = read_jsonl(args.probe)
    validate_probe_rows(rows)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 4 or local_rank not in range(4):
        raise RuntimeError("fixed-probe backfill requires exactly four GPU ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("fixed-probe model must be fully frozen")

    before = lora_checksums(model)
    events = evaluate_user_fixed_probe(
        model,
        tokenizer,
        rows,
        device,
        rank=rank,
        world_size=world_size,
        step=args.step,
        reason=args.reason,
        seed=args.seed,
        generate_fn=generate_route,
    )
    after = lora_checksums(model)
    if before != after:
        raise RuntimeError("LoRA checksum changed during rollout-only probe")
    if file_sha256(args.probe) != args.expected_sha:
        raise RuntimeError("frozen User fixed-probe SHA changed during probe")

    if rank == 0:
        if len(events) != len(rows):
            raise RuntimeError(f"expected {len(rows)} probe groups, got {len(events)}")
        append_events(args.output, events, args.append)
        print(json.dumps({
            "output": str(args.output),
            "groups": len(events),
            "completions": sum(len(event["candidates"]) for event in events),
            "lora_tensors_checked": len(before),
            "lora_checksum_unchanged": True,
            "probe_sha_unchanged": True,
            "metrics": aggregate_probe_steps(events)[0],
        }, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
