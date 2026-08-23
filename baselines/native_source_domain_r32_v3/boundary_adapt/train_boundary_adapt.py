"""Formal Boundary Adaptation trainer; formal launch requires an explicit shell guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_model import render_prompt
from boundary_adapt_loss import IGNORE_INDEX, assert_three_labels, load_and_validate_provenance, row_uniform_group_loss

BASE = "/data/models/onereason-8b-pretrain-competition"
ADAPTER = "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
DOMAIN = {"video": "<|video_begin|>", "prod": "<|prod_begin|>", "ad": "<|ad_begin|>", "living": "<|living_begin|>"}


def load(adapter_trainable: bool, device: str):
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # This attaches the existing adapter in-place; it neither merges nor creates one.
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=adapter_trainable)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = adapter_trainable and "lora" in name.lower()
    model.eval()
    return model, tok


def encode_row(tok, row: dict, device: str):
    ids = tok.encode(render_prompt(tok, row["prompt"]) + row["adapted_response"], add_special_tokens=False)
    abc = tok.encode(row["boundary_gold_sid"], add_special_tokens=False)
    domain = tok.encode(DOMAIN[row["target_domain"]], add_special_tokens=False)
    if len(abc) != 3 or len(domain) != 1 or ids[-3:] != abc:
        raise ValueError("token boundary contract failed")
    labels = [IGNORE_INDEX] * len(ids)
    labels[-3:] = abc
    labels = torch.tensor([labels], device=device)
    assert_three_labels(labels)
    return (
        torch.tensor([ids], device=device), labels,
        torch.tensor([row["boundary_sample_weight"]], device=device, dtype=torch.float32),
    )


def checksum(model, *, lora: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ("lora" in name.lower()) == lora:
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _write_rank_result(result_path: str, rank: int, item: dict, world: int) -> None:
    rank_path = Path(f"{result_path}.rank{rank}.json")
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps(item), encoding="utf-8")
    if rank != 0:
        return
    paths = [Path(f"{result_path}.rank{index}.json") for index in range(world)]
    deadline = time.time() + 180
    while not all(path.exists() for path in paths):
        if time.time() > deadline:
            raise TimeoutError("rank result files did not appear")
        time.sleep(.2)
    Path(result_path).write_text(json.dumps({"ranks": [json.loads(path.read_text()) for path in paths]}, indent=2), encoding="utf-8")


def verify_initial_parity(model, tokenizer, row, device: str) -> dict:
    ref, _ = load(False, device)
    ids, _, _ = encode_row(tokenizer, row, device)
    with torch.inference_mode():
        ref_logits = ref(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
        train_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
    diff = (train_logits - ref_logits).abs()
    del ref
    torch.cuda.empty_cache()
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def preflight(rows_path: str, stats_path: str, result_path: str) -> None:
    _run(rows_path, stats_path, result_path, mode="preflight", output_dir=None, max_steps=0)


def _run(rows_path: str, stats_path: str, result_path: str, *, mode: str, output_dir: str | None, max_steps: int) -> None:
    rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    # Match the repository's proven single-node four-GPU NCCL bootstrap.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = f"cuda:{rank}"
    provenance = load_and_validate_provenance(rows_path, stats_path)
    rows = [json.loads(line) for line in Path(rows_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[rank % len(rows)]
    model, tokenizer = load(True, device)
    parity = verify_initial_parity(model, tokenizer, row, device)
    if parity["max_abs_diff"] != 0.0:
        raise AssertionError(f"BASE_PLUS_EXISTING_LORA parity failed: {parity}")
    only_lora_trainable = all(("lora" in name.lower()) == parameter.requires_grad for name, parameter in model.named_parameters())
    if not only_lora_trainable:
        raise AssertionError("only LoRA parameters may be trainable")
    before_lora, before_base = checksum(model, lora=True), checksum(model, lora=False)
    ids, labels, weights = encode_row(tokenizer, row, device)
    model.train()
    wrapped = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)
    optimizer_steps = 0
    if mode == "preflight":
        loss = row_uniform_group_loss(wrapped(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, labels, weights, **provenance)
        loss.backward()
    else:
        sampler = DistributedSampler(rows, num_replicas=world, rank=rank, shuffle=(mode == "formal"), seed=20260824)
        loader = DataLoader(rows, batch_size=1, sampler=sampler, collate_fn=lambda batch: batch[0])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7)
        iterator = iter(loader)
        for step in range(max_steps):
            try:
                train_row = next(iterator)
            except StopIteration:
                sampler.set_epoch(step + 1); iterator = iter(loader); train_row = next(iterator)
            ids, labels, weights = encode_row(tokenizer, train_row, device)
            optimizer.zero_grad(set_to_none=True)
            loss = row_uniform_group_loss(wrapped(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, labels, weights, **provenance)
            loss.backward(); optimizer.step(); optimizer_steps += 1
            if mode == "formal" and (step + 1) in (50, 100, 200, 300):
                if rank == 0:
                    checkpoint = Path(output_dir) / f"checkpoint-{step + 1}"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
                dist.barrier()
    lora_nonzero = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora" in n.lower())
    base_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora" not in n.lower())
    item = {
        "rank": rank, "mode": mode, "loss": float(loss), "optimizer_steps": optimizer_steps,
        "labels": int(labels.ne(IGNORE_INDEX).sum()), "provenance": provenance,
        "init_parity": parity, "only_lora_trainable": only_lora_trainable,
        "lora_grad_nonzero": bool(lora_nonzero), "base_grad_nonzero": bool(base_nonzero),
        "lora_changed": checksum(model, lora=True) != before_lora,
        "base_changed": checksum(model, lora=False) != before_base,
        "old_optimizer_resumed": False, "second_lora_created": False,
    }
    _write_rank_result(result_path, rank, item, world)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--rows", required=True); parser.add_argument("--stats", required=True); parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir"); parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "formal" and os.environ.get("BOUNDARY_ADAPT_CONFIRM_FORMAL") != "1":
        raise SystemExit("Refusing formal launch without BOUNDARY_ADAPT_CONFIRM_FORMAL=1")
    if args.mode == "smoke" and args.max_steps != 1:
        raise SystemExit("Smoke requires exactly one optimizer step")
    if args.mode == "formal" and args.max_steps != 300:
        raise SystemExit("Formal trainer requires max_steps=300")
    if args.mode == "formal" and not args.output_dir:
        raise SystemExit("Formal trainer requires --output-dir")
    _run(args.rows, args.stats, args.result, mode=args.mode, output_dir=args.output_dir, max_steps=args.max_steps)
