"""Four-GPU transition-only trainer for Bridge-Inside SFT V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    ADAPTER_SHA256,
    BASE,
    FRESH_BATA,
    IGNORE_INDEX,
    SEED,
    file_sha,
    transition_group_loss,
)

LR = 1e-7
SAVE_STEPS = (10, 25, 50)


def load_model(device: str, trainable: bool, adapter: Path = FRESH_BATA):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable and "lora" in name.lower()
    model.eval()
    return model, tokenizer


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("EMPTY_TRAIN_ROWS")
    return rows


def validate_rows(rows: list[dict[str, Any]], split: dict[str, Any], tokenizer) -> dict[str, Any]:
    if len(rows) != split["train_groups"]:
        raise RuntimeError("TRAIN_ROW_COUNT_MISMATCH")
    close_id = int(split["close_token_id"])
    lengths = []
    for row in rows:
        ids, labels = row["input_ids"], row["labels"]
        if len(ids) != len(labels):
            raise RuntimeError("INPUT_LABEL_LENGTH_MISMATCH")
        positions = [index for index, label in enumerate(labels) if label != IGNORE_INDEX]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise RuntimeError("NONCONTIGUOUS_TRANSITION_LABELS")
        labeled = [labels[index] for index in positions]
        if labeled[-1] != close_id or positions[-1] != len(labels) - 5:
            raise RuntimeError("LABEL_IS_NOT_BRIDGE_PLUS_CLOSE_BEFORE_DOMAIN_ABC")
        if any(label != IGNORE_INDEX for label in labels[:positions[0]]) or any(label != IGNORE_INDEX for label in labels[positions[-1] + 1:]):
            raise RuntimeError("LABEL_MASK_LEAK")
        lengths.append(len(labeled))
    for row in sorted(rows, key=lambda item: hashlib.sha256(("system_train_audit|" + item["group_id"]).encode()).hexdigest())[:64]:
        text = tokenizer.decode(row["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if text.count("<|im_start|>system") != 1 or text.count("<|im_start|>user") != 1 or text.count("<|im_start|>assistant") != 1:
            raise RuntimeError(f"SYSTEM_PRESERVE_FAIL group={row['group_id']}")
    return {
        "rows": len(rows),
        "system_preserve_pass": True,
        "labeled_token_min": min(lengths),
        "labeled_token_max": max(lengths),
        "labeled_token_mean": statistics.fmean(lengths),
        "labeled_abc_token_count": 0,
        "labeled_domain_token_count": 0,
        "labeled_cot_body_token_count": 0,
    }


def encode_row(row: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    return ids, labels


def transition_forward(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labeled_count = int(labels.ne(IGNORE_INDEX).sum())
    # Keep the predicting position, transition labels, and masked Domain+ABC suffix.
    keep = labeled_count + 5
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        logits_to_keep=keep,
    ).logits
    return transition_group_loss(logits, labels[:, -keep:])


def sampled_digest(model, *, lora: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ("lora" in name.lower()) != lora:
            continue
        flat = parameter.detach().reshape(-1)
        if flat.numel() > 128:
            indices = torch.arange(128, device=flat.device, dtype=torch.int64) * (flat.numel() - 1) // 127
            flat = flat.index_select(0, indices)
        digest.update(name.encode())
        digest.update(flat.float().cpu().numpy().tobytes())
    return digest.hexdigest()


def initial_parity(model, tokenizer, row: dict[str, Any], device: str) -> dict[str, float]:
    reference, reference_tokenizer = load_model(device, False)
    if tokenizer.get_vocab() != reference_tokenizer.get_vocab():
        raise RuntimeError("TOKENIZER_VOCAB_MISMATCH")
    ids, _ = encode_row(row, device)
    keep = min(24, ids.shape[1])
    with torch.inference_mode():
        actual = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits.float()
        expected = reference(input_ids=ids, use_cache=False, logits_to_keep=keep).logits.float()
    diff = (actual - expected).abs()
    del reference
    torch.cuda.empty_cache()
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "logit_positions": keep}


def manual_loss_parity() -> dict[str, Any]:
    torch.manual_seed(17)
    logits = torch.randn(3, 7, 19, dtype=torch.float64, requires_grad=True)
    labels = torch.full((3, 7), IGNORE_INDEX, dtype=torch.long)
    labels[0, 5:] = torch.tensor([1, 2])
    labels[1, 4:] = torch.tensor([3, 4, 5])
    labels[2, 2:] = torch.tensor([6, 7, 8, 9, 10])
    actual = transition_group_loss(logits, labels)
    expected_rows = []
    for index in range(3):
        shifted = labels[index, 1:]
        mask = shifted.ne(IGNORE_INDEX)
        expected_rows.append(torch.nn.functional.cross_entropy(logits[index, :-1][mask], shifted[mask]))
    expected = torch.stack(expected_rows).mean()
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    return {
        "forward_max_abs_diff": float((actual - expected).abs()),
        "gradient_max_abs_diff": float((actual_grad - expected_grad).abs().max()),
        "pass": bool(torch.equal(actual, expected) and torch.equal(actual_grad, expected_grad)),
    }


def setup() -> tuple[int, int, str]:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("BRIDGE_INSIDE_SFT_REQUIRES_4_GPUS")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    return rank, world, f"cuda:{rank}"


def write_result(path: Path, local: dict[str, Any], rank: int, world: int) -> None:
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ranks": gathered}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args) -> None:
    rank, world, device = setup()
    random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)
    rows = read_rows(Path(args.rows))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if file_sha(FRESH_BATA / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("FRESH_BATA_SHA_MISMATCH")
    model, tokenizer = load_model(device, True)
    audit = validate_rows(rows, split, tokenizer)
    row = rows[rank]
    parity = initial_parity(model, tokenizer, row, device)
    if parity["max_abs_diff"] != 0 or parity["mean_abs_diff"] != 0:
        raise RuntimeError(f"INIT_MODEL_PARITY_FAIL={parity}")
    only_lora = all(("lora" in name.lower()) == parameter.requires_grad for name, parameter in model.named_parameters())
    if not only_lora or len(model.peft_config) != 1:
        raise RuntimeError("EXISTING_LORA_OWNERSHIP_FAIL")
    base_before = sampled_digest(model, lora=False)
    lora_before = sampled_digest(model, lora=True)
    loss_parity = manual_loss_parity()
    if not loss_parity["pass"]:
        raise RuntimeError(f"MANUAL_LOSS_PARITY_FAIL={loss_parity}")

    ids, labels = encode_row(row, device)
    model.train()
    preflight_loss = transition_forward(model, ids, labels)
    preflight_loss.backward()
    lora_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if "lora" in name.lower())
    base_grad = any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if "lora" not in name.lower())
    zero_update = {
        "loss": float(preflight_loss.detach()),
        "optimizer_steps": 0,
        "lora_grad_nonzero": bool(lora_grad),
        "base_grad_nonzero": bool(base_grad),
        "base_changed": sampled_digest(model, lora=False) != base_before,
        "lora_changed": sampled_digest(model, lora=True) != lora_before,
        "only_existing_lora_trainable": only_lora,
        "second_lora_created": len(model.peft_config) != 1,
        "old_optimizer_resumed": False,
    }
    zero_update["pass"] = lora_grad and not base_grad and not zero_update["base_changed"] and not zero_update["lora_changed"] and not zero_update["second_lora_created"]
    preflight = {
        "rank": rank,
        "mode": args.mode,
        "adapter": str(FRESH_BATA),
        "adapter_sha256": ADAPTER_SHA256,
        "init_parity": parity,
        "row_audit": audit,
        "manual_loss_parity": loss_parity,
        "zero_update": zero_update,
        "system_preserve_pass": audit["system_preserve_pass"],
        "train_holdout_intersection": split["train_holdout_intersection"],
    }
    if not zero_update["pass"]:
        raise RuntimeError(f"ZERO_UPDATE_PREFLIGHT_FAIL={zero_update}")
    if args.mode == "preflight":
        write_result(Path(args.result), preflight, rank, world)
        dist.destroy_process_group()
        return

    if os.environ.get("BRIDGE_INSIDE_CONFIRM_FORMAL") != "1" or args.max_steps != 50 or not args.output_dir:
        raise RuntimeError("FORMAL_50_GUARD_FAIL")
    model.zero_grad(set_to_none=True)
    wrapped = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=0.0,
    )
    sampler = DistributedSampler(rows, num_replicas=world, rank=rank, shuffle=True, seed=SEED, drop_last=False)
    sampler.set_epoch(0)
    indices = iter(sampler)
    metrics = []
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_row = rows[next(indices)]
        ids, labels = encode_row(train_row, device)
        loss = transition_forward(wrapped, ids, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NONFINITE_LOSS step={step}")
        loss.backward()
        optimizer.step()
        value = torch.tensor(float(loss.detach()), device=device, dtype=torch.float64)
        dist.all_reduce(value)
        metric = {"step": step, "loss": float(value / world), "lr": LR}
        metrics.append(metric)
        if rank == 0:
            print(json.dumps(metric), flush=True)
        if step in SAVE_STEPS:
            if rank == 0:
                checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
            dist.barrier()
    final = {
        **preflight,
        "mode": "formal",
        "optimizer_steps": args.max_steps,
        "effective_global_batch_groups": world,
        "learning_rate": LR,
        "weight_decay": 0.0,
        "seed": SEED,
        "packing": False,
        "shuffle": True,
        "base_changed": sampled_digest(model, lora=False) != base_before,
        "lora_changed": sampled_digest(model, lora=True) != lora_before,
        "second_lora_created": len(model.peft_config) != 1,
        "old_optimizer_resumed": False,
        "checkpoints": [str(Path(args.output_dir) / f"checkpoint-{step}") for step in SAVE_STEPS],
        "metrics": metrics,
    }
    if final["base_changed"] or not final["lora_changed"] or final["second_lora_created"]:
        raise RuntimeError(f"POST_TRAINING_INVARIANT_FAIL={final}")
    write_result(Path(args.result), final, rank, world)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
