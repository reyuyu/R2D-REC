#!/usr/bin/env python3
"""Four-rank load-only validation for Pilot300 LoRA and Adam resume state."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from run_user_pilot150 import validate_restored_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.get_world_size() != 4:
        raise RuntimeError("resume GPU preflight requires four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base, args.checkpoint, is_trainable=True, local_files_only=True
    )
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
        if parameter.requires_grad:
            trainable.append((name, parameter))
    if not trainable or any("lora_" not in name.lower() for name, _parameter in trainable):
        raise RuntimeError("resume preflight found an invalid trainable-parameter set")
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in trainable], lr=1e-6, weight_decay=0.0
    )
    state = torch.load(
        os.path.join(args.checkpoint, "optimizer.pt"),
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(state)
    audit = validate_restored_optimizer(optimizer, args.expected_step)
    if audit["state_count"] != len(trainable):
        raise RuntimeError(
            f"optimizer/LoRA parameter count mismatch: {audit['state_count']} != {len(trainable)}"
        )
    local = {
        "rank": rank,
        "gpu": local_rank,
        "lora_parameter_count": len(trainable),
        "optimizer_state_count": audit["state_count"],
        "optimizer_steps": audit["steps"],
        "requires_grad_non_lora": 0,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if rank == 0:
        print(json.dumps({"status": "PASS", "ranks": gathered}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
