#!/usr/bin/env python3
"""Four-prompt User GRPO rollout/loss and two-step optimizer correctness smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

from user_grpo_trainer import UserGRPOTrainer, prepare_scored_rollout
from user_training_objective import (
    clipped_grpo_loss_from_logps,
    standard_sequence_grpo_loss_from_logps,
)


BASE_MODEL = "/data/models/onereason-8b-pretrain-competition"
ADAPTER = (
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
PILOT = "/data/GRPO_USER/data/gr_user_v1/pilot_600.jsonl"
G = 4
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SEED = 20260819
SELECTED_SAMPLE_IDS = {
    "action": (
        "5b2e7e603229aeeae399ec52a7c0e9f0f22f4e4f7a3f1e126a4642c57bf96a49",
        "37400eceb1dc973fe5cf2112de3046aee1af34cc4339a512534ff8a43eb53baf",
    ),
    "chain": (
        "9c8f9b5fccb7ba236aa8d1797ae15818fce67028c4935f11b79e6b945082abdd",
        "61022eaae6ece8f7411832802ac34fe1ee60017c2a074fce4ca66b8550f4388b",
    ),
}
EXPECTED_DATA_SHA = {
    "train_3000.jsonl": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot_600.jsonl": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe_v1.jsonl": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha(data_dir):
    return {name: file_sha256(Path(data_dir) / name) for name in EXPECTED_DATA_SHA}


def gpu_preflight(physical_gpu_id):
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().splitlines()
    states = {}
    for row in rows:
        index, uuid, used, free, utilization = [item.strip() for item in row.split(",")]
        states[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    compute_text = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    processes = []
    for row in compute_text.splitlines() if compute_text else []:
        uuid, pid, process_name, used = [item.strip() for item in row.split(",", 3)]
        processes.append(
            {"uuid": uuid, "pid": int(pid), "process_name": process_name, "used_memory_mib": int(used)}
        )
    if physical_gpu_id not in states:
        raise RuntimeError(f"GPU {physical_gpu_id} does not exist")
    state = states[physical_gpu_id]
    selected_processes = [item for item in processes if item["uuid"] == state["uuid"]]
    if selected_processes or state["memory_used_mib"] > 1024:
        raise RuntimeError(f"GPU {physical_gpu_id} is not idle")
    return {**state, "compute_processes": selected_processes}


def render_prompt(tokenizer, row):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if count != row["prompt_token_count"]:
        raise AssertionError("Phase 1 prompt renderer drift")
    return rendered


def select_rows(rows):
    by_id = {row["sample_id"]: row for row in rows}
    selected = {
        route: [by_id[sample_id] for sample_id in sample_ids]
        for route, sample_ids in SELECTED_SAMPLE_IDS.items()
    }
    for route, route_rows in selected.items():
        if len(route_rows) != 2 or any(row["route"] != route for row in route_rows):
            raise AssertionError(f"invalid {route} smoke selection")
    if len({row["sample_id"] for values in selected.values() for row in values}) != 4:
        raise AssertionError("smoke selection is not four unique prompts")
    return selected


def trim_at_stop(token_ids, stop_ids):
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return list(token_ids[:index])
    return list(token_ids)


@torch.inference_mode()
def generate_route(model, tokenizer, rows, device):
    prompts = [render_prompt(tokenizer, row) for row in rows]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    stop_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=G,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=sorted(stop_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    generated = output[:, encoded["input_ids"].size(1) :].cpu().tolist()
    return [trim_at_stop(ids, stop_ids) for ids in generated]


def make_policy_batch(tokenizer, rollout, device, candidate_indices=None):
    indices = list(range(len(rollout["completion_ids_list"]))) if candidate_indices is None else list(candidate_indices)
    prompt_ids_list = []
    completion_ids_list = []
    for index in indices:
        row = rollout["expanded_rows"][index]
        prompt_ids_list.append(tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False))
        completion_ids_list.append(rollout["completion_ids_list"][index])
    prompt_width = max(map(len, prompt_ids_list))
    completion_width = max(map(len, completion_ids_list))
    pad_id = tokenizer.pad_token_id
    prompt_ids = torch.full((len(indices), prompt_width), pad_id, dtype=torch.long, device=device)
    prompt_mask = torch.zeros_like(prompt_ids)
    completion_ids = torch.full(
        (len(indices), completion_width), pad_id, dtype=torch.long, device=device
    )
    completion_mask = torch.zeros_like(completion_ids, dtype=torch.float32)
    for row_index, ids in enumerate(prompt_ids_list):
        prompt_ids[row_index, -len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
        prompt_mask[row_index, -len(ids) :] = 1
    for row_index, ids in enumerate(completion_ids_list):
        completion_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        completion_mask[row_index, : len(ids)] = 1
    token_advantages = rollout["token_advantages"][indices, :completion_width].to(device)
    local_penalty_mask = rollout["local_penalty_mask"][indices, :completion_width].to(device)
    sequence_advantages = rollout["sequence_advantages"][indices].to(device)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "token_advantages": token_advantages,
        "local_penalty_mask": local_penalty_mask,
        "sequence_advantages": sequence_advantages,
    }


def parameter_sha256(model, include_lora):
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in model.named_parameters():
        if ("lora_" in name.lower()) != include_lora:
            continue
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
        matched += 1
    if matched == 0:
        raise RuntimeError("parameter hash matched no tensors")
    return digest.hexdigest(), matched


def lora_snapshot(model):
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    }


def lora_delta(model, before):
    squared = 0.0
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        delta = parameter.detach().float().cpu() - before[name]
        squared += float((delta * delta).sum())
        maximum = max(maximum, float(delta.abs().max()))
    return math.sqrt(squared), maximum


def grad_norm(parameters):
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float((parameter.grad.detach().float() ** 2).sum())
    return math.sqrt(squared)


def no_mask_parity_audit(trials=32):
    generator = torch.Generator().manual_seed(SEED)
    max_loss = max_per_token = max_gradient = 0.0
    for _ in range(trials):
        current_a = torch.randn(8, 13, generator=generator, dtype=torch.float64, requires_grad=True)
        current_b = current_a.detach().clone().requires_grad_(True)
        old = torch.randn(8, 13, generator=generator, dtype=torch.float64)
        advantages = torch.randn(8, generator=generator, dtype=torch.float64)
        mask = (torch.rand(8, 13, generator=generator) > 0.15).to(torch.float64)
        standard, standard_tokens, _ = standard_sequence_grpo_loss_from_logps(
            current_a, old, advantages, mask
        )
        local, local_tokens, _ = clipped_grpo_loss_from_logps(
            current_b, old, advantages[:, None].expand_as(current_b), mask
        )
        standard.backward()
        local.backward()
        max_loss = max(max_loss, abs(float(standard - local)))
        max_per_token = max(max_per_token, float((standard_tokens - local_tokens).abs().max()))
        max_gradient = max(max_gradient, float((current_a.grad - current_b.grad).abs().max()))
    return {
        "trials": trials,
        "max_loss_error": max_loss,
        "max_per_token_loss_error": max_per_token,
        "max_gradient_error": max_gradient,
    }


def local_gradient_contrast(batch, old_logps):
    completion_mask = batch["completion_mask"]
    local_mask = batch["local_penalty_mask"] & completion_mask.bool()
    if not bool(local_mask.any()):
        raise RuntimeError("real smoke batch contains no local penalty tokens")
    valid = completion_mask.bool()
    unmasked = valid & ~local_mask
    sequence_advantages = batch["sequence_advantages"].double()
    token_advantages = batch["token_advantages"].double()
    expected_unmasked = sequence_advantages[:, None].expand_as(token_advantages)
    unmasked_advantage_error = float(
        (token_advantages - expected_unmasked).abs()[unmasked].max()
    )
    if unmasked_advantage_error != 0.0:
        raise RuntimeError("unmasked token advantages differ from sequence advantages")

    # This is a mathematical objective audit. Evaluate both branches in float64 so
    # BF16 model log-prob rounding cannot be mistaken for a gradient-locality bug.
    reference_old = old_logps.detach().double()
    off_current = reference_old.clone().requires_grad_(True)
    on_current = reference_old.clone().requires_grad_(True)
    off_loss, _, _ = standard_sequence_grpo_loss_from_logps(
        off_current,
        reference_old,
        sequence_advantages,
        completion_mask,
    )
    on_loss, _, _ = clipped_grpo_loss_from_logps(
        on_current,
        reference_old,
        token_advantages,
        completion_mask,
    )
    off_loss.backward()
    on_loss.backward()
    masked_change = (off_current.grad - on_current.grad).abs()[local_mask]
    return {
        "unmasked_advantage_max_error": unmasked_advantage_error,
        "unmasked_gradient_parity_max_error": float(
            (off_current.grad - on_current.grad).abs()[unmasked].max()
        ),
        "masked_gradient_changed_token_count": int((masked_change > 1e-12).sum()),
        "masked_gradient_max_abs_change": float(masked_change.max()),
        "off_loss": float(off_loss),
        "on_loss": float(on_loss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/data/GRPO_USER/runs"))
    parser.add_argument(
        "--result-output",
        type=Path,
        default=Path("/data/GRPO_USER/results/trainer_correctness_smoke_v1.json"),
    )
    parser.add_argument("--optimizer-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()
    if not 1 <= args.optimizer_steps <= 4:
        raise ValueError("optimizer smoke permits only 1-4 steps")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.physical_gpu_id):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match the selected physical GPU")
    gpu_state = gpu_preflight(args.physical_gpu_id)
    parity = no_mask_parity_audit()
    if parity["max_loss_error"] > 1e-12 or parity["max_gradient_error"] > 1e-12:
        raise RuntimeError("no-mask mathematical parity failed")

    data_dir = str(Path(PILOT).parent)
    sha_before = dataset_sha(data_dir)
    if sha_before != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen GR_USER_v1 data SHA mismatch")
    rows = read_jsonl(PILOT)
    selected = select_rows(rows)
    run_id = "GR-USER-TRAINER-SMOKE-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        model, ADAPTER, is_trainable=True, local_files_only=True
    )
    torch.cuda.reset_peak_memory_stats(0)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name.lower() for name, _ in trainable):
        raise RuntimeError("only LoRA parameters may be trainable")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name.lower()):
        raise RuntimeError("base model has trainable parameters")
    trainer = UserGRPOTrainer.for_correctness_smoke(model, forward_batch_size=1)

    model.eval()
    generated = {}
    rollouts = {}
    forward_results = {}
    started = time.perf_counter()
    for route in ("action", "chain"):
        generated[route] = generate_route(model, tokenizer, selected[route], device)
        rollouts[route] = prepare_scored_rollout(selected[route], generated[route], tokenizer)
        batch = make_policy_batch(tokenizer, rollouts[route], device)
        with torch.no_grad():
            loss = trainer._compute_loss(model, batch)
        forward_results[route] = {
            "loss": float(loss),
            "loss_metrics": dict(trainer.last_loss_metrics),
            "rollout_metrics": rollouts[route]["metrics"],
        }
        if not trainer.last_loss_metrics["finite"]:
            raise RuntimeError(f"{route} forward-only loss is non-finite")
        if batch["token_advantages"].shape != batch["completion_ids"].shape:
            raise RuntimeError("forward-only token advantage shape mismatch")
    if not any(bool(rollout["local_penalty_mask"].any()) for rollout in rollouts.values()):
        raise RuntimeError("forward-only smoke produced no whitelist penalty signal")
    forward_wall = time.perf_counter() - started
    forward_peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    optimizer_batches = {
        route: make_policy_batch(tokenizer, rollouts[route], device, range(G))
        for route in ("action", "chain")
    }
    old_logps = {}
    with torch.no_grad():
        for route, batch in optimizer_batches.items():
            combined = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
            attention = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
            old_logps[route] = trainer._get_user_per_token_logps(
                model, combined, attention, batch["completion_ids"].size(1)
            ).detach()
            batch["old_per_token_logps"] = old_logps[route]
    contrast_batches = [
        (route, batch)
        for route, batch in optimizer_batches.items()
        if bool(batch["local_penalty_mask"].any())
    ]
    if not contrast_batches:
        raise RuntimeError("optimizer smoke groups contain no local penalty tokens")
    contrast_route, contrast_batch = contrast_batches[0]
    contrast = {"route": contrast_route, **local_gradient_contrast(contrast_batch, old_logps[contrast_route])}
    if contrast["unmasked_gradient_parity_max_error"] > 1e-12:
        raise RuntimeError("unmasked OFF/ON gradient parity failed")
    if contrast["masked_gradient_changed_token_count"] == 0:
        raise RuntimeError("masked OFF/ON gradients did not change")

    base_hash_before, base_tensor_count = parameter_sha256(model, include_lora=False)
    lora_hash_before, lora_tensor_count = parameter_sha256(model, include_lora=True)
    lora_before = lora_snapshot(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    model.train()
    step_results = []
    route_order = ["action", "chain"] * 2
    for step in range(args.optimizer_steps):
        route = route_order[step]
        batch = optimizer_batches[route]
        optimizer.zero_grad(set_to_none=True)
        loss = trainer._compute_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("optimizer smoke loss is non-finite")
        loss.backward()
        raw_grad_norm = grad_norm([parameter for _, parameter in trainable])
        if not math.isfinite(raw_grad_norm) or raw_grad_norm == 0.0:
            raise RuntimeError("LoRA gradient norm is invalid")
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], 1.0
        )
        optimizer.step()
        step_results.append(
            {
                "step": step + 1,
                "route": route,
                "loss": float(loss),
                "grad_norm": raw_grad_norm,
                "clip_grad_norm_return": float(clipped_norm),
                "ratio_mean": trainer.last_loss_metrics["ratio_mean"],
                "clip_fraction": trainer.last_loss_metrics["clip_fraction"],
                "finite": trainer.last_loss_metrics["finite"],
            }
        )

    model.eval()
    base_hash_after, _ = parameter_sha256(model, include_lora=False)
    lora_hash_after, _ = parameter_sha256(model, include_lora=True)
    lora_delta_l2, lora_delta_max = lora_delta(model, lora_before)
    if base_hash_before != base_hash_after:
        raise RuntimeError("base model changed during optimizer smoke")
    if lora_hash_before == lora_hash_after or lora_delta_l2 <= 0.0:
        raise RuntimeError("LoRA parameters did not update")
    sha_after = dataset_sha(data_dir)
    if sha_after != sha_before:
        raise RuntimeError("frozen data changed during trainer smoke")

    all_rewards = [float(value) for rollout in rollouts.values() for value in rollout["rewards"]]
    total_masked = sum(int(rollout["local_penalty_mask"].sum()) for rollout in rollouts.values())
    total_tokens = sum(int(rollout["completion_mask"].sum()) for rollout in rollouts.values())
    result = {
        "contract_version": "gr_user_trainer_correctness_smoke_v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "gpu": {"physical_id": args.physical_gpu_id, "preflight": gpu_state},
        "frozen_contract": {
            "G": G,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sqrt_base_lambda": 0.50,
            "epsilon": 0.20,
            "beta": 0.0,
            "scale_rewards": "group",
            "population_std_correction": 0,
        },
        "no_mask_parity": parity,
        "forward_only": {
            "passed": True,
            "prompt_count": 4,
            "G": G,
            "candidate_count": 16,
            "routes": forward_results,
            "reward_std": statistics.pstdev(all_rewards),
            "masked_token_fraction": total_masked / total_tokens,
            "peak_vram_mib": forward_peak_vram_mib,
            "wall_seconds": forward_wall,
        },
        "optimizer_smoke": {
            "passed": True,
            "actual_steps": args.optimizer_steps,
            "learning_rate": args.learning_rate,
            "steps": step_results,
            "base_tensor_count_hashed": base_tensor_count,
            "base_sha_before": base_hash_before,
            "base_sha_after": base_hash_after,
            "base_delta": 0.0,
            "lora_tensor_count": lora_tensor_count,
            "lora_sha_before": lora_hash_before,
            "lora_sha_after": lora_hash_after,
            "lora_delta_l2": lora_delta_l2,
            "lora_delta_max_abs": lora_delta_max,
            "nan_or_inf": not all(item["finite"] for item in step_results),
            "max_grad_norm": max(item["grad_norm"] for item in step_results),
            "max_clip_fraction": max(item["clip_fraction"] for item in step_results),
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
        },
        "local_gradient_contrast": contrast,
        "integrity": {
            "data_sha_before": sha_before,
            "data_sha_after": sha_after,
            "frozen_data_unchanged": sha_before == sha_after == EXPECTED_DATA_SHA,
            "formal_pilot_run": False,
            "full_training_run": False,
            "gr_rec_v1_modified": False,
        },
        "trainer_correctness": "PASS",
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "result": str(args.result_output), "status": "PASS"}))


if __name__ == "__main__":
    main()
