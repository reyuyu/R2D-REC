# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 TRL 4x A800 GRPO GPU smoke.
6 generation batches (Think x3 / NoThink x3 alternating) x num_iterations=2.
Run: torchrun --nproc_per_node=4 run_grpo_trl_smoke.py"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/GRPO/scripts")
import trl_import_fix  # noqa: F401
from grpo_trl_trainer import (RecGRPOTrainer, build_route_dataset,
                              make_nothink_reward_func, make_think_reward_func,
                              M_THINK, M_NO)
from grpo_sid import final_sid, think_reward
from grpo_model import load_model, encode_prompt, generate_batch

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
OUT_DIR = "/data/GRPO/outputs"
WORLD_CHUNK = 4  # per_device_train_batch_size
N_THINK_BATCHES = 3
N_NO_BATCHES = 3


def make_beam32_fn(model, tokenizer):
    """Beam32 on sampled CoT (eval + inference_mode), hierarchical reward.
    Re-decodes completions from raw token ids (TRL's decoded completions strip
    special tokens). Accumulates timing/hit/closure stats onto model._beam_stats."""
    stats = dict(beam_sec=0.0, n_cot=0, exact=0, ab=0, a=0, invalid=0, closure=0, n_closure_check=0)

    def beam32_fn(prompts, completions, completion_ids, gold_sets):
        was_training = model.training
        model.eval()
        out = []
        try:
            with torch.inference_mode():
                for prompt, cot_ids, gs in zip(prompts, completion_ids, gold_sets):
                    # rebuild completion text from raw ids (keeps </think> & SIDs)
                    cot = tokenizer.decode(cot_ids, skip_special_tokens=False)
                    stats["n_cot"] += 1
                    stats["n_closure_check"] += 1
                    if "</think>" in cot:
                        stats["closure"] += 1
                    idx = cot.find("</think>")
                    cot_trim = cot[: idx + len("</think>")] if idx >= 0 else cot
                    prompt_ids = encode_prompt(tokenizer, prompt)
                    cot_ids2 = tokenizer.encode(cot_trim, add_special_tokens=False)
                    t0 = time.time()
                    texts = generate_batch(model, tokenizer, [prompt_ids + cot_ids2],
                                           max_new_tokens=128, num_beams=32,
                                           num_return_sequences=32)
                    stats["beam_sec"] += time.time() - t0
                    beam_sids = [final_sid(t) for t in texts]
                    stats["invalid"] += sum(1 for s in beam_sids if s is None)
                    r, ec, ac, a2 = think_reward(beam_sids, gs)
                    stats["exact"] += ec
                    stats["ab"] += ac
                    stats["a"] += a2
                    out.append(r)
        finally:
            if was_training:
                model.train()
        model._beam_stats = stats
        return out
    return beam32_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--tag", default="REC-MP-GRPO-V1-TRL-SMOKE12")
    ap.add_argument("--check-only", action="store_true", help="asserts only, no train")
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    model, tokenizer, template = load_model(device)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"trainable tensors: {n_trainable}, params: {n_trainable_params}", flush=True)

    def lora_norm():
        tot = 0.0
        for n, p in model.named_parameters():
            if "lora" in n.lower():
                tot += (p.detach().float() ** 2).sum().item()
        return tot ** 0.5

    def base_checksum():
        for n, p in model.named_parameters():
            if "lora" not in n.lower():
                return float(p.detach().float().norm())
        return 0.0

    pre_lora = lora_norm()
    pre_base = base_checksum()

    # route dataset: segment size 8 -> with RepeatSampler chunking (4 idx/step
    # per rank at G=4), segments give T,T,N,N,T,T = 3 Think + 3 NoThink rollouts.
    chunk = 8
    n_groups = N_THINK_BATCHES * chunk  # 24 groups -> 24 think + 24 no_think records
    ds = build_route_dataset(DATA, n_groups=n_groups, seed=args.seed, chunk=chunk)
    if is_main:
        routes = [r["route"] for r in list(ds)]
        chunks = [routes[i:i + chunk] for i in range(0, len(routes), chunk)]
        print(f"dataset records: {len(ds)}, chunk={chunk}, "
              f"chunk routes: {[c[0] for c in chunks]}", flush=True)

    from trl import GRPOConfig
    cfg = GRPOConfig(
        output_dir=os.path.join(OUT_DIR, args.tag + "-" + time.strftime("%Y%m%d-%H%M%S")),
        per_device_train_batch_size=WORLD_CHUNK,
        gradient_accumulation_steps=1,
        num_generations=M_THINK,          # initial; per-batch switched to 4/8
        max_completion_length=2048,
        num_iterations=2,
        steps_per_generation=1,
        beta=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        disable_dropout=True,
        learning_rate=args.lr,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_vllm=False,
        temperature=1.0,
        top_p=1.0,
        seed=args.seed,
        generation_kwargs=None,
        shuffle_dataset=False,  # keep route-homogeneous alternating order across ranks
    )
    # Note: TRL transformers path does not forward generation_kwargs to generate;
    # Think CoT stop is handled post-hoc in beam32_fn (trim at </think>). Closure
    # rate is recorded from completions.
    cfg.generation_kwargs = None

    beam32_fn = make_beam32_fn(model, tokenizer)
    trainer = RecGRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_funcs=[
            make_nothink_reward_func(),
            make_think_reward_func(beam32_fn=beam32_fn),
        ],
    )
    if is_main:
        print("reward funcs:", trainer.reward_func_names, flush=True)
        gc = trainer.generation_config.to_dict()
        print("gen_config stop_strings:", gc.get("stop_strings"),
              "| num_return_sequences:", gc.get("num_return_sequences"),
              "| gen_kwargs:", trainer.args.generation_kwargs, flush=True)

    t0 = time.time()
    try:
        trainer.train()
    except Exception as e:
        print(f"[rank{rank}] train failed: {type(e).__name__}: {str(e)[:400]}", flush=True)
        raise

    total = time.time() - t0

    post_lora = lora_norm()
    post_base = base_checksum()
    history = trainer.state.log_history if hasattr(trainer.state, "log_history") else []
    # per-rank smoke log
    smoke_log = getattr(trainer, "_smoke_log", [])

    summary = dict(
        rank=rank,
        world=world,
        gpu=torch.cuda.get_device_name(device),
        max_steps=args.max_steps,
        lr=args.lr,
        trainable_params=n_trainable_params,
        lora_norm_pre=pre_lora,
        lora_norm_post=post_lora,
        lora_delta=abs(post_lora - pre_lora),
        base_norm_pre=pre_base,
        base_norm_post=post_base,
        base_delta=abs(post_base - pre_base),
        total_sec=total,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) // (1024 * 1024),
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) // (1024 * 1024),
        rollout_ids=smoke_log,
        log_history=history,
        beam_stats=getattr(model, "_beam_stats", None),
    )
    with open(f"{OUT_DIR}/{args.tag}-rank{rank}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if is_main:
        # summarize history (steps where loss logged)
        steps = [h for h in history if "loss" in h]
        print(f"\n=== SMOKE DONE rank0: {len(steps)} logged steps in {total:.0f}s ===", flush=True)
        for h in steps[-12:]:
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in h.items() if k in ("step", "loss", "grad_norm",
                                                             "clip_ratio/region_mean",
                                                             "rewards/nothink_reward/mean",
                                                             "rewards/think_reward/mean")},
                             ensure_ascii=False), flush=True)
        print(f"lora_delta={abs(post_lora - pre_lora):.6f} base_delta={abs(post_base - pre_base):.2e}", flush=True)


if __name__ == "__main__":
    main()
