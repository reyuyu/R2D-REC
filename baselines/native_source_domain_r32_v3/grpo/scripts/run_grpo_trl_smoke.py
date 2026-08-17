# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 FINAL 4x A800 GRPO RESMOKE.
Real unique rollout sequence with n_groups=8 (one full cycle):
  T,T,N,N,N,N  = 2 Think rollouts + 4 NoThink rollouts
max_steps=12 with num_iterations=2 -> 12 optimizer steps (6 rollouts x 2).
Contract frozen: importance_sampling_level="token", top_entropy_quantile=1.0,
mask_truncated_completions=False, beta=0.0, loss_type="grpo", use_vllm=False,
max_prompt_length=8192, max_completion_length=2048.
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
N_GROUPS = 8     # one full T,T,N,N,N,N cycle (8 think + 8 no_think records)


def cached_prompt_ids(tokenizer, prompt, cache):
    """Encode a Beam prompt once within one reward-function invocation."""
    if prompt not in cache:
        cache[prompt] = encode_prompt(tokenizer, prompt)
    return cache[prompt]


def make_beam32_fn(model, tokenizer):
    """Beam32 on sampled CoT (eval + inference_mode), hierarchical reward.
    Re-decodes completions from raw token ids. Accumulates closed/no-close
    split stats (reward mean, beam invalid, beam wall) onto model._beam_stats.
    Wall time is THIS rank's beam time (4 ranks beam their own cots in
    parallel - never summed across ranks)."""
    stats = dict(
        beam_sec=0.0, n_cot=0, exact=0, ab=0, a=0, invalid=0,
        closure=0, n_closure_check=0,
        closed_n=0, closed_r_sum=0.0, closed_invalid=0, closed_beam_total=0,
        closed_beam_sec=0.0,
        noclose_n=0, noclose_r_sum=0.0, noclose_invalid=0, noclose_beam_total=0,
        noclose_beam_sec=0.0,
    )

    def beam32_fn(prompts, completions, completion_ids, gold_sets):
        # Reward-call scoped only: repeated G=4 completions share one prompt.
        prompt_ids_cache = {}
        was_training = model.training
        model.eval()
        out = []
        try:
            with torch.inference_mode():
                for prompt, cot_ids, gs in zip(prompts, completion_ids, gold_sets):
                    cot = tokenizer.decode(cot_ids, skip_special_tokens=False)
                    stats["n_cot"] += 1
                    stats["n_closure_check"] += 1
                    closed = "</think>" in cot
                    if closed:
                        stats["closure"] += 1
                    idx = cot.find("</think>")
                    cot_trim = cot[: idx + len("</think>")] if idx >= 0 else cot
                    prompt_ids = cached_prompt_ids(tokenizer, prompt, prompt_ids_cache)
                    cot_ids2 = tokenizer.encode(cot_trim, add_special_tokens=False)
                    t0 = time.time()
                    texts = generate_batch(model, tokenizer, [prompt_ids + cot_ids2],
                                           max_new_tokens=128, num_beams=32,
                                           num_return_sequences=32)
                    beam_t = time.time() - t0
                    stats["beam_sec"] += beam_t
                    beam_sids = [final_sid(t) for t in texts]
                    n_inv = sum(1 for s in beam_sids if s is None)
                    stats["invalid"] += n_inv
                    r, ec, ac, a2 = think_reward(beam_sids, gs)
                    stats["exact"] += ec
                    stats["ab"] += ac
                    stats["a"] += a2
                    out.append(r)
                    if closed:
                        stats["closed_n"] += 1
                        stats["closed_r_sum"] += r
                        stats["closed_invalid"] += n_inv
                        stats["closed_beam_total"] += len(beam_sids)
                        stats["closed_beam_sec"] += beam_t
                    else:
                        stats["noclose_n"] += 1
                        stats["noclose_r_sum"] += r
                        stats["noclose_invalid"] += n_inv
                        stats["noclose_beam_total"] += len(beam_sids)
                        stats["noclose_beam_sec"] += beam_t
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
    ap.add_argument("--tag", default="REC-MP-GRPO-V1-FINAL-SMOKE")
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
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"trainable params: {n_trainable_params}", flush=True)

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

    # one full unique cycle: T(2 chunks) + N(4 chunks) -> T,T,N,N,N,N
    chunk = 8
    ds = build_route_dataset(DATA, n_groups=N_GROUPS, seed=args.seed, chunk=chunk)
    if is_main:
        from grpo_trl_trainer import RouteAwareRepeatSampler
        _sam = RouteAwareRepeatSampler(ds, generation_batch_size=16, repeat_count=2)
        _seq = [_sam.data_source[i]["route"] for c in _sam._chunks for i in c[1][:1]]
        print(f"dataset records: {len(ds)}, chunks: {len(_sam._chunks)}, "
              f"unique rollout route sequence: {_seq}", flush=True)

    from trl import GRPOConfig
    cfg = GRPOConfig(
        output_dir=os.path.join(OUT_DIR, args.tag + "-" + time.strftime("%Y%m%d-%H%M%S")),
        per_device_train_batch_size=WORLD_CHUNK,
        gradient_accumulation_steps=1,
        num_generations=M_THINK,          # initial; per-batch switched to 4/8
        max_prompt_length=8192,           # explicit: SFT cutoff_len (no silent 512)
        max_completion_length=2048,
        num_iterations=2,
        steps_per_generation=1,
        beta=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        disable_dropout=True,
        # ---- frozen trainer contract ----
        importance_sampling_level="token",
        top_entropy_quantile=1.0,
        mask_truncated_completions=False,
        use_vllm=False,
        learning_rate=args.lr,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        temperature=1.0,
        top_p=1.0,
        seed=args.seed,
        generation_kwargs=None,
        shuffle_dataset=False,  # keep route-homogeneous alternating order across ranks
    )
    cfg.generation_kwargs = None

    beam32_fn = make_beam32_fn(model, tokenizer)
    trainer = RecGRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_funcs=[
            make_nothink_reward_func(tokenizer=tokenizer),
            make_think_reward_func(beam32_fn=beam32_fn),
        ],
    )
    # ---- frozen contract asserts ----
    assert cfg.importance_sampling_level == "token"
    assert cfg.top_entropy_quantile == 1.0
    assert cfg.mask_truncated_completions is False
    assert cfg.beta == 0.0
    assert cfg.loss_type == "grpo"
    assert cfg.use_vllm is False
    assert cfg.max_prompt_length == 8192
    assert cfg.max_completion_length == 2048
    if is_main:
        print("contract frozen: is=token q=1.0 mask=False beta=0 loss=grpo "
              "vllm=False max_prompt=8192 max_comp=2048", flush=True)
        print("reward funcs:", trainer.reward_func_names, flush=True)

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
    smoke_log = getattr(trainer, "_smoke_log", [])
    beam_stats = getattr(model, "_beam_stats", None)

    summary = dict(
        rank=rank, world=world,
        gpu=torch.cuda.get_device_name(device),
        max_steps=args.max_steps, lr=args.lr,
        trainable_params=n_trainable_params,
        lora_norm_pre=pre_lora, lora_norm_post=post_lora,
        lora_delta=abs(post_lora - pre_lora),
        base_norm_pre=pre_base, base_norm_post=post_base,
        base_delta=abs(post_base - pre_base),
        total_sec=total,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) // (1024 * 1024),
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) // (1024 * 1024),
        rollout_ids=smoke_log,
        log_history=history,
        beam_stats=beam_stats,
    )
    with open(f"{OUT_DIR}/{args.tag}-rank{rank}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if is_main:
        steps = [h for h in history if "loss" in h]
        print(f"\n=== SMOKE DONE rank0: {len(steps)} logged steps in {total:.0f}s ===", flush=True)
        for h in steps[-12:]:
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in h.items() if k in ("step", "epoch", "loss", "grad_norm",
                                                             "completions/mean_length",
                                                             "rewards/nothink_reward/mean",
                                                             "rewards/think_reward/mean")},
                             ensure_ascii=False), flush=True)
        print(f"lora_delta={abs(post_lora - pre_lora):.6f} base_delta={abs(post_base - pre_base):.2e}", flush=True)

        # ---- ETA with per-rollout wall times (never sum beam across ranks) ----
        if smoke_log and beam_stats:
            t_rollouts = [e for e in smoke_log if e.get("route") == "think" and "rollout_sec" in e]
            n_rollouts = [e for e in smoke_log if e.get("route") == "no_think" and "rollout_sec" in e]
            if t_rollouts and n_rollouts:
                mean_t = statistics.mean(e["rollout_sec"] for e in t_rollouts)
                mean_n = statistics.mean(e["rollout_sec"] for e in n_rollouts)
                pu = []
                for e in smoke_log:
                    for k in ("policy_fwb_sec_ep1", "policy_fwb_sec_ep2"):
                        if k in e:
                            pu.append(e[k])
                mean_pu = statistics.mean(pu) if pu else 0.0
                eta = 387 * mean_t + 774 * mean_n + 2322 * mean_pu
                print(f"\nETA: 387 x {mean_t:.1f}s + 774 x {mean_n:.1f}s + "
                      f"2322 x {mean_pu:.2f}s = {eta:.0f}s = {eta/3600:.1f}h", flush=True)
            bs = beam_stats
            print("closure:", bs.get("closure"), "/", bs.get("n_closure_check"),
                  f"({bs.get('closure', 0) / max(bs.get('n_closure_check', 1), 1) * 100:.1f}%)", flush=True)
            print("closed: n", bs.get("closed_n"), "reward_mean",
                  round(bs.get("closed_r_sum", 0.0) / max(bs.get("closed_n", 1), 1), 4),
                  "beam_invalid", f"{bs.get('closed_invalid', 0)}/{bs.get('closed_beam_total', 0)}",
                  f"beam_sec {bs.get('closed_beam_sec', 0.0):.1f}", flush=True)
            print("no-close: n", bs.get("noclose_n"), "reward_mean",
                  round(bs.get("noclose_r_sum", 0.0) / max(bs.get("noclose_n", 1), 1), 4),
                  "beam_invalid", f"{bs.get('noclose_invalid', 0)}/{bs.get('noclose_beam_total', 0)}",
                  f"beam_sec {bs.get('noclose_beam_sec', 0.0):.1f}", flush=True)
            print("beam total_sec (this rank):", round(bs.get("beam_sec", 0.0), 1), flush=True)
        for e in smoke_log:
            if "route" in e:
                print("rollout", e.get("rollout_id"), e.get("route"),
                      "gen", e.get("gen_wall_sec"), "total", e.get("rollout_sec"),
                      "plen", e.get("prompt_len_mean"), "/", e.get("prompt_len_p95"), "/", e.get("prompt_len_max"),
                      flush=True)


if __name__ == "__main__":
    main()
