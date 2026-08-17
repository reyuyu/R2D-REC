# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 4GPU GRPO smoke: 5 rollout batches x 2 policy updates.
Run with torchrun --nproc_per_node=4 run_grpo_smoke.py"""
import argparse
import json
import os
import statistics
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from grpo_sid import parse_sid, final_sid, q_reward, think_reward
from grpo_model import load_model, encode_prompt, generate_batch
from grpo_trainer import (compute_group_advantages,
                          approx_old_policy_kl, build_think_action_ids,
                          build_nothink_action_ids, action_logprobs,
                          route_loss_from_tokens, token_ppo_losses, flat_metrics,
                          CLIP_EPS)

DATA = "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
OUT_DIR = "/data/GRPO/outputs"
LR = 1e-6
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
M_THINK = 4
M_NO = 16


def load_data(path):
    import collections
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    return by_group


def gold_set(rec):
    out = set()
    for s in rec.get("all_gold_sids", []):
        t = parse_sid(s)
        if t:
            out.add(t)
    return out


def rollout_think(model, tokenizer, prompt_ids, golds):
    """Sample M_THINK CoT actions (no_grad inside generate_batch)."""
    texts = generate_batch(model, tokenizer, [prompt_ids], max_new_tokens=2048,
                           do_sample=True, temperature=0.9, top_p=0.95,
                           num_return_sequences=M_THINK)
    actions, truncated, cot_texts = [], [], []
    for t in texts:
        ids, tr = build_think_action_ids(tokenizer, t)
        actions.append(ids)
        truncated.append(tr)
        cot_texts.append(t)
    rewards = []
    per_cot = []
    for cot in cot_texts:
        idx = cot.find("</think>")
        if idx >= 0:
            cot = cot[: idx + len("</think>")]
        cot_ids = tokenizer.encode(cot, add_special_tokens=False)
        beam_texts = generate_batch(model, tokenizer, [prompt_ids + cot_ids],
                                    max_new_tokens=128, num_beams=32,
                                    num_return_sequences=32)
        beam_sids = [final_sid(t) for t in beam_texts]
        r, ec, ac, a2 = think_reward(beam_sids, golds)
        rewards.append(r)
        per_cot.append({"beam_sids": beam_sids, "exact": ec, "ab": ac, "a": a2})
    return actions, rewards, truncated, per_cot


def rollout_nothink(model, tokenizer, prompt_ids, golds):
    texts = generate_batch(model, tokenizer, [prompt_ids], max_new_tokens=128,
                           do_sample=True, temperature=1.0, top_p=1.0,
                           num_return_sequences=M_NO)
    actions, rewards, sids = [], [], []
    for t in texts:
        ids = build_nothink_action_ids(tokenizer, t)
        actions.append(ids)
        sid = final_sid(t)
        sids.append(sid)
        rewards.append(q_reward(sid, golds))
    return actions, rewards, sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--policy-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--tag", default="REC-MP-GRPO-V1-SMOKE10")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    model, tokenizer, template = load_model(device)

    # freeze base, train LoRA only
    n_trainable = 0
    for n, p in model.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
    if is_main:
        print(f"trainable LoRA params: {n_trainable}", flush=True)

    # parameter snapshots (pre)
    def lora_norm():
        tot = 0.0
        for n, p in model.named_parameters():
            if "lora" in n.lower():
                tot += (p.detach().float() ** 2).sum().item()
        return tot ** 0.5

    def base_norm():
        for n, p in model.named_parameters():
            if "lora" not in n.lower():
                return p.detach().float().norm().item()
        return 0.0

    pre_lora = lora_norm()
    pre_base = base_norm()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)

    # groups: 5 batches x 4 unique groups, fixed seed
    by_group = load_data(DATA)
    gids = sorted(by_group.keys())
    rng = __import__("random").Random(args.seed)
    rng.shuffle(gids)
    total_groups = args.rollouts * world
    chosen = [by_group[g] for g in gids[:total_groups]]

    all_log = []
    t_start = time.time()
    rollout_times = []
    step_times = []

    for rb in range(args.rollouts):
        recs = chosen[rb * world + rank]
        rec_think, rec_nothink = recs.get("think"), recs.get("no_think")
        assert rec_think and rec_nothink, f"group missing route on rank {rank}"
        golds = gold_set(rec_think)
        th_prompt_ids = encode_prompt(tokenizer, rec_think["prompt"])
        nt_prompt_ids = encode_prompt(tokenizer, rec_nothink["prompt"])

        t0 = time.time()
        # ---- rollout (eval + no_grad via generate_batch) ----
        th_actions, th_rewards, th_trunc, per_cot = rollout_think(
            model.module, tokenizer, th_prompt_ids, golds)
        nt_actions, nt_rewards, nt_sids = rollout_nothink(
            model.module, tokenizer, nt_prompt_ids, golds)
        dist.barrier()
        rollout_times.append(time.time() - t0)

        # ---- old logprobs (no_grad, same snapshot) ----
        with torch.no_grad():
            th_old = [action_logprobs(model.module, th_prompt_ids, [a])[0] for a in th_actions]
            nt_old = action_logprobs(model.module, nt_prompt_ids, nt_actions)
        torch.cuda.empty_cache()

        # ---- advantages (per route, group-relative) ----
        th_adv = compute_group_advantages(th_rewards)
        nt_adv = compute_group_advantages(nt_rewards)
        th_zero_std = float(torch.all(th_adv == 0))
        nt_zero_std = float(torch.all(nt_adv == 0))

        th_len = [len(a) for a in th_actions]
        nt_len = [len(a) for a in nt_actions]

        for epoch in range(args.policy_epochs):
            t1 = time.time()
            optimizer.zero_grad()
            # ---- Think forward/backward (per-action forward to bound memory) ----
            th_new = [action_logprobs(model.module, th_prompt_ids, [a])[0] for a in th_actions]
            Lmax_t = max(th_len) if th_len else 0
            th_lr = torch.zeros(len(th_actions), Lmax_t, dtype=torch.float32, device=device)
            th_mask = torch.zeros(len(th_actions), Lmax_t, dtype=torch.bool, device=device)
            for i in range(len(th_actions)):
                n = th_len[i]
                if n:
                    th_lr[i, :n] = th_new[i] - th_old[i]
                    th_mask[i, :n] = True
            th_valid_lr = th_lr[th_mask]
            if th_zero_std or th_valid_lr.numel() == 0:
                th_loss = torch.zeros((), dtype=torch.float32, device=device)
                th_metrics = dict(ratio_mean=1.0, ratio_p95=1.0, max_abs_log_ratio=0.0,
                                  clip_fraction=0.0, kl=0.0)
            else:
                th_tl = token_ppo_losses(th_lr, th_adv, CLIP_EPS).masked_fill(~th_mask, 0.0)
                th_loss = route_loss_from_tokens(th_tl, th_mask)
                th_metrics = flat_metrics(th_valid_lr)
            th_metrics["mean_action_tokens"] = statistics.mean(th_len) if th_len else 0.0
            (0.5 * th_loss).backward()
            del th_new, th_lr, th_mask
            torch.cuda.empty_cache()

            # ---- NoThink forward/backward ----
            nt_new = action_logprobs(model.module, nt_prompt_ids, nt_actions)
            Lmax_n = max(nt_len) if nt_len else 0
            nt_lr = torch.zeros(len(nt_actions), Lmax_n, dtype=torch.float32, device=device)
            nt_mask = torch.zeros(len(nt_actions), Lmax_n, dtype=torch.bool, device=device)
            for i in range(len(nt_actions)):
                n = nt_len[i]
                if n:
                    nt_lr[i, :n] = nt_new[i] - nt_old[i]
                    nt_mask[i, :n] = True
            nt_valid_lr = nt_lr[nt_mask]
            if nt_zero_std or nt_valid_lr.numel() == 0:
                nt_loss = torch.zeros((), dtype=torch.float32, device=device)
                nt_metrics = dict(ratio_mean=1.0, ratio_p95=1.0, max_abs_log_ratio=0.0,
                                  clip_fraction=0.0, kl=0.0)
            else:
                nt_tl = token_ppo_losses(nt_lr, nt_adv, CLIP_EPS).masked_fill(~nt_mask, 0.0)
                nt_loss = route_loss_from_tokens(nt_tl, nt_mask)
                nt_metrics = flat_metrics(nt_valid_lr)
            nt_metrics["mean_action_tokens"] = statistics.mean(nt_len) if nt_len else 0.0
            (0.5 * nt_loss).backward()
            del nt_new, nt_lr, nt_mask
            torch.cuda.empty_cache()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            step_times.append(time.time() - t1)

            # ---- epoch-1 ratio sanity ----
            if epoch == 0:
                rm = (th_metrics["ratio_mean"] + nt_metrics["ratio_mean"]) / 2
                cf = (th_metrics["clip_fraction"] + nt_metrics["clip_fraction"]) / 2
                if not (0.9 <= rm <= 1.1 and cf < 0.05):
                    if is_main:
                        print(f"BLOCKED: epoch1 ratio_mean={rm} clip={cf}", flush=True)
                    dist.destroy_process_group()
                    raise SystemExit(f"BLOCKED ratio epoch1 rm={rm} cf={cf}")

            log_entry = dict(
                rollout_batch=rb, optimizer_step=rb * args.policy_epochs + epoch,
                policy_epoch=epoch, lr=args.lr,
                grad_norm=float(grad_norm),
                think_reward_mean=statistics.mean(th_rewards),
                think_reward_std=statistics.pstdev(th_rewards) if len(th_rewards) > 1 else 0.0,
                think_zero_std=th_zero_std,
                think_advantage_mean=float(th_adv.mean()),
                think_advantage_std=float(th_adv.std(unbiased=False)),
                think_policy_loss=float(th_loss),
                think_ratio_mean=th_metrics["ratio_mean"],
                think_ratio_p95=th_metrics["ratio_p95"],
                think_max_abs_log_ratio=th_metrics["max_abs_log_ratio"],
                think_clip_fraction=th_metrics["clip_fraction"],
                think_approx_kl=th_metrics["kl"],
                think_mean_action_tokens=th_metrics["mean_action_tokens"],
                think_cot_truncated=sum(th_trunc),
                nothink_reward_mean=statistics.mean(nt_rewards),
                nothink_reward_std=statistics.pstdev(nt_rewards) if len(nt_rewards) > 1 else 0.0,
                nothink_zero_std=nt_zero_std,
                nothink_advantage_mean=float(nt_adv.mean()),
                nothink_advantage_std=float(nt_adv.std(unbiased=False)),
                nothink_policy_loss=float(nt_loss),
                nothink_ratio_mean=nt_metrics["ratio_mean"],
                nothink_ratio_p95=nt_metrics["ratio_p95"],
                nothink_max_abs_log_ratio=nt_metrics["max_abs_log_ratio"],
                nothink_clip_fraction=nt_metrics["clip_fraction"],
                nothink_approx_kl=nt_metrics["kl"],
                nothink_mean_action_tokens=nt_metrics["mean_action_tokens"],
                total_loss=float(th_loss + nt_loss),
            )
            if is_main:
                print(json.dumps(log_entry, ensure_ascii=False), flush=True)
            all_log.append(log_entry)

    dist.barrier()
    # ---- post checks ----
    post_lora = lora_norm()
    post_base = base_norm()
    if is_main:
        lora_delta = abs(post_lora - pre_lora)
        base_delta = abs(post_base - pre_base)
        summary = dict(
            rollout_batches=args.rollouts,
            optimizer_steps=args.rollouts * args.policy_epochs,
            policy_epochs=args.policy_epochs,
            lr=args.lr,
            trainable_params=n_trainable,
            lora_norm_pre=pre_lora, lora_norm_post=post_lora, lora_delta=lora_delta,
            base_norm_pre=pre_base, base_norm_post=post_base, base_delta=base_delta,
            base_unchanged=base_delta < 1e-6,
            lora_changed=lora_delta > 0,
            total_sec=time.time() - t_start,
            rollout_sec_mean=statistics.mean(rollout_times),
            step_sec_mean=statistics.mean(step_times),
            gpu=torch.cuda.get_device_name(device),
            peak_allocated_mb=torch.cuda.max_memory_allocated(device) // (1024 * 1024),
            log=all_log,
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = args.tag + "-" + time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(OUT_DIR, tag)
        os.makedirs(out_path, exist_ok=True)
        model.module.save_pretrained(out_path + "/adapter")
        tokenizer.save_pretrained(out_path + "/adapter")
        with open(os.path.join(out_path, "smoke_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_path, "SMOKE"), "w") as f:
            f.write("SMOKE checkpoint - not for formal training\n")
        print(f"SMOKE checkpoint saved: {out_path}", flush=True)
        print(f"lora_delta={lora_delta} base_delta={base_delta}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
