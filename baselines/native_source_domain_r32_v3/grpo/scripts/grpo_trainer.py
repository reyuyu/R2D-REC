# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1: GRPO Trainer core (inference rollout + clipped PPO updates).
Pure functions are CPU-testable; model-dependent parts run on GPU."""
import math
import torch
import torch.nn.functional as F

CLIP_EPS = 0.2
ADV_EPS = 1e-4
STD_FLOOR = 1e-6


# ---------------- group-relative advantages ----------------

def compute_group_advantages(rewards):
    """Population std normalization within one route/group.
    Returns advantages tensor; all zeros if std < STD_FLOOR."""
    r = torch.as_tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)  # population std
    if std < STD_FLOOR:
        return torch.zeros_like(r)
    return (r - mean) / (std + ADV_EPS)


# ---------------- clipped PPO objective ----------------

def ppo_clipped_loss(log_ratio, advantages, clip_eps=CLIP_EPS):
    """Token-level clipped objective.
    log_ratio: [B, L] or [L]; advantages: [B] (per action).
    Returns (loss, ratio_mean, ratio_p95, max_abs_log_ratio, clip_fraction)."""
    log_ratio = log_ratio.float()
    ratio = torch.exp(log_ratio)
    adv = advantages.float()
    if log_ratio.dim() == 2:
        adv = adv.unsqueeze(1)  # [B, 1]
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    token_loss = -torch.minimum(s1, s2)
    loss = token_loss.mean()
    with torch.no_grad():
        ratio_mean = ratio.mean().item()
        ratio_p95 = ratio.reshape(-1).float().quantile(0.95).item()
        max_abs_log_ratio = log_ratio.abs().max().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
    return loss, dict(
        ratio_mean=ratio_mean, ratio_p95=ratio_p95,
        max_abs_log_ratio=max_abs_log_ratio, clip_fraction=clip_fraction,
    )


def approx_old_policy_kl(log_ratio):
    """KL(policy_new || policy_old) monitor only: E[ratio - 1 - log_ratio]."""
    lr = log_ratio.float()
    ratio = torch.exp(lr)
    return (ratio - 1.0 - lr).mean().item()


# ---------------- action construction ----------------

THINK_CLOSE = "</think>"


def build_think_action_ids(tokenizer, cot_text):
    """Think action = completion tokens from response start up to and including
    </think>. Returns (action_ids, truncated_flag)."""
    idx = cot_text.find(THINK_CLOSE)
    if idx >= 0:
        cot = cot_text[: idx + len(THINK_CLOSE)]
        truncated = False
    else:
        cot = cot_text
        truncated = True
    ids = tokenizer.encode(cot, add_special_tokens=False)
    return ids, truncated


def build_nothink_action_ids(tokenizer, completion_text):
    """NoThink action = full completion tokens (malformed responses kept)."""
    ids = tokenizer.encode(completion_text, add_special_tokens=False)
    return ids


# ---------------- batched action logprobs ----------------

def action_logprobs(model, prompt_ids, action_ids_list):
    """Compute per-token logprobs of action tokens given prompt+action.
    prompt_ids: list[int] (shared prompt for all actions).
    Returns list of [action_len] fp32 logprob tensors (with grad if enabled)."""
    B = len(action_ids_list)
    seqs = [prompt_ids + a for a in action_ids_list]
    max_len = max(len(s) for s in seqs)
    device = next(model.parameters()).device
    input_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
    starts = []
    for i, s in enumerate(seqs):
        input_t[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :len(s)] = 1
        starts.append(len(prompt_ids))
    logits = model(input_ids=input_t, attention_mask=attn).logits  # [B, L, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)  # FP32
    out = []
    for i in range(B):
        s = starts[i]
        l = len(action_ids_list[i])
        if l == 0:
            out.append(torch.zeros(0, dtype=torch.float32, device=device))
            continue
        # predict action token j from position s-1+j
        lp = log_probs[i, s - 1: s + l - 1]  # [l, V]
        target = input_t[i, s: s + l].unsqueeze(-1)
        out.append(lp.gather(-1, target).squeeze(-1))
    return out


def log_ratio_between(logp_new_list, logp_old_list, lengths):
    """Concatenate per-action logprobs (variable length) into padded tensor [B, Lmax]
    with valid mask; returns (log_ratio, mask)."""
    B = len(logp_new_list)
    Lmax = max(lengths) if lengths else 0
    device = logp_new_list[0].device if logp_new_list else torch.device("cpu")
    if Lmax == 0:
        return torch.zeros(B, 1, device=device), torch.zeros(B, 1, dtype=torch.bool, device=device)
    lr = torch.zeros(B, Lmax, dtype=torch.float32, device=device)
    mask = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for i in range(B):
        n = lengths[i]
        if n == 0:
            continue
        lr[i, :n] = logp_new_list[i] - logp_old_list[i]
        mask[i, :n] = True
    return lr, mask


def token_ppo_losses(log_ratio, advantages, clip_eps=CLIP_EPS):
    """Token-level clipped objective preserving [B, L] shape.
    Returns -min(ratio*A, clip(ratio)*A) per token (pad positions will be masked
    by the caller; they carry no gradient since log_ratio there is constant 0)."""
    ratio = torch.exp(log_ratio.float())
    adv = advantages.float()
    if log_ratio.dim() == 2:
        adv = adv.unsqueeze(1)
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.minimum(s1, s2)


def flat_metrics(log_ratio_flat):
    """Metrics over valid log-ratios (1-D). Returns dict."""
    lr = log_ratio_flat.float()
    ratio = torch.exp(lr)
    with torch.no_grad():
        return dict(
            ratio_mean=ratio.mean().item(),
            ratio_p95=ratio.quantile(0.95).item(),
            max_abs_log_ratio=lr.abs().max().item(),
            clip_fraction=((ratio - 1.0).abs() > CLIP_EPS).float().mean().item(),
            kl=approx_old_policy_kl(lr),
        )


def masked_mean_loss(token_losses, mask):
    """L_rollout = mean over valid action tokens (token-length normalization)."""
    if mask.sum() == 0:
        return torch.zeros((), dtype=torch.float32, device=token_losses.device)
    return (token_losses * mask).sum() / mask.sum()


def route_loss_from_tokens(token_losses, mask):
    """L_route = mean over M actions of per-action token-mean loss.
    Guarantees each action contributes equally regardless of token count."""
    B = mask.shape[0]
    per_action = []
    for i in range(B):
        m = mask[i]
        if m.sum() == 0:
            per_action.append(torch.zeros((), dtype=torch.float32, device=token_losses.device))
        else:
            per_action.append((token_losses[i] * m).sum() / m.sum())
    return torch.stack(per_action).mean()
