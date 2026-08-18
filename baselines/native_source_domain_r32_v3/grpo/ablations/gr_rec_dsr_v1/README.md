# GR_REC_DSR_Ablation_v1

Status: `IMPLEMENTED / SMOKE ONLY`

This directory is a removable, isolated ablation over the frozen
`GR_REC_v1` implementation in `/data/GRPO/scripts`.

## Isolation

- The baseline trainer, reward, Beam32 implementation, sampler, model loader,
  smoke runner, and formal runner are imported read-only.
- Primary rewards are returned unchanged. DSR wrappers only retain statistics
  that the baseline has already computed.
- `DsrGRPOTrainer` subclasses `RecGRPOTrainer`. The only copied production
  method is `_compute_loss`, because TRL exposes no post-logprob loss hook.
- Removing this directory and the DSR experiment document removes the ablation.

## Objectives

Think uses `lambda_T=0.10` and the baseline PPO ratio/clipping semantics with a
separately normalized auxiliary advantage. NoThink activates token-local
wrong-A unlikelihood only for true all-zero `G=8` groups. Neither path adds a
model forward, generation call, Beam call, or sampling change.

## Tests

```bash
cd /data/GRPO/scripts/ablations
PYTHONPATH=/data/GRPO/scripts:/data/GRPO/scripts/ablations \
  /data/venvs/llamafactory-01398eb-liger081/bin/python \
  -m gr_rec_dsr_v1.run_cpu_tests
```

## Smoke

```bash
cd /data/GRPO
torchrun --standalone --nproc_per_node=4 \
  scripts/ablations/gr_rec_dsr_v1/run_dsr_smoke.py --max-steps 12
```

The smoke runner rejects more than 12 optimizer steps. The separate future
pilot runner rejects checkpoint resume and requires a `GR-REC-DSR-V1-*` run ID.

## Formal-run contract

`run_dsr_train.py` fail-closes the fairness settings before delegating to the
frozen baseline runner. It forces train seed `20260816`, all 1,544 usable train
groups, four permanently excluded fixed probes in `video/living/prod/ad`
order, probe seed `20260818`, and probe interval 200. Resume is forbidden.
Think/NoThink group sizes and Beam32 generation settings remain owned by the
baseline implementation. The baseline probe evaluator snapshots and restores
Python, CPU Torch, and CUDA RNG state around every probe.

Run the deterministic CPU preflight and offline trace audits with:

```bash
cd /data/GRPO
PYTHONPATH=/data/GRPO/scripts:/data/GRPO/scripts/ablations \
  /data/venvs/llamafactory-01398eb-liger081/bin/python \
  -m gr_rec_dsr_v1.audit_dsr_preflight

PYTHONPATH=/data/GRPO/scripts:/data/GRPO/scripts/ablations \
  /data/venvs/llamafactory-01398eb-liger081/bin/python \
  -m gr_rec_dsr_v1.audit_offline_statistics
```

The real 8B+BATA gradient-budget audit is also zero-update: it creates no
optimizer and performs no `optimizer.step`. It uses four ranks to generate
stratified Think groups plus the exact fixed probes, then independently
backpropagates primary, coefficient-weighted auxiliary, and total objectives:

```bash
cd /data/GRPO
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo GRPO_BEAM_RANK_BALANCE=1 \
PYTHONPATH=/data/GRPO/scripts:/data/GRPO/scripts/ablations \
  torchrun --nproc_per_node=4 \
  scripts/ablations/gr_rec_dsr_v1/run_gradient_budget_audit.py
```

`run_nothink_rescue_probe.py` is an optional no-update diagnostic for two hard
groups observed as all-zero in baseline monitor traces. It is not imported by
training and reports the sampled s_A log probabilities needed for manual smoke
inspection, including the exact raw completion token IDs needed for a
same-rollout backward audit. It performs zero optimizer steps and must use the same loopback
NCCL/Gloo interface settings as the production launch scripts.
