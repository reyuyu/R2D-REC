# Experiment A1: think_prompt GradNorm

## Purpose

A1 is the rank16 Lagged GradNorm-lite experiment A with one data change: select the managed v1_thought_prompt dataset.
It also disables gradient checkpointing to measure the no-GC throughput/memory condition.

## Controlled Differences from A

| Item | A | A1 |
| --- | --- | --- |
| Dataset version | raw | v1_thought_prompt |
| Prompt suffix | raw, mixed | cot ends in /think; nocot ends in /no_think |
| Gradient checkpointing | original default | disabled |

All model, LoRA rank/alpha/dropout, learning rate, task mixture, balanced_40, GradNorm, global 8-microbatch semantics, seed, max_steps, optimizer and scheduler settings match experiment A.

## Config

configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_a1_think_prompt_nogc.yaml

    multitask_dataset_version: v1_thought_prompt
    multitask_dataset_version_overrides: {}
    disable_gradient_checkpointing: true

## Status

Launched on GPU 0/1 on 2026-08-04. The first macro-step is the no-GC feasibility gate. If it OOMs, this document retains that result and a selective-checkpoint follow-up uses a separate configuration.

## Checkpoint 评测结果

以下按物料四域、用户两项、推荐四域、world 的固定顺序记录。

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1500 | 1.1563 | 0.0434, 0.0380, 0.0434, 0.0413 | 0.1233, 0.0959 | 0.0933, 0.1326, 0.1694, 0.1449 | 0.2309 |
| 5200 | 1.1986 | 0.0454, 0.0379, 0.0460, 0.0424 | 0.1424, 0.0985 | 0.0859, 0.1530, 0.1792, 0.1341 | 0.2338 |

以上为用户提供的 checkpoint 评测结果。A1 在 5200 步完成，后续横向比较以相同评测口径为准。
