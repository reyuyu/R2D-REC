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

