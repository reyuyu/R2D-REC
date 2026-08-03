"""Controller-path microbenchmark; it does not run the 8B model or training data."""

import argparse
import gc
import json
import statistics
import time
from types import SimpleNamespace

import torch
from torch import nn

from llamafactory.train.sft.multitask_gradient_controller import MultiTaskGradientController


TASKS = ("material", "user", "recommendation")


class Holder(nn.Module):
    def __init__(self, numel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(numel, device="cuda"))


class FlatReferenceModel(nn.Module):
    def __init__(self, numel):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        block = self.model.layers[0]
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Module()
        block.self_attn.q_proj.lora_B = nn.ModuleDict({"default": Holder(numel)})

    def forward(self, feature):
        weight = self.model.layers[0].self_attn.q_proj.lora_B["default"].weight
        return torch.dot(weight, feature)


def make_args(ortho_enabled):
    return SimpleNamespace(
        multitask_gradnorm_tasks=list(TASKS),
        multitask_gradient_monitor_enabled=True,
        multitask_gradnorm_enabled=True,
        multitask_world_loss_weight=1.0,
        multitask_gradnorm_warmup_steps=10_000,
        multitask_gradnorm_update_interval=10,
        multitask_gradnorm_alpha=0.5,
        multitask_gradnorm_update_rate=0.1,
        multitask_gradnorm_loss_ema_beta=0.9,
        multitask_gradnorm_grad_ema_beta=0.9,
        multitask_gradnorm_weight_min=0.5,
        multitask_gradnorm_weight_max=2.0,
        multitask_gradnorm_step_ratio_min=0.9,
        multitask_gradnorm_step_ratio_max=1.1,
        multitask_grad_reference_last_n_layers=1,
        multitask_grad_reference_modules=["q_proj"],
        multitask_grad_reference_lora_matrix="B",
        multitask_ortho_enabled=ortho_enabled,
        multitask_ortho_monitor_only=False,
        multitask_ortho_start_step=0,
        multitask_ortho_interval=2,
        multitask_ortho_current_cosine_threshold=-0.05,
        multitask_ortho_ema_cosine_threshold=0.0,
        multitask_ortho_cosine_ema_beta=0.9,
        multitask_ortho_use_ema_gate=False,
        multitask_ortho_norm_ratio_min=0.7,
        multitask_ortho_norm_ratio_max=1.3,
        multitask_ortho_rotate_order=True,
    )


def vectors(numel):
    material = torch.zeros(numel, device="cuda")
    user = torch.zeros(numel, device="cuda")
    recommendation = torch.zeros(numel, device="cuda")
    material[0::3] = 1
    user[0::3] = -1
    user[1::3] = 1
    recommendation[1::3] = 1
    recommendation[2::3] = 1
    return dict(zip(TASKS, (material, user, recommendation)))


def one_step(model, controller, task_vectors, step):
    model.zero_grad(set_to_none=True)
    controller.begin_macro_step(step)
    for task in TASKS:
        controller.set_current_task(task)
        try:
            model(task_vectors[task]).div(3).backward()
        finally:
            controller.clear_current_task()
    controller.finish_macro_step(dict.fromkeys(TASKS, 1.0), dict.fromkeys(TASKS, 1))


def benchmark(ortho_enabled, numel, steps, warmup):
    model = FlatReferenceModel(numel)
    before_controller = torch.cuda.memory_allocated()
    controller = MultiTaskGradientController(model, make_args(ortho_enabled), loss_divisor=3)
    controller_bytes = torch.cuda.memory_allocated() - before_controller
    task_vectors = vectors(numel)
    for step in range(warmup):
        one_step(model, controller, task_vectors, step)
    durations = []
    active = []
    inactive = []
    project_times = []
    writeback_times = []
    for step in range(warmup, warmup + steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        one_step(model, controller, task_vectors, step)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        durations.append(elapsed_ms)
        (active if controller.ortho_step_due else inactive).append(elapsed_ms)
        if controller.ortho_step_due:
            project_times.append(controller.ortho_project_ms)
            writeback_times.append(controller.ortho_writeback_ms)
    return {
        "mean_ms": statistics.mean(durations),
        "ortho_active_mean_ms": statistics.mean(active) if active else None,
        "ortho_inactive_mean_ms": statistics.mean(inactive) if inactive else None,
        "ortho_project_mean_ms": statistics.mean(project_times) if project_times else None,
        "ortho_writeback_mean_ms": statistics.mean(writeback_times) if writeback_times else None,
        "controller_allocated_bytes": controller_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-numel", type=int, default=851_968)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: CUDA is required for timing and memory measurements")
        return
    torch.cuda.set_device(0)
    gradnorm_only = benchmark(False, args.reference_numel, args.steps, args.warmup)
    gc.collect()
    torch.cuda.empty_cache()
    gradnorm_ortho = benchmark(True, args.reference_numel, args.steps, args.warmup)
    result = {
        "scope": "synthetic controller path; excludes 8B forward, data loading, and optimizer",
        "reference_numel": args.reference_numel,
        "steps": args.steps,
        "gradnorm_only": gradnorm_only,
        "gradnorm_ortho": gradnorm_ortho,
        "extra_persistent_controller_bytes": (
            gradnorm_ortho["controller_allocated_bytes"] - gradnorm_only["controller_allocated_bytes"]
        ),
        "all_reduce_payload_bytes_per_capture": len(TASKS) * args.reference_numel * 4,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
