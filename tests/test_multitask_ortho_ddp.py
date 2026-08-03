"""Two-GPU end-to-end smoke test for capture, projection, writeback, and optimizer step.

Run with:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/test_multitask_ortho_ddp.py
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from llamafactory.train.sft.multitask_gradient_controller import MultiTaskGradientController


TASKS = ("material", "user", "recommendation")


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        block = self.model.layers[0]
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Module()
        block.self_attn.q_proj.lora_B = nn.ModuleDict({"default": Holder()})
        self.unselected = nn.Parameter(torch.zeros(2))

    def forward(self, feature):
        selected = self.model.layers[0].self_attn.q_proj.lora_B["default"].weight
        return torch.dot(selected, feature) + torch.dot(self.unselected, feature)


def make_args():
    return SimpleNamespace(
        multitask_gradnorm_tasks=list(TASKS),
        multitask_gradient_monitor_enabled=False,
        multitask_gradnorm_enabled=False,
        multitask_world_loss_weight=1.0,
        multitask_gradnorm_warmup_steps=0,
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
        multitask_ortho_enabled=True,
        multitask_ortho_monitor_only=False,
        multitask_ortho_start_step=0,
        multitask_ortho_interval=1,
        multitask_ortho_current_cosine_threshold=-0.05,
        multitask_ortho_ema_cosine_threshold=0.0,
        multitask_ortho_cosine_ema_beta=0.9,
        multitask_ortho_use_ema_gate=False,
        multitask_ortho_norm_ratio_min=0.7,
        multitask_ortho_norm_ratio_max=1.3,
        multitask_ortho_rotate_order=True,
    )


def main():
    if "RANK" not in os.environ:
        print("SKIP: run with torchrun --nproc_per_node=2")
        return
    if torch.cuda.device_count() < 2:
        print("SKIP: two visible CUDA devices are required")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = Model().to(device)
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=3)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    local_tasks = (
        (
            ("material", torch.tensor([1.0, 0.0], device=device)),
            ("user", torch.tensor([-1.0, 1.0], device=device)),
            ("recommendation", torch.tensor([0.0, 1.0], device=device)),
        )
        if rank == 0
        else (
            ("material", torch.tensor([1.0, 0.0], device=device)),
            ("user", torch.tensor([-1.0, 1.0], device=device)),
            ("world", torch.tensor([0.5, -0.5], device=device)),
        )
    )

    controller.begin_macro_step(0)
    for task, feature in local_tasks:
        controller.set_current_task(task)
        try:
            ddp_model(feature).div(3).backward()
        finally:
            controller.clear_current_task()
    local_sums = {task: 1.0 for task, _ in local_tasks if task in TASKS}
    local_counts = {task: 1 for task, _ in local_tasks if task in TASKS}
    unselected_before = model.unselected.grad.detach().clone()
    metrics = controller.finish_macro_step(local_sums, local_counts)

    assert metrics["mtg/ortho/active"] == 1
    assert metrics["mtg/ortho/pair/material_user"] == 1
    assert torch.equal(model.unselected.grad, unselected_before)
    selected_grad = controller.reference_parameters[0].parameter.grad.detach()
    projected_sum = sum(controller._ortho_buffers[f"projected_{task}"] for task in TASKS)
    expected_world_residual = torch.tensor([0.25, -0.25], device=device) / 3
    assert torch.allclose(selected_grad - projected_sum, expected_world_residual, atol=1.0e-6, rtol=0)

    gathered_grads = [torch.empty_like(selected_grad) for _ in range(2)]
    dist.all_gather(gathered_grads, selected_grad)
    assert torch.equal(gathered_grads[0], gathered_grads[1])
    optimizer.step()
    selected_parameter = controller.reference_parameters[0].parameter.detach()
    gathered_parameters = [torch.empty_like(selected_parameter) for _ in range(2)]
    dist.all_gather(gathered_parameters, selected_parameter)
    assert torch.equal(gathered_parameters[0], gathered_parameters[1])

    state = controller.state_dict()
    states = [None, None]
    dist.all_gather_object(states, state)
    assert states[0] == states[1]
    if rank == 0:
        print("PASS: DDP capture -> all_reduce -> cosine -> projection -> writeback -> optimizer.step")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
