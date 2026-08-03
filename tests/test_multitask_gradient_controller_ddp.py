"""Two-GPU smoke test for task-gradient hooks and vector collectives.

Run with:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    tests/test_multitask_gradient_controller_ddp.py
"""

import os
import tempfile
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


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = nn.ModuleDict({"default": Holder()})


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = Projection()
        self.self_attn.v_proj = Projection()
        self.self_attn.o_proj = Projection()
        self.mlp = nn.Module()
        self.mlp.down_proj = Projection()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, feature):
        values = []
        for block in self.model.layers:
            values.extend(
                torch.dot(projection.lora_B["default"].weight, feature)
                for projection in (
                    block.self_attn.q_proj,
                    block.self_attn.v_proj,
                    block.self_attn.o_proj,
                    block.mlp.down_proj,
                )
            )
        return torch.stack(values).sum()


def make_args():
    return SimpleNamespace(
        multitask_gradnorm_tasks=list(TASKS),
        multitask_gradient_monitor_enabled=True,
        multitask_gradnorm_enabled=True,
        multitask_world_loss_weight=1.0,
        multitask_gradnorm_warmup_steps=0,
        multitask_gradnorm_update_interval=1,
        multitask_gradnorm_alpha=0.5,
        multitask_gradnorm_update_rate=0.1,
        multitask_gradnorm_loss_ema_beta=0.9,
        multitask_gradnorm_grad_ema_beta=0.9,
        multitask_gradnorm_weight_min=0.5,
        multitask_gradnorm_weight_max=2.0,
        multitask_gradnorm_step_ratio_min=0.9,
        multitask_gradnorm_step_ratio_max=1.1,
        multitask_grad_reference_last_n_layers=2,
        multitask_grad_reference_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        multitask_grad_reference_lora_matrix="B",
    )


def repeated(feature, parameter_count):
    return feature.repeat(parameter_count)


def main():
    if "RANK" not in os.environ:
        print("SKIP: run this file with torchrun --nproc_per_node=2")
        return
    if torch.cuda.device_count() < 2:
        print("SKIP: two visible CUDA devices are required")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This smoke test requires exactly two ranks."
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = Model().to(device)
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=3)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    local_tasks = (
        {
            "material": torch.tensor([1.0, 0.0], device=device),
            "user": torch.tensor([0.0, 1.0], device=device),
            "recommendation": torch.tensor([1.0, 1.0], device=device),
        }
        if rank == 0
        else {
            "material": torch.tensor([3.0, 0.0], device=device),
            "user": torch.tensor([0.0, 3.0], device=device),
            "world": torch.tensor([-1.0, -1.0], device=device),
        }
    )

    controller.begin_macro_step(1)
    for task, feature in local_tasks.items():
        controller.set_current_task(task)
        try:
            ddp_model(feature).div(3).backward()
        finally:
            controller.clear_current_task()
    local_sums = {task: float(rank + 1) for task in local_tasks if task in TASKS}
    local_counts = {task: 1 for task in local_tasks if task in TASKS}
    controller.finish_macro_step(local_sums, local_counts)

    parameter_count = len(controller.reference_parameters)
    expected_raw = {
        "material": repeated(torch.tensor([2.0, 0.0], device=device), parameter_count),
        "user": repeated(torch.tensor([0.0, 2.0], device=device), parameter_count),
        "recommendation": repeated(torch.tensor([1.0, 1.0], device=device), parameter_count),
    }
    for task in TASKS:
        assert torch.allclose(controller.raw_average_gradients[task], expected_raw[task], atol=1.0e-6, rtol=1.0e-6)

    selected_grad = torch.cat(
        [reference.parameter.grad.detach().reshape(-1) for reference in controller.reference_parameters]
    )
    expected_world = repeated(torch.tensor([-1.0, -1.0], device=device), parameter_count) / (3 * world_size)
    expected_final = sum(controller.actual_weighted_contributions.values()) + expected_world
    assert torch.allclose(selected_grad, expected_final, atol=1.0e-6, rtol=1.0e-6)

    checkpoint_holder = [tempfile.mkdtemp(prefix="gradnorm-ddp-smoke-") if rank == 0 else None]
    dist.broadcast_object_list(checkpoint_holder, src=0)
    checkpoint_dir = checkpoint_holder[0]
    controller.save_checkpoint(checkpoint_dir, is_main_process=rank == 0)
    dist.barrier()
    restored = MultiTaskGradientController(Model().to(device), make_args(), loss_divisor=3)
    assert restored.load_checkpoint(checkpoint_dir)
    assert restored.state_dict() == controller.state_dict()

    restored.begin_macro_step(2)
    weights = torch.tensor([restored.task_weights[task] for task in TASKS], device=device)
    gathered = [torch.empty_like(weights) for _ in range(world_size)]
    dist.all_gather(gathered, weights)
    assert all(torch.equal(gathered[0], other) for other in gathered[1:])
    dist.barrier()
    if rank == 0:
        os.remove(os.path.join(checkpoint_dir, MultiTaskGradientController.STATE_FILENAME))
        os.rmdir(checkpoint_dir)
        print("PASS: two-rank missing-task collectives, DDP hook scaling, and synchronized weights")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
