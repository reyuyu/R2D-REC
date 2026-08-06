# Dual-rank smoke test for the Experiment E four-task layout.
# Runs under torchrun --nproc_per_node=2 (CPU/GLOO or GPU/NCCL):
#   cd /app/LLaMA-Factory
#   PYTHONPATH=src torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29555 \
#       tests/smoke_expE_4task_ddp.py
# Verifies the four-task balanced_40 loader, four-task GradNorm controller and
# DDP all-reduce agree on both ranks and that weights normalize to four.
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

from llamafactory.data.multitask import (
    TASK_LAYOUT_USER_SPLIT_NO_WORLD,
    Balanced40SuperCycle,
    MultiTaskMacroStepLoader,
    TaskDataLoader,
    TokenizedSubDataset,
    get_subtask_ratios,
    get_task_datasets,
    get_task_ids,
)
from llamafactory.train.sft.multitask_gradient_controller import MultiTaskGradientController


class TinyDataset(Dataset):
    def __init__(self, offset):
        self.offset = offset

    def __len__(self):
        return 32

    def __getitem__(self, index):
        length = 5 + ((self.offset + index) % 4)
        return {"input_ids": list(range(length)), "labels": [-100, 1] + [2] * (length - 2)}


class ToyLora(nn.Module):
    """Nested modules so parameter names look like
    model.layers.3.q_proj.lora_B (dots are legal via nn containers)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for layer in range(4):
            layer_mod = nn.Module()
            for module in ("q_proj", "v_proj", "o_proj", "down_proj"):
                sub = nn.Module()
                sub.lora_B = nn.Parameter(torch.randn(4, 4))
                layer_mod.add_module(module, sub)
            self.model.layers.append(layer_mod)

    def forward(self, feature):
        total = None
        for name, parameter in self.named_parameters():
            if name.endswith("lora_B"):
                contribution = (parameter @ feature).sum()
                total = contribution if total is None else total + contribution
        assert total is not None
        return total


def make_loaders(rank, world_size):
    task_ids = get_task_ids(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    task_datasets = get_task_datasets(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    ratios = get_subtask_ratios(TASK_LAYOUT_USER_SPLIT_NO_WORLD)
    loaders = {}
    subtask_id = 0
    for task_name, subtasks in task_datasets.items():
        wrapped = {}
        for subtask_name in subtasks:
            wrapped[subtask_name] = TokenizedSubDataset(
                TinyDataset(subtask_id),
                task_name,
                subtask_name,
                subtask_id,
                task_ids=task_ids,
            )
            subtask_id += 1
        loaders[task_name] = TaskDataLoader(
            task_name,
            wrapped,
            ratios[task_name],
            max_pack_length=64,
            max_segments=6,
            seed=11,
            rank=rank,
            world_size=world_size,
        )
    return loaders, task_ids


def main():
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    assert world_size == 2, "This smoke test expects exactly two ranks."
    torch.manual_seed(42 + rank)

    allocation = {"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
    loaders, task_ids = make_loaders(rank, world_size)
    loader = MultiTaskMacroStepLoader(
        loaders,
        allocation,
        max_steps=8,
        supercycle=Balanced40SuperCycle(seed=42, layout=TASK_LAYOUT_USER_SPLIT_NO_WORLD),
        rank=rank,
        world_size=world_size,
        synchronized_global_consumption=True,
        task_ids=task_ids,
    )

    model = ToyLora()
    controller_args = {
        "multitask_gradnorm_tasks": ["material", "user_action", "user_chain", "recommendation"],
        "multitask_gradient_monitor_enabled": True,
        "multitask_gradnorm_enabled": True,
        "multitask_world_loss_weight": 1.0,
        "multitask_gradnorm_warmup_steps": 2,
        "multitask_gradnorm_update_interval": 1,
        "multitask_gradnorm_alpha": 0.5,
        "multitask_gradnorm_update_rate": 1.0,
        "multitask_gradnorm_loss_ema_beta": 0.5,
        "multitask_gradnorm_grad_ema_beta": 0.5,
        "multitask_gradnorm_weight_min": 0.5,
        "multitask_gradnorm_weight_max": 2.0,
        "multitask_gradnorm_step_ratio_min": 0.9,
        "multitask_gradnorm_step_ratio_max": 1.1,
        "multitask_grad_reference_last_n_layers": 2,
        "multitask_grad_reference_modules": ["q_proj", "v_proj", "o_proj", "down_proj"],
        "multitask_grad_reference_lora_matrix": "B",
    }
    from types import SimpleNamespace

    controller = MultiTaskGradientController(model, SimpleNamespace(**controller_args), loss_divisor=4)
    assert set(controller.tasks) == set(task_ids)

    global_totals = {task: 0 for task in task_ids}
    for step in range(1, 9):
        macro = next(loader)
        local_count = sum(len(items) for items in macro.values())
        assert local_count == 4, (rank, step, local_count)
        controller.begin_macro_step(step)
        loss_sums, counts = {}, {}
        for task, microbatches in macro.items():
            count = len(microbatches)
            counts[task] = count
            loss = model(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32))
            controller.set_current_task(task)
            try:
                (loss * controller.task_weight(task) / 4.0).backward()
            finally:
                controller.clear_current_task()
            loss_sums[task] = float(loss.detach())
        controller.finish_macro_step(loss_sums, counts)
        for task in task_ids:
            global_totals[task] += counts.get(task, 0)
        model.zero_grad()

    weights = [controller.task_weights[task] for task in controller.tasks]
    gathered = [torch.tensor(weights, dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(gathered, gathered[rank])
    for other in gathered:
        assert torch.allclose(gathered[rank], other, atol=1.0e-9, rtol=1.0e-9), (rank, gathered)
    total_weight = float(gathered[rank].sum().item())
    assert math.isclose(total_weight, 4.0, abs_tol=1.0e-8), total_weight
    assert all(value >= 0.5 - 1.0e-9 and value <= 2.0 + 1.0e-9 for value in weights)
    assert global_totals == {"material": 8, "user_action": 8, "user_chain": 8, "recommendation": 8}, global_totals
    if rank == 0:
        print("SMOKE PASS", {"weights": [round(float(v), 6) for v in weights], "global_totals": global_totals})


if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    main()
    dist.destroy_process_group()
