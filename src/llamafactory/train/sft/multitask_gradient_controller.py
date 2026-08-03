# Copyright 2025 the LlamaFactory team.

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import torch

from ...extras import logging


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class _ReferenceParameter:
    name: str
    parameter: torch.nn.Parameter
    offset: int
    numel: int
    layer: int
    module: str


class MultiTaskGradientController:
    """Optional lagged GradNorm-lite controller with local LoRA gradient projection."""

    STATE_FILENAME = "multitask_gradient_controller.json"
    _LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def __init__(self, model: torch.nn.Module, data_args: Any, loss_divisor: float) -> None:
        self.tasks = tuple(data_args.multitask_gradnorm_tasks)
        self.monitor_enabled = bool(data_args.multitask_gradient_monitor_enabled)
        self.gradnorm_enabled = bool(data_args.multitask_gradnorm_enabled)
        self.world_loss_weight = float(data_args.multitask_world_loss_weight)
        self.loss_divisor = float(loss_divisor)
        self.warmup_steps = int(data_args.multitask_gradnorm_warmup_steps)
        self.update_interval = int(data_args.multitask_gradnorm_update_interval)
        self.alpha = float(data_args.multitask_gradnorm_alpha)
        self.update_rate = float(data_args.multitask_gradnorm_update_rate)
        self.loss_ema_beta = float(data_args.multitask_gradnorm_loss_ema_beta)
        self.grad_ema_beta = float(data_args.multitask_gradnorm_grad_ema_beta)
        self.weight_min = float(data_args.multitask_gradnorm_weight_min)
        self.weight_max = float(data_args.multitask_gradnorm_weight_max)
        self.step_ratio_min = float(data_args.multitask_gradnorm_step_ratio_min)
        self.step_ratio_max = float(data_args.multitask_gradnorm_step_ratio_max)
        self.ortho_enabled = bool(getattr(data_args, "multitask_ortho_enabled", False))
        self.ortho_monitor_only = bool(getattr(data_args, "multitask_ortho_monitor_only", False))
        self.ortho_start_step = int(getattr(data_args, "multitask_ortho_start_step", 600))
        self.ortho_interval = int(getattr(data_args, "multitask_ortho_interval", 2))
        self.ortho_current_cosine_threshold = float(
            getattr(data_args, "multitask_ortho_current_cosine_threshold", -0.05)
        )
        self.ortho_ema_cosine_threshold = float(getattr(data_args, "multitask_ortho_ema_cosine_threshold", 0.0))
        self.ortho_cosine_ema_beta = float(getattr(data_args, "multitask_ortho_cosine_ema_beta", 0.90))
        self.ortho_use_ema_gate = bool(getattr(data_args, "multitask_ortho_use_ema_gate", True))
        self.ortho_norm_ratio_min = float(getattr(data_args, "multitask_ortho_norm_ratio_min", 0.7))
        self.ortho_norm_ratio_max = float(getattr(data_args, "multitask_ortho_norm_ratio_max", 1.3))
        self.ortho_rotate_order = bool(getattr(data_args, "multitask_ortho_rotate_order", True))
        self.eps = 1.0e-12

        self.task_weights = dict.fromkeys(self.tasks, 1.0)
        self.pending_weights: dict[str, float] | None = None
        self.loss_ema: dict[str, float] = {}
        self.grad_norm_ema: dict[str, float] = {}
        self.initial_loss_baseline: dict[str, float] = {}
        self.baseline_initialized = False
        self.measurement_count = 0
        self.weight_update_count = 0
        self.last_cosines: dict[str, float | None] = {}
        self.last_raw_norms: dict[str, float] = {}
        self.last_effective_norms: dict[str, float] = {}
        self.actual_weighted_contributions: dict[str, torch.Tensor] = {}
        self.raw_average_gradients: dict[str, torch.Tensor] = {}
        self.cosine_ema: dict[str, float] = {}
        self.ortho_activation_state = False
        self.ortho_execution_count = 0
        self.ortho_pair_counts = {
            f"{left}_{right}": 0
            for left_index, left in enumerate(self.tasks)
            for right in self.tasks[left_index + 1 :]
        }
        self.last_removed_ratio = 0.0
        self.last_tracked_norm_ratio = 1.0
        self.last_total_norm_ratio = 1.0
        self.last_ortho_pairs = dict.fromkeys(self.ortho_pair_counts, 0)

        self.capture_enabled = False
        self.current_task: str | None = None
        self.current_macro_step = 0
        self.measure_active = False
        self.gradnorm_measure_due = False
        self.ortho_step_due = False
        self.weight_update_active = False
        self.max_weight_delta = 0.0
        self.capture_ms = 0.0
        self.sync_ms = 0.0
        self.update_ms = 0.0
        self.ortho_project_ms = 0.0
        self.ortho_writeback_ms = 0.0
        self._capture_cpu_seconds = 0.0
        self._capture_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._hooks_registered = False
        self._hook_handles: list[Any] = []

        self.reference_parameters = self.select_reference_parameters(
            model,
            last_n_layers=int(data_args.multitask_grad_reference_last_n_layers),
            module_names=tuple(data_args.multitask_grad_reference_modules),
            lora_matrix=str(data_args.multitask_grad_reference_lora_matrix),
        )
        self.reference_numel = sum(item.numel for item in self.reference_parameters)
        device = self.reference_parameters[0].parameter.device
        self.flat_buffers = {
            task: torch.zeros(self.reference_numel, dtype=torch.float32, device=device) for task in self.tasks
        }
        self._ortho_buffers: dict[str, torch.Tensor] = {}
        if self.ortho_enabled:
            self._ortho_buffers = {
                **{
                    f"projected_{task}": torch.zeros(self.reference_numel, dtype=torch.float32, device=device)
                    for task in self.tasks
                },
                "original_total": torch.zeros(self.reference_numel, dtype=torch.float32, device=device),
                "original_tracked": torch.zeros(self.reference_numel, dtype=torch.float32, device=device),
                "projected_tracked": torch.zeros(self.reference_numel, dtype=torch.float32, device=device),
                "new_total": torch.zeros(self.reference_numel, dtype=torch.float32, device=device),
            }
        if self.monitor_enabled or self.ortho_enabled:
            self.register_hooks()

        layers = sorted({item.layer for item in self.reference_parameters})
        modules = sorted({item.module for item in self.reference_parameters})
        logger.info_rank0(
            "Multitask gradient reference selection:\n"
            f"Layer count: {len(layers)} ({layers})\n"
            f"Parameter tensors: {len(self.reference_parameters)}\n"
            f"Parameter elements: {self.reference_numel}\n"
            f"Modules: {modules}"
        )

    @classmethod
    def select_reference_parameters(
        cls,
        model: torch.nn.Module,
        last_n_layers: int,
        module_names: tuple[str, ...],
        lora_matrix: str = "B",
    ) -> list[_ReferenceParameter]:
        if last_n_layers <= 0:
            raise ValueError("multitask_grad_reference_last_n_layers must be positive.")
        matrix_token = f"lora_{lora_matrix.upper()}"
        module_pattern = re.compile(
            rf"(?:^|\.)({'|'.join(re.escape(name) for name in module_names)})\.{matrix_token}(?:\.|$)"
        )
        candidates: list[tuple[str, torch.nn.Parameter, int, str]] = []
        lora_candidates: list[str] = []
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                lora_candidates.append(name)
            layer_match = cls._LAYER_PATTERN.search(name)
            module_match = module_pattern.search(name)
            if parameter.requires_grad and layer_match and module_match:
                candidates.append((name, parameter, int(layer_match.group(1)), module_match.group(1)))

        if candidates:
            selected_layers = sorted({item[2] for item in candidates})[-last_n_layers:]
            candidates = [item for item in candidates if item[2] in selected_layers]
        if not candidates:
            candidate_text = "\n".join(lora_candidates) if lora_candidates else "<none>"
            raise ValueError(
                "No trainable LoRA reference parameters matched the configured layers/modules/matrix. "
                f"Candidate LoRA parameters:\n{candidate_text}"
            )

        references: list[_ReferenceParameter] = []
        offset = 0
        for name, parameter, layer, module in candidates:
            references.append(_ReferenceParameter(name, parameter, offset, parameter.numel(), layer, module))
            offset += parameter.numel()
        return references

    def register_hooks(self) -> None:
        if self._hooks_registered:
            return
        for reference in self.reference_parameters:
            self._hook_handles.append(reference.parameter.register_hook(self._make_hook(reference)))
        self._hooks_registered = True

    def _make_hook(self, reference: _ReferenceParameter):
        def capture_hook(gradient: torch.Tensor) -> torch.Tensor:
            task = self.current_task
            if not self.capture_enabled or task not in self.flat_buffers:
                return gradient
            cpu_started = time.perf_counter()
            cuda_events = None
            if gradient.is_cuda:
                cuda_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                cuda_events[0].record()
            with torch.no_grad():
                self.flat_buffers[task].narrow(0, reference.offset, reference.numel).add_(
                    gradient.detach().reshape(-1).to(dtype=torch.float32)
                )
            if cuda_events is not None:
                cuda_events[1].record()
                self._capture_cuda_events.append(cuda_events)
            else:
                self._capture_cpu_seconds += time.perf_counter() - cpu_started
            return gradient

        return capture_hook

    def begin_macro_step(self, macro_step: int) -> None:
        if self.pending_weights is not None:
            self.task_weights = dict(self.pending_weights)
            self.pending_weights = None
        self.current_macro_step = int(macro_step)
        self.gradnorm_measure_due = self.gradnorm_enabled and macro_step % self.update_interval == 0
        monitor_due = self.monitor_enabled and macro_step % self.update_interval == 0
        self.measure_active = self.gradnorm_measure_due or monitor_due
        self.ortho_step_due = (
            self.ortho_enabled
            and macro_step >= self.ortho_start_step
            and macro_step % self.ortho_interval == 0
        )
        self.capture_enabled = self.measure_active or self.ortho_step_due
        self.current_task = None
        self.weight_update_active = False
        self.max_weight_delta = 0.0
        self.capture_ms = 0.0
        self.sync_ms = 0.0
        self.update_ms = 0.0
        self.ortho_project_ms = 0.0
        self.ortho_writeback_ms = 0.0
        self.ortho_activation_state = self.ortho_step_due
        self.last_removed_ratio = 0.0
        self.last_tracked_norm_ratio = 1.0
        self.last_total_norm_ratio = 1.0
        self.last_ortho_pairs = dict.fromkeys(self.ortho_pair_counts, 0)
        self._capture_cpu_seconds = 0.0
        self._capture_cuda_events.clear()
        if self.capture_enabled:
            for buffer in self.flat_buffers.values():
                buffer.zero_()

    def task_weight(self, task: str) -> float:
        if task == "world":
            return self.world_loss_weight
        return self.task_weights.get(task, 1.0)

    def set_current_task(self, task: str) -> None:
        self.current_task = task if self.capture_enabled and task in self.tasks else None

    def clear_current_task(self) -> None:
        self.current_task = None

    @staticmethod
    def restore_raw_average_gradient(
        averaged_weighted_contribution: torch.Tensor,
        loss_divisor: float,
        world_size: int,
        task_weight: float,
        global_microbatch_count: int,
    ) -> torch.Tensor:
        if global_microbatch_count <= 0 or task_weight == 0:
            return torch.zeros_like(averaged_weighted_contribution)
        scale = loss_divisor * world_size / (task_weight * global_microbatch_count)
        return averaged_weighted_contribution * scale

    def _distributed_world_size(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def _all_reduce_sum(self, tensor: torch.Tensor) -> None:
        if self._distributed_world_size() > 1:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    def _finish_capture_timing(self) -> None:
        if self._capture_cuda_events:
            torch.cuda.synchronize(self.flat_buffers[self.tasks[0]].device)
            self.capture_ms = sum(start.elapsed_time(end) for start, end in self._capture_cuda_events)
        else:
            self.capture_ms = self._capture_cpu_seconds * 1000.0

    def finish_macro_step(self, local_loss_sums: dict[str, float], local_counts: dict[str, int]) -> dict[str, float]:
        self.clear_current_task()
        capture_was_active = self.capture_enabled
        self.capture_enabled = False
        self._finish_capture_timing()
        device = self.flat_buffers[self.tasks[0]].device
        packed_stats = torch.zeros(2 * len(self.tasks), dtype=torch.float64, device=device)
        for index, task in enumerate(self.tasks):
            packed_stats[2 * index] = float(local_loss_sums.get(task, 0.0))
            packed_stats[2 * index + 1] = int(local_counts.get(task, 0))

        sync_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._all_reduce_sum(packed_stats)
        world_size = self._distributed_world_size()
        global_counts: dict[str, int] = {}
        for index, task in enumerate(self.tasks):
            loss_sum = float(packed_stats[2 * index].item())
            count = int(packed_stats[2 * index + 1].item())
            global_counts[task] = count
            if count > 0:
                current_loss = loss_sum / count
                previous = self.loss_ema.get(task)
                self.loss_ema[task] = (
                    current_loss if previous is None else self.loss_ema_beta * previous + (1.0 - self.loss_ema_beta) * current_loss
                )

        if capture_was_active:
            if self.measure_active:
                self.measurement_count += 1
            self.actual_weighted_contributions = {}
            self.raw_average_gradients = {}
            for task in self.tasks:
                contribution = self.flat_buffers[task].clone()
                self._all_reduce_sum(contribution)
                contribution.div_(world_size)
                self.actual_weighted_contributions[task] = contribution
                self.raw_average_gradients[task] = self.restore_raw_average_gradient(
                    contribution,
                    self.loss_divisor,
                    world_size,
                    self.task_weights[task],
                    global_counts[task],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        self.sync_ms = (time.perf_counter() - sync_started) * 1000.0

        update_started = time.perf_counter()
        if capture_was_active:
            self._update_gradient_statistics(global_counts, update_grad_ema=self.measure_active)
        if not self.baseline_initialized and self.current_macro_step >= self.warmup_steps:
            missing = [task for task in self.tasks if task not in self.loss_ema]
            if not missing:
                self.initial_loss_baseline = dict(self.loss_ema)
                self.baseline_initialized = True
        if (
            self.gradnorm_enabled
            and self.measure_active
            and self.baseline_initialized
            and self.current_macro_step > self.warmup_steps
            and all(task in self.grad_norm_ema for task in self.tasks)
            and all(global_counts[task] > 0 for task in self.tasks)
        ):
            self.pending_weights = self._compute_pending_weights()
            self.weight_update_active = True
            self.weight_update_count += 1
            self.max_weight_delta = max(
                abs(self.pending_weights[task] - self.task_weights[task]) for task in self.tasks
            )
        if self.ortho_step_due:
            self._apply_ortho_projection(global_counts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.update_ms = (time.perf_counter() - update_started) * 1000.0
        return self.metrics()

    def _update_gradient_statistics(self, global_counts: dict[str, int], update_grad_ema: bool) -> None:
        valid_tasks: set[str] = set()
        for task in self.tasks:
            gradient = self.raw_average_gradients[task]
            norm = float(torch.linalg.vector_norm(gradient).item())
            if global_counts[task] > 0 and math.isfinite(norm):
                valid_tasks.add(task)
                self.last_raw_norms[task] = norm
                self.last_effective_norms[task] = self.task_weights[task] * norm
                if update_grad_ema:
                    previous = self.grad_norm_ema.get(task)
                    self.grad_norm_ema[task] = (
                        norm if previous is None else self.grad_ema_beta * previous + (1.0 - self.grad_ema_beta) * norm
                    )

        self.last_cosines = {}
        for left_index, left in enumerate(self.tasks):
            for right in self.tasks[left_index + 1 :]:
                key = f"{left}_{right}"
                left_norm = self.last_raw_norms.get(left, 0.0)
                right_norm = self.last_raw_norms.get(right, 0.0)
                if left not in valid_tasks or right not in valid_tasks or left_norm <= 0 or right_norm <= 0:
                    self.last_cosines[key] = None
                    continue
                cosine = float(
                    torch.dot(self.raw_average_gradients[left], self.raw_average_gradients[right]).item()
                    / (left_norm * right_norm)
                )
                self.last_cosines[key] = cosine if math.isfinite(cosine) else None
                if self.ortho_enabled and self.last_cosines[key] is not None:
                    previous = self.cosine_ema.get(key)
                    self.cosine_ema[key] = (
                        self.last_cosines[key]
                        if previous is None
                        else self.ortho_cosine_ema_beta * previous
                        + (1.0 - self.ortho_cosine_ema_beta) * self.last_cosines[key]
                    )

    def _pair_key(self, left: str, right: str) -> str:
        left_index = self.tasks.index(left)
        right_index = self.tasks.index(right)
        return f"{left}_{right}" if left_index < right_index else f"{right}_{left}"

    def _ortho_order(self) -> tuple[str, ...]:
        base_order = tuple(task for task in ("material", "user", "recommendation") if task in self.tasks)
        if not self.ortho_rotate_order:
            return base_order
        shift = self.current_macro_step % len(base_order)
        return base_order[shift:] + base_order[:shift]

    def _pair_passes_gate(self, key: str) -> bool:
        current = self.last_cosines.get(key)
        if (
            current is None
            or not math.isfinite(current)
            or current >= self.ortho_current_cosine_threshold
        ):
            return False
        left, right = key.split("_", maxsplit=1)
        if self.last_raw_norms.get(left, 0.0) <= self.eps or self.last_raw_norms.get(right, 0.0) <= self.eps:
            return False
        if self.ortho_use_ema_gate:
            ema = self.cosine_ema.get(key)
            return ema is not None and math.isfinite(ema) and ema < self.ortho_ema_cosine_threshold
        return True

    def _flatten_reference_gradients(self, destination: torch.Tensor) -> int:
        destination.zero_()
        missing = 0
        for reference in self.reference_parameters:
            gradient = reference.parameter.grad
            if gradient is None:
                missing += 1
                continue
            destination.narrow(0, reference.offset, reference.numel).copy_(
                gradient.detach().reshape(-1).to(dtype=torch.float32)
            )
        return missing

    def _writeback_reference_gradients(self, source: torch.Tensor) -> int:
        missing = 0
        with torch.no_grad():
            for reference in self.reference_parameters:
                gradient = reference.parameter.grad
                if gradient is None:
                    missing += 1
                    continue
                value = source.narrow(0, reference.offset, reference.numel).view_as(gradient)
                gradient.copy_(value.to(dtype=gradient.dtype))
        return missing

    def _apply_ortho_projection(self, global_counts: dict[str, int]) -> None:
        self.ortho_execution_count += 1
        device = self.flat_buffers[self.tasks[0]].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        project_started = time.perf_counter()

        gated_pairs: set[str] = set()
        for key in self.ortho_pair_counts:
            if self._pair_passes_gate(key):
                left, right = key.split("_", maxsplit=1)
                if global_counts.get(left, 0) > 0 and global_counts.get(right, 0) > 0:
                    gated_pairs.add(key)
                    self.last_ortho_pairs[key] = 1
                    self.ortho_pair_counts[key] += 1

        projected = {task: self._ortho_buffers[f"projected_{task}"] for task in self.tasks}
        for task in self.tasks:
            projected[task].copy_(self.actual_weighted_contributions[task])

        original_tracked = self._ortho_buffers["original_tracked"]
        original_tracked.zero_()
        for task in self.tasks:
            original_tracked.add_(self.actual_weighted_contributions[task])
        original_norm = float(torch.linalg.vector_norm(original_tracked).item())
        projection_allowed = bool(gated_pairs) and math.isfinite(original_norm) and original_norm > self.eps

        if projection_allowed:
            order = self._ortho_order()
            for task in order:
                current = projected[task]
                for reference_task in order:
                    if task == reference_task or self._pair_key(task, reference_task) not in gated_pairs:
                        continue
                    reference = self.actual_weighted_contributions[reference_task]
                    reference_norm_sq = torch.dot(reference, reference)
                    if not torch.isfinite(reference_norm_sq) or float(reference_norm_sq.item()) <= self.eps:
                        continue
                    dot = torch.dot(current, reference)
                    if torch.isfinite(dot) and float(dot.item()) < 0:
                        current.add_(reference, alpha=-float((dot / (reference_norm_sq + self.eps)).item()))

        projected_tracked = self._ortho_buffers["projected_tracked"]
        projected_tracked.zero_()
        for task in self.tasks:
            projected_tracked.add_(projected[task])

        projected_norm = float(torch.linalg.vector_norm(projected_tracked).item())
        if projection_allowed:
            ratio = projected_norm / original_norm if math.isfinite(projected_norm) else 0.0
            target_ratio = min(max(ratio, self.ortho_norm_ratio_min), self.ortho_norm_ratio_max)
            if projected_norm > self.eps and math.isfinite(projected_norm):
                scale = target_ratio / max(ratio, self.eps)
                if abs(scale - 1.0) > 1.0e-12:
                    projected_tracked.zero_()
                    for task in self.tasks:
                        projected[task].mul_(scale)
                        projected_tracked.add_(projected[task])
            elif self.ortho_norm_ratio_min > 0:
                for task in self.tasks:
                    projected[task].copy_(self.actual_weighted_contributions[task])
                projected_tracked.copy_(original_tracked)

        projected_norm = float(torch.linalg.vector_norm(projected_tracked).item())
        if original_norm > self.eps and math.isfinite(original_norm) and math.isfinite(projected_norm):
            self.last_tracked_norm_ratio = projected_norm / original_norm
            difference = self._ortho_buffers["new_total"]
            difference.copy_(original_tracked).sub_(projected_tracked)
            self.last_removed_ratio = float(torch.linalg.vector_norm(difference).item() / original_norm)

        original_total = self._ortho_buffers["original_total"]
        new_total = self._ortho_buffers["new_total"]
        self._flatten_reference_gradients(original_total)
        new_total.copy_(original_total).sub_(original_tracked).add_(projected_tracked)
        original_total_norm = float(torch.linalg.vector_norm(original_total).item())
        new_total_norm = float(torch.linalg.vector_norm(new_total).item())
        if original_total_norm > self.eps and math.isfinite(original_total_norm) and math.isfinite(new_total_norm):
            self.last_total_norm_ratio = new_total_norm / original_total_norm

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.ortho_project_ms = (time.perf_counter() - project_started) * 1000.0

        if projection_allowed and not self.ortho_monitor_only:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            writeback_started = time.perf_counter()
            self._writeback_reference_gradients(new_total)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.ortho_writeback_ms = (time.perf_counter() - writeback_started) * 1000.0

    def _compute_pending_weights(self) -> dict[str, float]:
        relative_losses = {
            task: self.loss_ema[task] / max(self.initial_loss_baseline[task], self.eps) for task in self.tasks
        }
        mean_relative = sum(relative_losses.values()) / len(self.tasks)
        effective = {task: self.task_weights[task] * self.grad_norm_ema[task] for task in self.tasks}
        mean_effective = sum(effective.values()) / len(self.tasks)
        candidates: dict[str, float] = {}
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        for task in self.tasks:
            target = mean_effective * (relative_losses[task] / max(mean_relative, self.eps)) ** self.alpha
            candidate = self.task_weights[task] * (target / max(effective[task], self.eps)) ** self.update_rate
            lower[task] = max(self.weight_min, self.task_weights[task] * self.step_ratio_min)
            upper[task] = min(self.weight_max, self.task_weights[task] * self.step_ratio_max)
            candidates[task] = min(max(candidate, lower[task]), upper[task])
        return self._normalize_with_bounds(candidates, lower, upper, float(len(self.tasks)))

    def _normalize_with_bounds(
        self,
        values: dict[str, float],
        lower: dict[str, float],
        upper: dict[str, float],
        target_sum: float,
    ) -> dict[str, float]:
        low_multiplier, high_multiplier = 0.0, 1.0
        while sum(min(max(value * high_multiplier, lower[task]), upper[task]) for task, value in values.items()) < target_sum:
            high_multiplier *= 2.0
        for _ in range(80):
            multiplier = (low_multiplier + high_multiplier) / 2.0
            total = sum(min(max(value * multiplier, lower[task]), upper[task]) for task, value in values.items())
            if total < target_sum:
                low_multiplier = multiplier
            else:
                high_multiplier = multiplier
        multiplier = (low_multiplier + high_multiplier) / 2.0
        normalized = {
            task: min(max(value * multiplier, lower[task]), upper[task]) for task, value in values.items()
        }
        correction = target_sum - sum(normalized.values())
        if abs(correction) > 1.0e-10:
            for task in self.tasks:
                room = upper[task] - normalized[task] if correction > 0 else normalized[task] - lower[task]
                delta = math.copysign(min(abs(correction), room), correction)
                normalized[task] += delta
                correction -= delta
                if abs(correction) <= 1.0e-10:
                    break
        return normalized

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for task in self.tasks:
            metrics[f"a_gn_w_{task}"] = self.task_weights[task]
            if self.baseline_initialized and task in self.loss_ema:
                metrics[f"b_gn_loss_ratio_{task}"] = self.loss_ema[task] / max(
                    self.initial_loss_baseline[task], self.eps
                )
            if task in self.last_raw_norms:
                metrics[f"c_gn_raw_norm_{task}"] = self.last_raw_norms[task]
                metrics[f"c_gn_effective_norm_{task}"] = self.last_effective_norms[task]
        for key, value in self.last_cosines.items():
            if value is not None:
                metrics[f"d_gn_cos_{key}"] = value
        metrics.update(
            {
                "e_gn_measure_active": float(self.measure_active),
                "e_gn_weight_update_active": float(self.weight_update_active),
                "e_gn_max_weight_delta": self.max_weight_delta,
            }
        )
        if self.ortho_enabled:
            metrics.update(
                {
                    "mtg/ortho/active": float(self.ortho_step_due),
                    "mtg/ortho/monitor_only": float(self.ortho_monitor_only),
                    "mtg/ortho/pair_count": float(sum(self.last_ortho_pairs.values())),
                    "mtg/ortho/removed_ratio": self.last_removed_ratio,
                    "mtg/ortho/tracked_norm_ratio": self.last_tracked_norm_ratio,
                    "mtg/ortho/total_norm_ratio": self.last_total_norm_ratio,
                    "mtg/ortho/time/capture_ms": self.capture_ms if self.ortho_step_due else 0.0,
                    "mtg/ortho/time/sync_ms": self.sync_ms if self.ortho_step_due else 0.0,
                    "mtg/ortho/time/project_ms": self.ortho_project_ms,
                    "mtg/ortho/time/writeback_ms": self.ortho_writeback_ms,
                }
            )
            for key, active in self.last_ortho_pairs.items():
                metrics[f"mtg/ortho/pair/{key}"] = float(active)
                ema = self.cosine_ema.get(key)
                if ema is not None and math.isfinite(ema):
                    metrics[f"mtg/ortho/cosine_ema/{key}"] = ema
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = {
            "task_weights": self.task_weights,
            "pending_weights": self.pending_weights,
            "loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
            "initial_loss_baseline": self.initial_loss_baseline,
            "baseline_initialized": self.baseline_initialized,
            "measurement_count": self.measurement_count,
            "weight_update_count": self.weight_update_count,
            "last_cosines": self.last_cosines,
        }
        if self.ortho_enabled:
            state.update(
                {
                    "cosine_ema": self.cosine_ema,
                    "ortho_activation_state": self.ortho_activation_state,
                    "ortho_execution_count": self.ortho_execution_count,
                    "ortho_pair_counts": self.ortho_pair_counts,
                    "last_removed_ratio": self.last_removed_ratio,
                    "last_tracked_norm_ratio": self.last_tracked_norm_ratio,
                    "last_total_norm_ratio": self.last_total_norm_ratio,
                }
            )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state_tasks = set(state["task_weights"])
        if state_tasks != set(self.tasks):
            raise ValueError(f"Gradient controller checkpoint tasks {sorted(state_tasks)} do not match {self.tasks}.")
        self.task_weights = {task: float(state["task_weights"][task]) for task in self.tasks}
        pending = state.get("pending_weights")
        self.pending_weights = None if pending is None else {task: float(pending[task]) for task in self.tasks}
        self.loss_ema = {task: float(value) for task, value in state.get("loss_ema", {}).items()}
        self.grad_norm_ema = {task: float(value) for task, value in state.get("grad_norm_ema", {}).items()}
        self.initial_loss_baseline = {
            task: float(value) for task, value in state.get("initial_loss_baseline", {}).items()
        }
        self.baseline_initialized = bool(state.get("baseline_initialized", False))
        self.measurement_count = int(state.get("measurement_count", 0))
        self.weight_update_count = int(state.get("weight_update_count", 0))
        self.last_cosines = {
            key: (None if value is None else float(value)) for key, value in state.get("last_cosines", {}).items()
        }
        self.cosine_ema = {key: float(value) for key, value in state.get("cosine_ema", {}).items()}
        self.ortho_activation_state = bool(state.get("ortho_activation_state", False))
        self.ortho_execution_count = int(state.get("ortho_execution_count", 0))
        saved_pair_counts = state.get("ortho_pair_counts", {})
        self.ortho_pair_counts = {
            key: int(saved_pair_counts.get(key, 0)) for key in self.ortho_pair_counts
        }
        self.last_removed_ratio = float(state.get("last_removed_ratio", 0.0))
        self.last_tracked_norm_ratio = float(state.get("last_tracked_norm_ratio", 1.0))
        self.last_total_norm_ratio = float(state.get("last_total_norm_ratio", 1.0))

    def save_checkpoint(self, checkpoint_dir: str, is_main_process: bool) -> None:
        if not is_main_process:
            return
        with open(os.path.join(checkpoint_dir, self.STATE_FILENAME), "w", encoding="utf-8") as file:
            json.dump(self.state_dict(), file, ensure_ascii=False, indent=2)

    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        state = None
        if self._distributed_world_size() == 1 or torch.distributed.get_rank() == 0:
            state_path = os.path.join(checkpoint_dir, self.STATE_FILENAME)
            if os.path.isfile(state_path):
                with open(state_path, encoding="utf-8") as file:
                    state = json.load(file)
        if self._distributed_world_size() > 1:
            payload = [state]
            torch.distributed.broadcast_object_list(payload, src=0)
            state = payload[0]
        if state is None:
            return False
        self.load_state_dict(state)
        return True
