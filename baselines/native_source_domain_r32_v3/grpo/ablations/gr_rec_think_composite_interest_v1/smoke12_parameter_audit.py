"""Smoke-only runtime parameter evidence with CPU-resident LoRA snapshots."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


ARTIFACT_NAME = "smoke_parameter_audit.json"


def _is_lora(name: str) -> bool:
    return "lora" in name.lower()


class Smoke12ParameterAuditCallback(TrainerCallback):
    """Capture LoRA deltas and a lightweight frozen-base mutation sentinel."""

    def __init__(self, model, *, rank: int, run_dir: str | Path):
        self.model = model
        self.rank = int(rank)
        self.run_dir = Path(run_dir)
        self.artifact_path = self.run_dir / ARTIFACT_NAME
        self._lora_snapshot: dict[str, Any] = {}
        self._base_versions: dict[str, int] = {}
        self._base_requires_grad_count = 0
        self.completed = False

    def _capture_initial(self) -> None:
        for name, parameter in self.model.named_parameters():
            if _is_lora(name) and parameter.requires_grad:
                self._lora_snapshot[name] = parameter.detach().cpu().clone()
            elif not _is_lora(name):
                self._base_versions[name] = int(parameter._version)
                self._base_requires_grad_count += int(bool(parameter.requires_grad))

    def on_train_begin(self, args, state, control, **kwargs):
        if self.rank == 0 and not self._lora_snapshot and not self._base_versions:
            self._capture_initial()
        return control

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload

    def audit(self, optimizer_steps: int) -> dict[str, Any]:
        if self.rank != 0:
            return {}
        current = dict(self.model.named_parameters())
        total_squared = 0.0
        max_abs = 0.0
        finite = True
        for name, before in self._lora_snapshot.items():
            parameter = current.get(name)
            if parameter is None:
                finite = False
                continue
            after = parameter.detach().cpu()
            delta = after - before
            squared = float(delta.double().square().sum().item())
            element_max = float(delta.abs().max().item()) if delta.numel() else 0.0
            finite = finite and math.isfinite(squared) and math.isfinite(element_max)
            total_squared += squared
            max_abs = max(max_abs, element_max)

        total_l2 = math.sqrt(total_squared) if math.isfinite(total_squared) else math.nan
        finite = finite and math.isfinite(total_l2) and math.isfinite(max_abs)
        changed_base = 0
        for name, version in self._base_versions.items():
            parameter = current.get(name)
            changed_base += int(parameter is None or int(parameter._version) != version)

        payload = {
            "optimizer_steps": int(optimizer_steps),
            "runtime_optimizer_steps": int(optimizer_steps),
            "BASE_DELTA": int(changed_base > 0),
            "BASE_CHANGED": changed_base > 0,
            "base_version_changed_count": changed_base,
            "base_requires_grad_count": self._base_requires_grad_count,
            "base_delta_semantics": "base mutation sentinel; not full-parameter L2",
            "LORA_CHANGED": bool(finite and total_l2 > 0.0),
            "lora_total_l2_delta": total_l2 if finite else None,
            "lora_max_abs_delta": max_abs if finite else None,
            "lora_parameter_tensor_count": len(self._lora_snapshot),
            "finite": bool(finite),
            "runtime_error": False,
        }
        self.completed = True
        self._lora_snapshot.clear()
        self._base_versions.clear()
        return self._write(payload)

    def write_runtime_error(self, optimizer_steps: int) -> dict[str, Any]:
        if self.rank != 0:
            return {}
        payload = {
            "optimizer_steps": int(optimizer_steps),
            "runtime_optimizer_steps": int(optimizer_steps),
            "BASE_DELTA": None,
            "BASE_CHANGED": None,
            "base_version_changed_count": None,
            "base_requires_grad_count": self._base_requires_grad_count,
            "base_delta_semantics": "base mutation sentinel; not full-parameter L2",
            "LORA_CHANGED": None,
            "lora_total_l2_delta": None,
            "lora_max_abs_delta": None,
            "lora_parameter_tensor_count": len(self._lora_snapshot),
            "finite": False,
            "runtime_error": True,
        }
        self._lora_snapshot.clear()
        self._base_versions.clear()
        return self._write(payload)

    def on_train_end(self, args, state, control, **kwargs):
        self.audit(state.global_step)
        return control
