"""Historical Think-only trainer with continued-adapter evidence and lineage."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

from evidence import EvidenceRecGRPOTrainerMixin, StepEvidenceCallback
from modeling import assert_optimizer_lora_only, base_tensor_fingerprints
from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import ThinkSample8FullSIDTrainer
from contracts import file_sha256


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ContinuedAdapterLineageCallback(TrainerCallback):
    def __init__(self, payload: dict):
        self.payload = dict(payload)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        adapter = checkpoint / "adapter_model.safetensors"
        value = {
            **self.payload,
            "schema": "grpo2_continued_adapter_lineage_v1",
            "recipe": "grpo2_think_continued_adapter_v1",
            "stage": "GRPO2_REC_THINK",
            "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
            "adapter_weight_parent": "GRPO1 checkpoint-500",
            "optimizer_parent": "NONE",
            "trainer_state_parent": "NONE",
            "rng_parent": "NONE",
            "training_resume": False,
            "stage_fresh_start": True,
            "adapter_continuation": True,
            "grpo2_step": int(state.global_step),
            "adapter_only": True,
            "resume_supported": True,
            "adapter_sha256": file_sha256(adapter),
        }
        _write_json(checkpoint / "lineage.json", value)


class GRPO2ContinuedAdapterTrainer(EvidenceRecGRPOTrainerMixin, ThinkSample8FullSIDTrainer):
    def __init__(self, *args, lineage: dict, evidence_dir: Path, **kwargs):
        self._continued_lineage = dict(lineage)
        self._continued_evidence_dir = Path(evidence_dir)
        super().__init__(*args, **kwargs)
        self._initial_lora = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if "lora_" in name.lower()
        }
        self.add_callback(StepEvidenceCallback(self._continued_evidence_dir, int(os.environ.get("LOCAL_RANK", "0"))))
        self.add_callback(ContinuedAdapterLineageCallback(self._continued_lineage))

    def create_optimizer(self):
        result = super().create_optimizer()
        self._optimizer_audit = assert_optimizer_lora_only(self.model, self.optimizer)
        return result

    def train(self, *args, **kwargs):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        base_before = base_tensor_fingerprints(self.model)
        result = super().train(*args, **kwargs)
        base_after = base_tensor_fingerprints(self.model)
        if base_before != base_after:
            raise RuntimeError("GRPO2_IN_MEMORY_BASE_MUTATED")
        squared = 0.0
        max_abs = 0.0
        changed_tensors = 0
        for name, parameter in self.model.named_parameters():
            if name not in self._initial_lora:
                continue
            delta = parameter.detach().float().cpu() - self._initial_lora[name].float()
            norm = float(delta.norm().item())
            if norm > 0:
                changed_tensors += 1
            squared += norm * norm
            max_abs = max(max_abs, float(delta.abs().max().item()))
        adapter_delta_norm = math.sqrt(squared)
        if adapter_delta_norm <= 0:
            raise RuntimeError("GRPO2_INHERITED_ADAPTER_DID_NOT_CHANGE")
        payload = {
            "rank": rank,
            "base_before": base_before,
            "base_after": base_after,
            "base_unchanged": True,
            "base_max_parameter_delta": 0.0,
            "base_delta_proof": "requires_grad_false_optimizer_excluded_and_sampled_fingerprints_exact",
            "initial_lora_tensor_count": len(self._initial_lora),
            "changed_lora_tensor_count": changed_tensors,
            "adapter_delta_norm": adapter_delta_norm,
            "adapter_delta_max_abs": max_abs,
            "lora_changed": True,
            "optimizer_audit": getattr(self, "_optimizer_audit", None),
            "parity_log": self._parity_log,
            "rollouts": self._smoke_log,
        }
        _write_json(self._continued_evidence_dir / f"trainer-evidence-rank{rank}.json", payload)
        return result
