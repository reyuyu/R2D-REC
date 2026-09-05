"""Historical Sample8 trainer with deterministic evidence and GRPO-2 lineage."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import TrainerCallback

from evidence import EvidenceRecGRPOTrainerMixin, StepEvidenceCallback
from modeling import (
    assert_optimizer_lora_only,
    base_tensor_fingerprints,
    sampled_parameter_fingerprint,
)
from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import (
    ThinkSample8FullSIDTrainer,
)
from contracts import file_sha256


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GRPO2LineageCallback(TrainerCallback):
    def __init__(self, payload: dict):
        self.payload = dict(payload)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        adapter = checkpoint / "adapter_model.safetensors"
        value = {
            **self.payload,
            "schema": "grpo2_adapter_lineage_v1",
            "recipe": "grpo2_think_fullbase_conservative_v1",
            "step": int(state.global_step),
            "adapter_sha256": None,
            "resume_supported": int(state.global_step) % 2 == 0,
        }
        value["adapter_sha256"] = file_sha256(adapter)
        _write_json(checkpoint / "lineage.json", value)


class GRPO2ThinkTrainer(EvidenceRecGRPOTrainerMixin, ThinkSample8FullSIDTrainer):
    def __init__(self, *args, lineage: dict, evidence_dir: Path, **kwargs):
        self._grpo2_lineage = dict(lineage)
        self._grpo2_evidence_dir = Path(evidence_dir)
        super().__init__(*args, **kwargs)
        self.add_callback(StepEvidenceCallback(self._grpo2_evidence_dir, int(os.environ.get("LOCAL_RANK", "0"))))
        self.add_callback(GRPO2LineageCallback(self._grpo2_lineage))

    def create_optimizer(self):
        result = super().create_optimizer()
        self._grpo2_optimizer_audit = assert_optimizer_lora_only(self.model, self.optimizer)
        return result

    def train(self, *args, **kwargs):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        base_before = base_tensor_fingerprints(self.model)
        lora_before = sampled_parameter_fingerprint(self.model.named_parameters(), include_lora=True)
        result = super().train(*args, **kwargs)
        base_after = base_tensor_fingerprints(self.model)
        lora_after = sampled_parameter_fingerprint(self.model.named_parameters(), include_lora=True)
        if base_before != base_after:
            raise RuntimeError("GRPO2_IN_MEMORY_BASE_MUTATED")
        if lora_before == lora_after:
            raise RuntimeError("GRPO2_LORA_DID_NOT_CHANGE")
        payload = {
            "rank": rank,
            "base_before": base_before,
            "base_after": base_after,
            "base_unchanged": True,
            "lora_before": lora_before,
            "lora_after": lora_after,
            "lora_changed": True,
            "optimizer_audit": getattr(self, "_grpo2_optimizer_audit", None),
            "parity_log": self._parity_log,
            "rollouts": self._smoke_log,
        }
        _write_json(self._grpo2_evidence_dir / f"trainer-evidence-rank{rank}.json", payload)
        return result
