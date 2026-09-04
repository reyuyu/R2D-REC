"""Compact, deterministic evidence captured around optimizer updates."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ModuleNotFoundError:  # Pure contract tests do not require Transformers.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from checkpointing import write_json_atomic
from modeling import gradient_fingerprints, sampled_parameter_fingerprint, sample_positions


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rng_fingerprint(device: Any | None = None) -> dict[str, str]:
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256()
    numpy_digest.update(str(numpy_state[0]).encode("ascii"))
    numpy_digest.update(numpy_state[1].tobytes())
    numpy_digest.update(str(numpy_state[2:]).encode("ascii"))
    result = {
        "python": _digest_json(random.getstate()),
        "numpy": numpy_digest.hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
        result["torch_cuda"] = hashlib.sha256(cuda_state.cpu().numpy().tobytes()).hexdigest()
    return result


def optimizer_fingerprint(optimizer: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(str(group_index).encode("ascii"))
        for key in sorted(key for key in group if key != "params"):
            digest.update(key.encode("utf-8"))
            digest.update(str(group[key]).encode("utf-8"))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for key in sorted(state):
                value = state[key]
                digest.update(key.encode("utf-8"))
                if torch.is_tensor(value):
                    flat = value.detach().reshape(-1)
                    if flat.numel():
                        positions = sample_positions(flat.numel(), 32, flat.device)
                        sample = flat.index_select(0, positions).contiguous().cpu()
                        digest.update(sample.view(torch.uint8).numpy().tobytes())
                else:
                    digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


class EvidenceRecGRPOTrainerMixin:
    """Adds evidence only; inherited GRPO generation and loss math stay untouched."""

    def _generate_and_score_completions(self, inputs):
        from grpo_model import render_prompt

        prompt_fingerprints = []
        for item in inputs:
            rendered = render_prompt(self.processing_class, item["prompt"])
            token_ids = self.processing_class.encode(rendered, add_special_tokens=False)
            prompt_fingerprints.append(_digest_json(token_ids))
        result = super()._generate_and_score_completions(inputs)
        if self._parity_log:
            self._parity_log[-1]["prompt_token_fingerprints"] = prompt_fingerprints
        return result

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super()._compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if self._parity_log:
            self._parity_log[-1].setdefault("policy_losses", []).append(float(loss.detach()))
        return loss


class StepEvidenceCallback(TrainerCallback):
    def __init__(self, output_dir: str | Path, rank: int):
        self.path = Path(output_dir) / f"step-evidence-rank{rank}.jsonl"
        self.rank = rank
        self._pre_optimizer: dict[str, Any] | None = None

    def _append(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        gradients, global_norm = gradient_fingerprints(model)
        if not math.isfinite(global_norm):
            raise RuntimeError(f"non-finite global gradient norm: {global_norm}")
        self._pre_optimizer = {
            "gradient_global_norm": global_norm,
            "per_lora_gradients": gradients,
        }

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        record = {
            "step": int(state.global_step),
            "rank": self.rank,
            "rng": rng_fingerprint(getattr(model, "device", None)),
            "lora_sample_fingerprint": sampled_parameter_fingerprint(
                model.named_parameters(), include_lora=True
            ),
            "optimizer_fingerprint": optimizer_fingerprint(optimizer),
        }
        if self._pre_optimizer is not None:
            record.update(self._pre_optimizer)
        self._append(record)
        self._pre_optimizer = None


def write_rank_summary(path: str | Path, value: dict[str, Any]) -> None:
    write_json_atomic(Path(path), value)
