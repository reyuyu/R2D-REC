"""Low-interference evidence capture for the recovered BATA 553->554 replay."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from transformers import TrainerCallback


_CONTROLLER: "ReplayForensics | None" = None
_BATCH_FIELDS = (
    "input_ids",
    "labels",
    "loss_weights",
    "sample_ids",
    "sample_task_ids",
    "sample_domain_weights",
    "attention_mask",
    "position_ids",
)
_DEFAULT_HEAVY_STEPS = {555, 560}
_FROZEN_BATCH_MODE = "frozen_step554_contract"


def _parse_step_set(value: str | None) -> set[int]:
    if value is None or not value.strip():
        return set(_DEFAULT_HEAVY_STEPS)
    steps = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(step < 0 for step in steps):
        raise ValueError("BATA_REPLAY_HEAVY_STEPS must contain non-negative integers.")
    return steps


def configure_deterministic_diagnostic(enabled: bool | None = None) -> dict[str, Any]:
    """Enable strict PyTorch determinism before any CUDA context is created."""
    if enabled is None:
        enabled = os.environ.get("BATA_REPLAY_DETERMINISTIC") == "1"
    if not enabled:
        return {"enabled": False}
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace != ":4096:8":
        raise RuntimeError("Deterministic replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")
    if torch.cuda.is_initialized():
        raise RuntimeError("Deterministic replay was configured after CUDA initialization.")
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms did not become active.")
    return {
        "enabled": True,
        "warn_only": False,
        "cublas_workspace_config": workspace,
        "configured_before_cuda": True,
    }


_DETERMINISTIC_SETUP = configure_deterministic_diagnostic()


def _hash_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_fingerprint(name: str, tensor: torch.Tensor) -> str:
    return _hash_bytes(
        [
            name.encode(),
            str(tensor.dtype).encode(),
            json.dumps(list(tensor.shape)).encode(),
            _tensor_bytes(tensor),
        ]
    )


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device_type": tensor.device.type,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_lora_name(name: str) -> str:
    name = name.removeprefix("module.")
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def _jsonable_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_fingerprint("state", value),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable_state(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class ReplayForensics:
    def __init__(self) -> None:
        self.output_dir = Path(os.environ["BATA_REPLAY_EVIDENCE_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.run_label = os.environ.get("BATA_REPLAY_LABEL", "unknown")
        self.checkpoint = Path(os.environ["BATA_REPLAY_CHECKPOINT"])
        self.events_path = self.output_dir / f"rank{self.rank}.jsonl"
        self.batch_hashes: dict[int, list[str]] = {}
        self.batch_metadata: dict[int, list[str]] = {}
        self.losses: dict[int, list[torch.Tensor]] = {}
        self.last_grad_norm: torch.Tensor | float | None = None
        self.train_begin_seen = False
        self.restore_seen = False
        self.ddp_hook_installed = False
        self.ddp_bucket_sequence = 0
        self._emit_lock = threading.Lock()
        self.heavy_steps = _parse_step_set(os.environ.get("BATA_REPLAY_HEAVY_STEPS"))
        self.batch_fingerprint_mode = os.environ.get("BATA_REPLAY_BATCH_FINGERPRINT_MODE", "full")
        self.batch_contract = self._load_batch_contract()

    def _load_batch_contract(self) -> dict[str, Any] | None:
        if self.batch_fingerprint_mode != _FROZEN_BATCH_MODE:
            return None
        path_value = os.environ.get("BATA_REPLAY_BATCH_CONTRACT_JSON")
        if not path_value:
            raise RuntimeError("Frozen batch mode requires BATA_REPLAY_BATCH_CONTRACT_JSON.")
        path = Path(path_value)
        contract = json.loads(path.read_text(encoding="utf-8"))
        if contract.get("step") != 554 or contract.get("world_size") != self.world_size:
            raise RuntimeError("Frozen batch contract step/world-size mismatch.")
        rank_contract = contract.get("ranks", {}).get(str(self.rank))
        if not rank_contract or rank_contract.get("microbatch_count") != 16:
            raise RuntimeError(f"Frozen batch contract is missing rank {self.rank}.")
        return {
            "path": str(path),
            "sha256": _file_sha256(path),
            "rank": rank_contract,
        }

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "rank": self.rank,
            "run_label": self.run_label,
            "time_unix": time.time(),
            **payload,
        }
        with self._emit_lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    def environment(self) -> dict[str, Any]:
        selected = {}
        for key, value in sorted(os.environ.items()):
            if key in {"PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES"} or key.startswith(
                ("NCCL_", "TORCH_")
            ):
                selected[key] = value
        cuda_props = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            cuda_props = {"name": props.name, "total_memory": props.total_memory}
        return {
            "environment": selected,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": cuda_props,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_setup": _DETERMINISTIC_SETUP,
        }

    def rng_hashes(self) -> dict[str, str]:
        numpy_state = np.random.get_state()
        parts = [
            numpy_state[0].encode(),
            numpy_state[1].tobytes(),
            str(numpy_state[2:]).encode(),
        ]
        result = {
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": _hash_bytes(parts),
            "torch_cpu": hashlib.sha256(_tensor_bytes(torch.get_rng_state())).hexdigest(),
        }
        if torch.cuda.is_available():
            result["torch_cuda"] = hashlib.sha256(_tensor_bytes(torch.cuda.get_rng_state())).hexdigest()
        return result

    def record_batch(self, trainer: Any, inputs: dict[str, Any]) -> None:
        target_step = int(trainer.state.global_step) + 1
        if self.batch_fingerprint_mode == _FROZEN_BATCH_MODE:
            fields = {
                field: _tensor_metadata(value) if torch.is_tensor(value) else None
                for field in _BATCH_FIELDS
                for value in [inputs.get(field)]
            }
            metadata_fingerprint = _canonical_json_hash(fields)
            values = self.batch_metadata.setdefault(target_step, [])
            values.append(metadata_fingerprint)
            self.emit(
                "microbatch_contract_observation",
                target_step=target_step,
                micro_index=len(values) - 1,
                metadata_fingerprint=metadata_fingerprint,
                fields=fields,
                value_fingerprint_source="frozen_prior_exact_contract",
            )
            return

        parts = []
        fields = {}
        for field in _BATCH_FIELDS:
            value = inputs.get(field)
            if torch.is_tensor(value):
                fingerprint = _tensor_fingerprint(field, value)
                fields[field] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": fingerprint,
                }
                parts.extend((field.encode(), fingerprint.encode()))
            else:
                fields[field] = None
                parts.extend((field.encode(), b"ABSENT"))
        fingerprint = _hash_bytes(parts)
        values = self.batch_hashes.setdefault(target_step, [])
        values.append(fingerprint)
        self.emit(
            "microbatch",
            target_step=target_step,
            micro_index=len(values) - 1,
            fingerprint=fingerprint,
            fields=fields,
        )

    def record_loss(self, trainer: Any, loss: Any) -> None:
        value = loss[0] if isinstance(loss, tuple) else loss
        if torch.is_tensor(value):
            target_step = int(trainer.state.global_step) + 1
            self.losses.setdefault(target_step, []).append(value.detach())

    def record_grad_norm(self, value: Any) -> None:
        self.last_grad_norm = value.detach() if torch.is_tensor(value) else value

    def _canonical_lora_gradients(self, model: torch.nn.Module) -> dict[str, Any]:
        model = getattr(model, "module", model)
        tensors = {
            _normalise_lora_name(name): parameter.grad.detach()
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad and parameter.grad is not None
        }
        digest = hashlib.sha256()
        norm_sq = 0.0
        tensor_records = []
        for name in sorted(tensors):
            value = tensors[name]
            cpu_value = value.detach().contiguous().cpu()
            raw_fingerprint = _tensor_fingerprint("gradient", cpu_value)
            digest.update(name.encode())
            digest.update(raw_fingerprint.encode())
            norm_sq += float(torch.sum(cpu_value.double() ** 2))
            tensor_records.append(
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "numel": value.numel(),
                    "sha256": raw_fingerprint,
                }
            )
        return {
            "sha256": digest.hexdigest(),
            "tensor_count": len(tensors),
            "numel": sum(value.numel() for value in tensors.values()),
            "l2_norm": norm_sq**0.5,
            "tensors": tensor_records,
        }

    def record_clip(self, stage: str, model: torch.nn.Module, returned_grad_norm: Any = None) -> None:
        payload = {
            "stage": stage,
            "gradients": self._canonical_lora_gradients(model),
        }
        if returned_grad_norm is not None:
            if torch.is_tensor(returned_grad_norm):
                returned_grad_norm = float(returned_grad_norm.detach().float().cpu().item())
            payload["returned_grad_norm"] = float(returned_grad_norm)
        self.emit("gradient_clip", **payload)

    def _bucket_metadata(self, bucket: Any, parameter_names: dict[int, str]) -> dict[str, Any]:
        parameters = []
        offset = 0
        for position, parameter in enumerate(bucket.parameters()):
            numel = parameter.numel()
            parameters.append(
                {
                    "position": position,
                    "name": parameter_names.get(id(parameter), f"unknown:{position}"),
                    "dtype": str(parameter.dtype),
                    "shape": list(parameter.shape),
                    "numel": numel,
                    "offset": offset,
                }
            )
            offset += numel
        layout = {
            "parameters": parameters,
            "parameter_count": len(parameters),
            "parameter_numel": offset,
        }
        buffer = bucket.buffer()
        return {
            "bucket_index": int(bucket.index()),
            "is_last": bool(bucket.is_last()),
            "dtype": str(buffer.dtype),
            "numel": buffer.numel(),
            "shape": list(buffer.shape),
            "layout_sha256": _canonical_json_hash(layout),
            **layout,
        }

    def install_ddp_comm_hook(self, model: torch.nn.Module) -> None:
        if self.ddp_hook_installed:
            raise RuntimeError("DDP forensic communication hook was installed more than once.")
        if not isinstance(model, DistributedDataParallel):
            if self.world_size > 1:
                raise RuntimeError(f"Expected DistributedDataParallel, got {type(model).__name__}.")
            self.emit("ddp_hook_not_required", world_size=self.world_size)
            return

        parameter_names = {
            id(parameter): _normalise_lora_name(name)
            for name, parameter in model.named_parameters()
        }
        process_group = model.process_group

        def forensic_allreduce_hook(state: Any, bucket: Any):
            sequence = self.ddp_bucket_sequence
            self.ddp_bucket_sequence += 1
            metadata = self._bucket_metadata(bucket, parameter_names)
            pre_sha = _tensor_fingerprint("pre_allreduce", bucket.buffer())
            self.emit(
                "ddp_pre_allreduce",
                bucket_call_index=sequence,
                fingerprint=pre_sha,
                **metadata,
            )
            future = default_hooks.allreduce_hook(state, bucket)

            def record_post(completed: Any):
                reduced = completed.value()
                self.emit(
                    "ddp_post_allreduce",
                    bucket_call_index=sequence,
                    fingerprint=_tensor_fingerprint("post_allreduce", reduced),
                    **metadata,
                )
                return reduced

            return future.then(record_post)

        model.register_comm_hook(process_group, forensic_allreduce_hook)
        self.ddp_hook_installed = True
        self.emit(
            "ddp_hook_installed",
            implementation="torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook",
            torch_version=torch.__version__,
            default_allreduce_source_sha256=hashlib.sha256(
                inspect.getsource(default_hooks._allreduce_fut).encode()
            ).hexdigest(),
        )

    def _canonical_lora(self, model: torch.nn.Module) -> tuple[str, dict[str, torch.Tensor]]:
        tensors = {
            _normalise_lora_name(name): parameter.detach()
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        }
        digest = hashlib.sha256()
        for name in sorted(tensors):
            value = tensors[name]
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(_tensor_bytes(value))
        return digest.hexdigest(), tensors

    def _effective_ba(self, tensors: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, Any]:
        config = next(iter(getattr(model, "peft_config", {}).values()), None)
        scaling = float(config.lora_alpha) / float(config.r) if config is not None else 2.0
        prefixes = sorted(name[: -len(".lora_A.weight")] for name in tensors if name.endswith(".lora_A.weight"))
        digest = hashlib.sha256()
        norm_sq = 0.0
        for prefix in prefixes:
            a = tensors[prefix + ".lora_A.weight"].float().cpu()
            b = tensors[prefix + ".lora_B.weight"].float().cpu()
            gram_norm = torch.sum((b.double().T @ b.double()) * (a.double() @ a.double().T)).item()
            norm_sq += scaling * scaling * gram_norm
            right_indices = sorted({0, a.shape[1] // 2, a.shape[1] - 1})
            left_indices = sorted({0, b.shape[0] // 2, b.shape[0] - 1})
            right_projection = scaling * (b @ a[:, right_indices])
            left_projection = scaling * (b[left_indices, :] @ a)
            digest.update(prefix.encode())
            digest.update(json.dumps([list(b.shape), list(a.shape), scaling]).encode())
            digest.update(_tensor_bytes(right_projection))
            digest.update(_tensor_bytes(left_projection))
            digest.update(repr(gram_norm).encode())
        return {
            "method": "effective_ba_projection_v1",
            "sha256": digest.hexdigest(),
            "frobenius_norm": norm_sq**0.5,
            "module_count": len(prefixes),
            "scaling": scaling,
        }

    def _optimizer_fingerprint(self, optimizer: Any, model: torch.nn.Module) -> dict[str, Any]:
        optimizer = getattr(optimizer, "optimizer", optimizer)
        names = {id(parameter): _normalise_lora_name(name) for name, parameter in model.named_parameters()}
        digest = hashlib.sha256()
        step_values = []
        state_entries = 0
        for parameter, state in sorted(optimizer.state.items(), key=lambda item: names.get(id(item[0]), "")):
            name = names.get(id(parameter), f"unknown:{id(parameter)}")
            digest.update(name.encode())
            state_entries += 1
            for key in sorted(state, key=str):
                digest.update(str(key).encode())
                value = state[key]
                if torch.is_tensor(value):
                    digest.update(_tensor_bytes(value))
                    if str(key) == "step" and value.numel() == 1:
                        step_values.append(float(value.detach().cpu().item()))
                else:
                    digest.update(repr(value).encode())
                    if str(key) == "step":
                        step_values.append(float(value))
        return {
            "sha256": digest.hexdigest(),
            "state_entries": state_entries,
            "adam_step_min": min(step_values) if step_values else None,
            "adam_step_max": max(step_values) if step_values else None,
        }

    def _restored_adapter_match(self, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
        from safetensors import safe_open

        path = self.checkpoint / "adapter_model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as saved:
            saved_keys = sorted(saved.keys())
            missing = [key for key in saved_keys if key not in tensors]
            extra = sorted(set(tensors) - set(saved_keys))
            mismatched = []
            for key in saved_keys:
                if key in tensors and not torch.equal(saved.get_tensor(key), tensors[key].detach().cpu()):
                    mismatched.append(key)
        return {
            "exact": not missing and not extra and not mismatched,
            "saved_count": len(saved_keys),
            "runtime_count": len(tensors),
            "missing": missing[:10],
            "extra": extra[:10],
            "mismatched": mismatched[:10],
        }

    def heavy(self, label: str, model: torch.nn.Module, optimizer: Any, scheduler: Any) -> None:
        if self.rank != 0:
            return
        model = getattr(model, "module", model)
        adapter_sha, tensors = self._canonical_lora(model)
        payload = {
            "label": label,
            "canonical_lora_sha256": adapter_sha,
            "lora_tensor_count": len(tensors),
            "lora_norm": sum(float(torch.sum(value.detach().double() ** 2).cpu()) for value in tensors.values()) ** 0.5,
            "effective_ba": self._effective_ba(tensors, model),
            "optimizer": self._optimizer_fingerprint(optimizer, model),
            "scheduler": _jsonable_state(scheduler.state_dict()),
        }
        if label == "initial553":
            payload["restored_adapter"] = self._restored_adapter_match(tensors)
            expected_path = Path(os.environ["BATA_REPLAY_EXPECTED_SHA_JSON"])
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            actual = {name: _file_sha256(self.checkpoint / name) for name in expected}
            payload["checkpoint_sha"] = {
                "expected": expected,
                "actual": actual,
                "exact": expected == actual,
            }
            if expected != actual or not payload["restored_adapter"]["exact"]:
                self.emit("restore_validation_failed", **payload)
                raise RuntimeError("Historical checkpoint restore validation failed.")
        self.emit("heavy_fingerprint", **payload)

    def restored(self, trainer: Any, checkpoint: str) -> None:
        self.restore_seen = True
        self.emit(
            "rng_restored",
            checkpoint=str(checkpoint),
            global_step=int(trainer.state.global_step),
            epoch=trainer.state.epoch,
            train_begin_seen=self.train_begin_seen,
            rng=self.rng_hashes(),
        )
        self.heavy("initial553", trainer.model, trainer.optimizer, trainer.lr_scheduler)

    def step_end(self, state: Any, model: Any, optimizer: Any, scheduler: Any) -> None:
        step = int(state.global_step)
        batch_hashes = self.batch_hashes.pop(step, [])
        batch_metadata = self.batch_metadata.pop(step, [])
        losses = self.losses.pop(step, [])
        combined = _hash_bytes([value.encode() for value in batch_hashes]) if batch_hashes else None
        loss_values = [float(value.float().cpu().item()) for value in losses]
        if torch.is_tensor(self.last_grad_norm):
            grad_norm = float(self.last_grad_norm.float().cpu().item())
        elif self.last_grad_norm is None:
            grad_norm = None
        else:
            grad_norm = float(self.last_grad_norm)
        lr_values = scheduler.get_last_lr() if scheduler is not None else []
        batch_contract = None
        if self.batch_fingerprint_mode == _FROZEN_BATCH_MODE:
            assert self.batch_contract is not None
            expected = self.batch_contract["rank"]
            batch_contract = {
                "source": _FROZEN_BATCH_MODE,
                "contract_sha256": self.batch_contract["sha256"],
                "rank_ordered_batch_fingerprint": expected["ordered_batch_fingerprint"],
                "expected_microbatch_count": expected["microbatch_count"],
                "observed_microbatch_count": len(batch_metadata),
                "runtime_ordered_metadata_sha256": batch_metadata,
                "runtime_metadata_fingerprint": _hash_bytes(
                    [value.encode() for value in batch_metadata]
                ),
            }
            if len(batch_metadata) != expected["microbatch_count"]:
                raise RuntimeError("Runtime microbatch count violates the frozen step554 contract.")
            combined = expected["ordered_batch_fingerprint"]
        else:
            batch_contract = {
                "source": "runtime_full_gpu_hash",
                "expected_microbatch_count": len(batch_hashes),
                "observed_microbatch_count": len(batch_hashes),
                "runtime_ordered_microbatch_sha256": batch_hashes,
            }

        self.emit(
            "optimizer_step",
            global_step=step,
            epoch=state.epoch,
            lr=lr_values,
            rank_local_loss_mean=sum(loss_values) / len(loss_values) if loss_values else None,
            rank_local_micro_losses=loss_values,
            grad_norm=grad_norm,
            microbatch_count=len(batch_metadata) if self.batch_fingerprint_mode == _FROZEN_BATCH_MODE else len(batch_hashes),
            ordered_microbatch_sha256=batch_hashes,
            ordered_batch_fingerprint=combined,
            batch_contract=batch_contract,
            rng=self.rng_hashes(),
        )
        if step in self.heavy_steps:
            self.heavy(f"step{step}", model, optimizer, scheduler)


class ReplayForensicsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        controller = get_controller()
        controller.train_begin_seen = True
        controller.emit(
            "train_begin_before_epoch_rng_restore",
            global_step=int(state.global_step),
            epoch=state.epoch,
            environment=controller.environment(),
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        get_controller().step_end(state, kwargs["model"], kwargs["optimizer"], kwargs["lr_scheduler"])
        return control

    def on_train_end(self, args, state, control, **kwargs):
        controller = get_controller()
        controller.emit(
            "train_end",
            global_step=int(state.global_step),
            epoch=state.epoch,
            restore_seen=controller.restore_seen,
        )
        return control


class StopAfterOptimizerStepCallback(TrainerCallback):
    """Stop after an optimizer step without changing the scheduler horizon."""

    def __init__(self, stop_after_step: int) -> None:
        if stop_after_step <= 0:
            raise ValueError("stop_after_step must be positive.")
        self.stop_after_step = int(stop_after_step)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) >= self.stop_after_step:
            control.should_training_stop = True
        return control


def get_controller() -> ReplayForensics:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = ReplayForensics()
    return _CONTROLLER


def install_replay_instrumentation(trainer_cls: type) -> None:
    if not os.environ.get("BATA_REPLAY_EVIDENCE_DIR") or getattr(trainer_cls, "_bata_replay_installed", False):
        return
    if os.environ.get("BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD") == "1":
        expected_path = Path(os.environ["BATA_REPLAY_EXPECTED_SHA_JSON"])
        checkpoint = Path(os.environ["BATA_REPLAY_CHECKPOINT"])
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        actual = {name: _file_sha256(checkpoint / name) for name in expected}
        if actual != expected:
            raise RuntimeError("Refusing trusted torch.load compatibility for a checkpoint SHA mismatch.")

        # Transformers added a torch>=2.6 policy gate after this trusted local
        # checkpoint was written. Preserve its weights_only=True load path while
        # restoring the historical torch 2.5 behavior for this SHA-pinned replay.
        import transformers.trainer as transformers_trainer
        from numpy._core.multiarray import _reconstruct

        safe_numpy_globals = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
        torch.serialization.add_safe_globals(safe_numpy_globals)

        transformers_trainer.check_torch_load_is_safe = lambda: None
        get_controller().emit(
            "trusted_torch_load_compatibility",
            checkpoint=str(checkpoint),
            checkpoint_sha_exact=True,
            weights_only_load_preserved=True,
            safe_numpy_globals=[f"{item.__module__}.{item.__name__}" for item in safe_numpy_globals],
        )
    original_compute_loss = trainer_cls.compute_loss
    original_load_rng_state = trainer_cls._load_rng_state
    original_clip_grad_norm = trainer_cls._clip_grad_norm
    original_prepare_for_training = getattr(trainer_cls, "_prepare_for_training", None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        controller = get_controller()
        controller.record_batch(self, inputs)
        result = original_compute_loss(self, model, inputs, return_outputs=return_outputs, **kwargs)
        controller.record_loss(self, result)
        return result

    def load_rng_state(self, checkpoint):
        result = original_load_rng_state(self, checkpoint)
        get_controller().restored(self, checkpoint)
        return result

    def clip_grad_norm(self, model):
        controller = get_controller()
        controller.record_clip("PRE_CLIP", model)
        result = original_clip_grad_norm(self, model)
        controller.record_clip("POST_CLIP", model, returned_grad_norm=result)
        controller.record_grad_norm(result)
        return result

    def prepare_for_training(self, *args, **kwargs):
        model, train_dataloader = original_prepare_for_training(self, *args, **kwargs)
        get_controller().install_ddp_comm_hook(model)
        return model, train_dataloader

    trainer_cls.compute_loss = compute_loss
    trainer_cls._load_rng_state = load_rng_state
    trainer_cls._clip_grad_norm = clip_grad_norm
    if original_prepare_for_training is not None:
        trainer_cls._prepare_for_training = prepare_for_training
    trainer_cls._bata_replay_installed = True


def replay_callbacks() -> list[TrainerCallback]:
    callbacks: list[TrainerCallback] = []
    if os.environ.get("BATA_REPLAY_EVIDENCE_DIR"):
        callbacks.append(ReplayForensicsCallback())
    stop_after = int(os.environ.get("BATA_REPLAY_STOP_AFTER_STEP", "0"))
    if stop_after:
        callbacks.append(StopAfterOptimizerStepCallback(stop_after))
    return callbacks
