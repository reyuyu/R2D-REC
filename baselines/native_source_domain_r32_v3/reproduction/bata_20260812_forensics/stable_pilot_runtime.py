"""Production-like deterministic runtime controls for BATA-STABLE-V0.

This module deliberately contains no gradient, parameter, or DDP communication
instrumentation. It validates the frozen runtime once, resumes a SHA-pinned
checkpoint, and asks Trainer to save and stop at the requested optimizer step.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback


EXPECTED_STOP_STEP = 560
EXPECTED_SCHEDULER_HORIZON = 1106
EXPECTED_WORLD_SIZE = 4
EXPECTED_TORCH_FLAGS = {
    "allow_tf32_matmul": False,
    "allow_tf32_cudnn": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "float32_matmul_precision": "highest",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_stable_runtime() -> dict[str, Any]:
    """Freeze the FA2-deterministic runtime before CUDA is initialized."""

    if os.environ.get("BATA_STABLE_V0") != "1":
        return {"enabled": False}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("BATA-STABLE-V0 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") != "1":
        raise RuntimeError("BATA-STABLE-V0 requires FLASH_ATTENTION_DETERMINISTIC=1")
    if torch.cuda.is_initialized():
        raise RuntimeError("BATA-STABLE-V0 deterministic controls were configured after CUDA init")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = EXPECTED_TORCH_FLAGS["allow_tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = EXPECTED_TORCH_FLAGS["allow_tf32_cudnn"]
    torch.backends.cudnn.benchmark = EXPECTED_TORCH_FLAGS["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = EXPECTED_TORCH_FLAGS["cudnn_deterministic"]
    torch.set_float32_matmul_precision(EXPECTED_TORCH_FLAGS["float32_matmul_precision"])
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms did not become active")
    return {
        "enabled": True,
        "warn_only": False,
        "configured_before_cuda": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "flash_attention_deterministic": os.environ["FLASH_ATTENTION_DETERMINISTIC"],
        **EXPECTED_TORCH_FLAGS,
    }


RUNTIME_SETUP = configure_stable_runtime()


def _expected_checkpoint() -> tuple[Path, dict[str, str]]:
    checkpoint = Path(os.environ["BATA_STABLE_CHECKPOINT"])
    expected_path = Path(os.environ["BATA_STABLE_EXPECTED_CHECKPOINT_SHA"])
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = {name: sha256_file(checkpoint / name) for name in expected}
    if actual != expected:
        raise RuntimeError("BATA-STABLE-V0 historical checkpoint SHA contract mismatch")
    return checkpoint, expected


def install_trusted_checkpoint_compatibility() -> None:
    """Restore torch-2.5 checkpoint loading while retaining weights_only=True."""

    if os.environ.get("BATA_STABLE_V0") != "1":
        return
    _expected_checkpoint()
    import transformers.trainer as transformers_trainer
    from numpy._core.multiarray import _reconstruct

    torch.serialization.add_safe_globals(
        [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    )
    transformers_trainer.check_torch_load_is_safe = lambda: None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_snapshot() -> dict[str, Any]:
    selected_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key in {
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "FLASH_ATTENTION_DETERMINISTIC",
            "NCCL_IB_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        }
    }
    return {
        "recipe": "BATA-STABLE-V0",
        "runtime_setup": RUNTIME_SETUP,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": _package_version("transformers"),
        "flash_attn": _package_version("flash-attn"),
        "liger_kernel": _package_version("liger-kernel"),
        "environment": selected_env,
        "torch_flags": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "gpu_topology": _command_output(["nvidia-smi", "topo", "-m"]),
    }


class StableStopAndSaveCallback(TrainerCallback):
    """Save checkpoint-560 and stop while preserving max_steps=1106."""

    def __init__(self, stop_step: int = EXPECTED_STOP_STEP) -> None:
        self.stop_step = int(stop_step)
        self.started_at = time.time()

    @property
    def evidence_path(self) -> Path:
        root = Path(os.environ["BATA_STABLE_EVIDENCE_DIR"])
        return root / f"runtime_rank{int(os.environ.get('RANK', '0'))}.json"

    def _write(self, payload: dict[str, Any]) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.evidence_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.evidence_path)

    def on_train_begin(self, args, state, control, **kwargs):
        if int(state.max_steps) != EXPECTED_SCHEDULER_HORIZON:
            raise RuntimeError(
                f"BATA-STABLE-V0 requires scheduler horizon {EXPECTED_SCHEDULER_HORIZON}, "
                f"got {state.max_steps}"
            )
        if int(state.global_step) != 553:
            raise RuntimeError(f"BATA-STABLE-V0 must resume at step 553, got {state.global_step}")
        if int(os.environ.get("WORLD_SIZE", "1")) != EXPECTED_WORLD_SIZE:
            raise RuntimeError("BATA-STABLE-V0 requires exactly four ranks")
        checkpoint, expected = _expected_checkpoint()
        snapshot = runtime_snapshot()
        snapshot.update(
            {
                "status": "RUNNING",
                "start_global_step": int(state.global_step),
                "scheduler_horizon": int(state.max_steps),
                "stop_step": self.stop_step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected,
            }
        )
        self._write(snapshot)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        payload = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "status": "PASS" if int(state.global_step) == self.stop_step else "STOPPED",
                "completed_global_step": int(state.global_step),
                "wall_seconds": time.time() - self.started_at,
            }
        )
        self._write(payload)
        return control


def stable_callbacks() -> list[TrainerCallback]:
    if os.environ.get("BATA_STABLE_V0") != "1":
        return []
    stop_step = int(os.environ.get("BATA_STABLE_STOP_STEP", str(EXPECTED_STOP_STEP)))
    if stop_step != EXPECTED_STOP_STEP:
        raise RuntimeError(f"BATA-STABLE-V0 stop step must remain {EXPECTED_STOP_STEP}")
    return [StableStopAndSaveCallback(stop_step)]
