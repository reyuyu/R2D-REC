"""Shared bootstrap for this experiment's single-node four-GPU jobs."""
from __future__ import annotations

import os


SOCKET_IFNAME = "lo"
EXPECTED_WORLD_SIZE = 4


def _required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is required under torchrun")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def configure_single_node_nccl(
    *,
    initialize: bool = True,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
    torch_module=None,
    dist_module=None,
):
    """Bind NCCL bootstrap to loopback before creating the process group."""
    local_rank = _required_int("LOCAL_RANK")
    world_size = _required_int("WORLD_SIZE")
    if world_size != expected_world_size:
        raise RuntimeError(
            f"single-node job requires WORLD_SIZE={expected_world_size}, got {world_size}"
        )
    if local_rank not in range(expected_world_size):
        raise RuntimeError(
            f"LOCAL_RANK must be in 0..{expected_world_size - 1}, got {local_rank}"
        )

    # This task is explicitly single-node, so bootstrap needs no external NIC.
    # Set the interface before any process-group initialization.
    os.environ["NCCL_SOCKET_IFNAME"] = SOCKET_IFNAME
    if torch_module is None:
        import torch as torch_module
    if dist_module is None:
        import torch.distributed as dist_module

    torch_module.cuda.set_device(local_rank)
    device = torch_module.device(f"cuda:{local_rank}")
    initialized_here = False
    if initialize and not dist_module.is_initialized():
        dist_module.init_process_group(backend="nccl", device_id=device)
        initialized_here = True

    return {
        "nccl_socket_ifname": os.environ["NCCL_SOCKET_IFNAME"],
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "initialized_here": initialized_here,
        "local_rank_device_mapping": [
            {"local_rank": rank, "device": f"cuda:{rank}"}
            for rank in range(expected_world_size)
        ],
    }
