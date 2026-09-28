"""NVML topology helpers."""

from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch


def nvml_available() -> bool:
    try:
        import pynvml  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


@contextmanager
def nvml_session() -> Iterator[object]:
    """Initialize NVML for the block and shut it down afterwards."""
    import pynvml  # noqa: PLC0415

    pynvml.nvmlInit()
    try:
        yield pynvml
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def node_identity() -> str:
    """Stable identity of the machine this process runs on.

    The kernel boot id is shared by every container on a host and differs
    between hosts, which is exactly the "same node" notion collectives care
    about; hostnames can collide across containers, so they are a fallback.
    """
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return socket.gethostname()


def physical_device_index(logical: int) -> int:
    """NVML (physical) index of torch device ``logical``.

    Honours ``CUDA_VISIBLE_DEVICES`` including UUID entries; identity when no
    mask is set. Under full device visibility with index binding this is the
    same mapping every process of a deployment sees.
    """
    return int(torch.cuda._get_nvml_device_index(logical))


def nvlink_fully_connected(physical_ids: Sequence[int]) -> bool:
    """Whether every pair of the given local GPUs has an NVLink P2P path.

    This is the property NCCL's NVLink transport and custom all-reduce kernels
    rely on. A single GPU is trivially connected.
    """
    unique = list(dict.fromkeys(int(i) for i in physical_ids))
    if len(unique) < 2:
        return True
    with nvml_session() as nvml:
        handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in unique]
        for index, first in enumerate(handles):
            for second in handles[index + 1 :]:
                status = nvml.nvmlDeviceGetP2PStatus(
                    first, second, nvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                if status != nvml.NVML_P2P_STATUS_OK:
                    return False
    return True


def fabric_clique(physical_id: int) -> str | None:
    """Identity of the NVLink fabric domain this GPU is part of, or ``None``.

    Only NVSwitch-connected systems whose fabric manager finished training
    report one (state ``COMPLETED``). Ranks on different hosts that share a
    clique are NVLink-connected across those hosts (GB200 NVL72 style).
    """
    try:
        with nvml_session() as nvml:
            handle = nvml.nvmlDeviceGetHandleByIndex(int(physical_id))
            info = nvml.nvmlDeviceGetGpuFabricInfo(handle)
            completed = getattr(nvml, "NVML_GPU_FABRIC_STATE_COMPLETED", 3)
    except Exception:
        return None
    if getattr(info, "state", None) != completed:
        return None
    cluster = bytes(bytearray(info.clusterUuid)).hex()
    return f"{cluster}:{int(info.cliqueId)}"


__all__ = [
    "fabric_clique",
    "node_identity",
    "nvlink_fully_connected",
    "nvml_available",
    "nvml_session",
    "physical_device_index",
]
