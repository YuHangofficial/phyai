"""Discrete ``torch.distributed`` bootstrap entry point."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from phyai.utils.cuda import resolve_device


def init_dist(
    *,
    world_size: int,
    device_type: str,
    timeout: timedelta | None = None,
    require_launcher: bool = False,
    device: torch.device | str | None = None,
) -> bool:
    """Bring up the process group for the requested ``world_size``.

    ``world_size`` is the rank count of one complete model replica — the
    resolved rank count for the configured model-parallel groups.
    Serving replicas are outside this process group.

    Args:
        world_size: total rank count for the global mesh.
        device_type: ``"cuda"`` / ``"cpu"``; picks the backend and decides
            whether a device gets pinned.
        timeout: collective timeout handed to ``init_process_group``.
            ``None`` keeps torch's own default. Engine passes
            :attr:`~phyai.engine_config.RuntimeConfig.dist_timeout_s`.
        require_launcher: require ``RANK``/``WORLD_SIZE``/rendezvous
            environment variables for a multi-rank group. Engine workers set
            this flag so a missing launcher cannot silently create duplicate
            rank-zero processes.
        device: the device target this rank should own. An explicit index
            (``"cuda:3"``) is pinned as-is; ``None`` or a bare ``"cuda"``
            resolves through ``LOCAL_RANK`` exactly as before. Engine passes
            its configured ``device.target`` so a managed worker bound to
            ``cuda:i`` is not rebound to device 0 here.

    Returns
    -------
    bool
        ``True`` if this call **owns** the process group (created it
        and is responsible for ``dist.destroy_process_group()`` on
        shutdown). ``False`` if a group was already up or single-rank
        short-circuited.

    Behaviour
    ---------
    * ``world_size == 1`` with no existing group -> no-op, returns ``False``.
    * ``world_size > 1`` with no existing group -> spin one up
      (``nccl`` for cuda, ``gloo`` for cpu) and return ``True``.
    * Any ``world_size`` with an existing group -> reuse it; raise
      :class:`ValueError` if its world size disagrees with the request.
    """
    if not dist.is_initialized():
        if world_size == 1:
            if device_type == "cuda":
                torch.cuda.set_device(resolve_device(device or "cuda"))
            return False

        backend = "nccl" if device_type == "cuda" else "gloo"
        # WorkerSupervisor or an external launcher populates the rendezvous
        # environment so each process has its own replica-local rank.
        launcher_keys = ("RANK", "WORLD_SIZE")
        missing = tuple(key for key in launcher_keys if key not in os.environ)
        if require_launcher and missing:
            raise RuntimeError(
                "multi-rank Engine workers require launcher environment "
                f"variables {launcher_keys!r}; missing {missing!r}."
            )
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        try:
            rank = int(os.environ["RANK"])
            launcher_world_size = int(os.environ["WORLD_SIZE"])
        except ValueError as error:
            raise RuntimeError(
                "RANK and WORLD_SIZE must be integer launcher values."
            ) from error
        if launcher_world_size != world_size:
            raise ValueError(
                f"launcher WORLD_SIZE={launcher_world_size} does not match "
                f"requested world_size={world_size}."
            )
        if rank < 0 or rank >= world_size:
            raise ValueError(f"launcher RANK={rank} is outside [0, {world_size}).")
        # Pin the device first: NCCL reads the current device while building
        # its communicator, so binding after init_process_group leaves a
        # window where the group and this rank's allocations disagree.
        if backend == "nccl":
            torch.cuda.set_device(resolve_device(device or "cuda"))
        kwargs = {} if timeout is None else {"timeout": timeout}
        dist.init_process_group(backend, rank=rank, world_size=world_size, **kwargs)
        return True

    actual = dist.get_world_size()
    if actual != world_size:
        raise ValueError(
            f"init_dist: world_size={world_size} does not match the existing "
            f"process-group world_size={actual}. Either set the parallel sizes "
            f"to match the launcher, or destroy the current process group first."
        )
    return False


__all__ = ["init_dist"]
