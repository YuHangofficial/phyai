"""Spawn targets for shared-memory and CUDA IPC tests."""

from __future__ import annotations

from typing import Any


def write_cuda_ipc_buffer(connection: Any) -> None:
    """Attach the received CUDA IPC handle, mutate it, and report completion."""
    buffer = None
    try:
        import torch

        from phyai.runtime.ipc_buffer import CudaIpcBuffer

        handle = connection.recv()
        torch.cuda.set_device(0)
        buffer = CudaIpcBuffer.attach(handle, device="cuda:0")
        values = buffer.tensor(torch.float32, (4,))
        values.add_(10.0)
        torch.cuda.synchronize()
        connection.send(("ok", values.cpu().tolist()))
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except BaseException:
            pass
    finally:
        if buffer is not None:
            buffer.close()
        connection.close()


def write_host_shm_buffer(connection: Any) -> None:
    """Attach a host shared-memory handle, mutate it, and report completion."""
    buffer = None
    try:
        from phyai.runtime.ipc_buffer import HostShmBuffer

        handle = connection.recv()
        buffer = HostShmBuffer.attach(handle)
        values = buffer.as_numpy("int32", (4,))
        values += 10
        connection.send(("ok", values.tolist()))
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except BaseException:
            pass
    finally:
        if buffer is not None:
            buffer.close()
        connection.close()


__all__ = ["write_cuda_ipc_buffer", "write_host_shm_buffer"]
