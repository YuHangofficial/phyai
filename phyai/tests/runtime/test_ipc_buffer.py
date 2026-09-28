"""IPC buffer tests, including a real cross-process CUDA IPC transfer."""

from __future__ import annotations

import multiprocessing as mp
import errno
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from phyai.runtime.ipc_buffer import (
    CudaIpcBuffer,
    CudaIpcHandle,
    HostShmBuffer,
    HostShmHandle,
)


# ``--import-mode=importlib`` does not make the repository's test tree an
# importable ``phyai.tests`` package.  Add this narrow directory to the child
# import path so multiprocessing ``spawn`` can resolve its top-level target.
_TEST_RUNTIME_DIR = str(Path(__file__).parent)
if _TEST_RUNTIME_DIR not in sys.path:
    sys.path.insert(0, _TEST_RUNTIME_DIR)

from ipc_worker_support import (  # noqa: E402
    write_cuda_ipc_buffer,
    write_host_shm_buffer,
)


def _start_process(process: mp.Process, *connections) -> None:
    """Start a spawned worker, translating sandbox process restrictions to skip."""
    try:
        process.start()
    except (PermissionError, OSError) as error:
        for connection in connections:
            connection.close()
        if (
            getattr(error, "errno", None) == errno.EPERM
            or "operation not permitted" in str(error).lower()
        ):
            pytest.skip(
                f"cross-process IPC is unavailable in this environment: {error}"
            )
        raise


def _require_cuda_ipc() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        import cuda.bindings.runtime  # noqa: F401
    except ImportError:
        pytest.skip("cuda-python is unavailable")


def _assert_child_result(connection, process: mp.Process) -> list[float] | list[int]:
    assert connection.poll(30), "IPC worker did not report within 30 seconds"
    status, payload = connection.recv()
    process.join(timeout=30)
    assert process.exitcode == 0
    assert status == "ok", payload
    return payload


@pytest.mark.ipc
def test_host_shared_memory_round_trips_across_spawned_processes():
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    buffer = HostShmBuffer.create(4 * np.dtype("int32").itemsize)
    values = buffer.as_numpy("int32", (4,))
    values[:] = (1, 2, 3, 4)
    process = context.Process(target=write_host_shm_buffer, args=(child,))
    _start_process(process, parent, child)
    child.close()
    try:
        parent.send(buffer.handle)
        payload = _assert_child_result(parent, process)
        assert payload == [11, 12, 13, 14]
        assert values.tolist() == [11, 12, 13, 14]
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        # Lifecycle: the creator's handle is typed, close is idempotent, and a
        # closed buffer refuses to hand out views.
        assert isinstance(buffer.handle, HostShmHandle) and buffer.is_creator
        buffer.close()
        buffer.close()
    with pytest.raises(RuntimeError, match="closed"):
        buffer.handle
    with pytest.raises(RuntimeError, match="closed"):
        buffer.as_numpy()


@pytest.mark.ipc
@pytest.mark.gpu
def test_cuda_ipc_buffer_crosses_process_boundary_and_preserves_writes():
    """A child maps the creator's allocation and writes through the same bytes."""
    _require_cuda_ipc()
    torch.cuda.set_device(0)
    with pytest.raises(TypeError, match="expected CudaIpcHandle"):
        CudaIpcBuffer.attach(object())
    with pytest.raises(ValueError, match="handle_bytes length"):
        CudaIpcBuffer.attach(
            CudaIpcHandle(handle_bytes=b"bad", nbytes=4, device_index=0)
        )
    buffer = CudaIpcBuffer.create(
        4 * torch.tensor([], dtype=torch.float32).element_size()
    )
    values = buffer.tensor(torch.float32, (4,))
    values.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"))
    torch.cuda.synchronize()

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=write_cuda_ipc_buffer, args=(child,))
    _start_process(process, parent, child)
    child.close()
    try:
        parent.send(buffer.handle)
        payload = _assert_child_result(parent, process)
        assert payload == pytest.approx([11.0, 12.0, 13.0, 14.0])
        torch.cuda.synchronize()
        assert values.cpu().tolist() == pytest.approx([11.0, 12.0, 13.0, 14.0])
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        buffer.close()


@pytest.mark.ipc
@pytest.mark.gpu
def test_cuda_registered_host_memory_supports_device_copy():
    """The optional pinned mapping path is usable by a CUDA stream."""
    _require_cuda_ipc()
    buffer = HostShmBuffer.create(
        4 * torch.tensor([], dtype=torch.float32).element_size(),
        cuda_register=True,
    )
    try:
        values = buffer.as_tensor(torch.float32, (4,))
        values.copy_(torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32))
        device_values = values.to(device="cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        assert device_values.cpu().tolist() == pytest.approx([5.0, 6.0, 7.0, 8.0])
    finally:
        buffer.close()
