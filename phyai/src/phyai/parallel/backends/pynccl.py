"""PyNCCLBackend — direct ctypes call to libnccl, bypassing
``torch.distributed`` host-side machinery.

Two layers:

* ``_PyNcclComm``: one NCCL communicator bound to a specific subgroup of
  ranks and a specific CUDA device. Built once via a CPU-side gloo group
  for unique_id bootstrap.
* ``PyNCCLBackend``: implements the ``Backend`` protocol; holds a dict of
  ``_PyNcclComm`` keyed by ``(mesh_name, group)``, built eagerly by
  :meth:`attach` during ``phyai.parallel.init``.

Why bypass ``torch.distributed`` even when it is capture-compatible?
Two practical reasons:

1. Kernel launches go directly to the **caller's current stream**, so
   overlapping on a side stream (``torch.cuda.stream``) is straightforward.
2. No PG watchdog thread / event polling overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from phyai.parallel.backend import Op, Topology
from phyai.parallel.state import Mode
from phyai.parallel.backends.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from phyai.utils import get_logger

if TYPE_CHECKING:
    from phyai.parallel.mesh import Mesh

logger = get_logger(__name__)


class _PyNcclComm:
    """One NCCL communicator bound to a (subgroup, device) pair."""

    def __init__(
        self,
        device_group: ProcessGroup,
        cpu_group: ProcessGroup,
        device: torch.device,
        nccl: NCCLLibrary,
    ) -> None:
        self.rank = dist.get_rank(device_group)
        self.world_size = dist.get_world_size(device_group)
        self.device = device
        self.nccl = nccl

        if self.world_size == 1:
            self.comm: ncclComm_t | None = None
            return

        # rank 0 of the subgroup creates the unique id, broadcasts via cpu_group
        if self.rank == 0:
            uid = nccl.ncclGetUniqueId()
        else:
            uid = ncclUniqueId()
        tensor = torch.ByteTensor(list(uid.internal))
        ranks = dist.get_process_group_ranks(cpu_group)
        dist.broadcast(tensor, src=ranks[0], group=cpu_group)
        for i, b in enumerate(tensor.tolist()):
            uid.internal[i] = b

        with torch.cuda.device(device):
            self.comm = nccl.ncclCommInitRank(self.world_size, uid, self.rank)
            # Warmup AR on a side stream — pulls all NCCL lazy init out of
            # the way so cuda graph capture doesn't see one-time side effects.
            warmup = torch.cuda.Stream()
            with torch.cuda.stream(warmup):
                data = torch.zeros(1, device=device)
                self._all_reduce(data, ReduceOp.SUM, stream=warmup)
            warmup.synchronize()
            del data

    def _stream(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream()

    # ------------------------------------------------------------------
    # primitive bindings (all launch on caller's current stream)
    # ------------------------------------------------------------------

    def _all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Reduce ``tensor`` in place."""
        s = stream or self._stream()
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return tensor

    def _all_gather(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclAllGather(
            buffer_type(input.data_ptr()),
            buffer_type(output.data_ptr()),
            input.numel(),
            ncclDataTypeEnum.from_torch(input.dtype),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _reduce_scatter(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        op: ReduceOp,
    ) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclReduceScatter(
            buffer_type(input.data_ptr()),
            buffer_type(output.data_ptr()),
            output.numel(),
            ncclDataTypeEnum.from_torch(input.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _broadcast(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        s = self._stream()
        send_ptr = buffer_type(input.data_ptr()) if src == self.rank else buffer_type()
        self.nccl.ncclBroadcast(
            send_ptr,
            buffer_type(output.data_ptr()),
            output.numel(),
            ncclDataTypeEnum.from_torch(output.dtype),
            src,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _send(self, tensor: torch.Tensor, dst: int) -> None:
        s = self._stream()
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )

    def _recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return tensor

    def destroy(self) -> None:
        if self.comm is not None:
            try:
                self.nccl.ncclCommDestroy(self.comm)
            except Exception as e:
                logger.warning("ncclCommDestroy failed: %s", e)
            self.comm = None


class PyNCCLBackend:
    """Backend that drives NCCL via a direct ctypes binding.

    Capture-safe; in eager mode it returns False from ``can_handle`` so the
    Dispatcher falls through to ``NcclBackend`` (which has the better
    watchdog story for production eager paths). Communicators are built
    per membership by :meth:`attach`; logical groups with identical members
    share one communicator, which assumes their collectives never run
    concurrently on different streams (see ``ProcessGroupPool``). Unattached
    groups fall through to ``NcclBackend``.
    """

    name = "pynccl"

    _OPS = {
        Op.ALL_REDUCE,
        Op.ALL_GATHER,
        Op.REDUCE_SCATTER,
        Op.BROADCAST,
        Op.SEND,
        Op.RECV,
    }

    def __init__(
        self,
        *,
        library_path: str | None = None,
        prefer_in_eager: bool = False,
    ) -> None:
        self._library_path = library_path
        self._nccl: NCCLLibrary | None = None
        self._comms: dict[tuple[str, str], _PyNcclComm] = {}
        self._prefer_in_eager = prefer_in_eager
        self._handlers: dict[Op, callable] = {
            Op.ALL_REDUCE: self._h_all_reduce,
            Op.ALL_GATHER: self._h_all_gather,
            Op.REDUCE_SCATTER: self._h_reduce_scatter,
            Op.BROADCAST: self._h_broadcast,
            Op.SEND: self._h_send,
            Op.RECV: self._h_recv,
        }

    # --- Backend protocol --------------------------------------------------

    def can_handle(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        mesh_name: str | None = None,
        group: str | None = None,
        **extra: object,
    ) -> bool:
        if op not in self._OPS:
            return False
        if mode == Mode.EAGER and not self._prefer_in_eager:
            return False
        if world_size <= 1:
            return False
        if (
            mesh_name is not None
            and group is not None
            and (mesh_name, group) not in self._comms
        ):
            # Communicators are per-(mesh, group); decline anything attach()
            # never built so the registry falls through to NcclBackend.
            # None means capability probe (registry.validate) — stay general.
            return False
        return True

    def supports_capture(self) -> bool:
        return True

    def close(self) -> None:
        """Destroy every attached NCCL communicator.

        Must run before ``torch.distributed.destroy_process_group``: the
        communicators are independent of torch's groups, so nothing else
        frees them, and a later ``attach`` in the same process would otherwise
        leak the old ones.
        """
        for comm in dict.fromkeys(self._comms.values()):
            comm.destroy()
        self._comms.clear()

    def execute(
        self,
        *,
        op: Op,
        pg: ProcessGroup,
        **kwargs,
    ) -> torch.Tensor | None:
        handler = self._handlers.get(op)
        if handler is None:
            raise NotImplementedError(f"PyNCCLBackend.execute: op={op}")
        comm = self._comm_for(
            pg, mesh_name=kwargs["_mesh_name"], group=kwargs["_group"]
        )
        return handler(comm, **kwargs)

    # --- per-op handlers (take a `comm` first arg) ------------------------

    def _h_all_reduce(self, comm, *, input, output, reduce_op, **_):
        # NCCL reads and writes raw pointers linearly, so a non-contiguous
        # input (a sliced view, say) would be reduced in the wrong memory
        # order — silently. Stage through the dense `output` buffer the op
        # layer allocated and reduce in place, exactly as NcclBackend does.
        output.copy_(input)
        return comm._all_reduce(output, reduce_op)

    def _h_all_gather(self, comm, *, input, output, dim, **_):
        # ncclAllGather concatenates rank blocks along dim 0. For dim != 0,
        # gather into a rank-major staging buffer, then move the rank dimension
        # into place — the same rearrangement NcclBackend uses; plain device
        # kernels, so it is graph-capture safe. `output` arrives preallocated
        # in the final layout (phyai.parallel.ops), never in dim-0 layout.
        x = input.contiguous()
        if dim == 0:
            return comm._all_gather(x, output)
        stacked = torch.empty(
            (comm.world_size, *x.shape), dtype=x.dtype, device=x.device
        )
        comm._all_gather(x, stacked)
        output.copy_(stacked.movedim(0, dim).contiguous().flatten(dim, dim + 1))
        return output

    def _h_reduce_scatter(self, comm, *, input, output, dim, reduce_op, **_):
        # ncclReduceScatter consumes world_size contiguous rank blocks. For
        # dim != 0, restack the per-rank chunks along a leading rank dimension so
        # rank r receives the reduction of chunk r (capture-safe).
        x = input.contiguous()
        if dim == 0:
            return comm._reduce_scatter(x, output, reduce_op)
        stacked = torch.stack(x.chunk(comm.world_size, dim=dim), dim=0)
        return comm._reduce_scatter(stacked, output, reduce_op)

    def _h_broadcast(self, comm, *, input, output, src, **_):
        # Same staging as all_reduce: the source rank's payload must be dense
        # before its pointer is handed to NCCL; in-place broadcast is legal.
        output.copy_(input)
        return comm._broadcast(output, output, src)

    def _h_send(self, comm, *, input, dst, **_):
        comm._send(input.contiguous(), dst)
        return None

    def _h_recv(self, comm, *, output, src, **_):
        return comm._recv(output, src)

    # --- lifecycle --------------------------------------------------------

    def attach(self, mesh: "Mesh", groups: list[str], *, device: torch.device) -> None:
        """Build per-group comms eagerly for every group we expect to use.

        Eager construction sidesteps the lazy-init-during-capture trap
        that PyTorch's ProcessGroupNCCL also exhibits — pre-warming
        keeps NCCL's one-time side effects out of the recorded graph.

        The mesh already owns matching CPU bootstrap groups, created in the
        same order on every rank. Singleton collectives need no communicator.
        """
        if self._nccl is None:
            self._nccl = NCCLLibrary(self._library_path)
        shared = {
            mesh.group_members(name): comm
            for (mesh_name, name), comm in self._comms.items()
            if mesh_name == mesh.name
        }
        for group in groups:
            if mesh.group_size(group) <= 1:
                continue
            key = (mesh.name, group)
            if key in self._comms:
                continue
            members = mesh.group_members(group)
            if members not in shared:
                shared[members] = _PyNcclComm(
                    device_group=mesh.group(group),
                    cpu_group=mesh.cpu_group(group),
                    device=device,
                    nccl=self._nccl,
                )
            self._comms[key] = shared[members]

    def attached(self) -> dict[tuple[str, str], torch.device]:
        """``(mesh_name, group) -> bound device`` for every attached comm."""
        return {key: comm.device for key, comm in self._comms.items()}

    def _comm_for(self, pg: ProcessGroup, *, mesh_name: str, group: str) -> _PyNcclComm:
        key = (mesh_name, group)
        comm = self._comms.get(key)
        if comm is None:
            raise RuntimeError(
                f"PyNCCLBackend.attach() was not called for {key}. "
                "Pass `pynccl_groups=[...]` to phyai.parallel.init()."
            )
        return comm
