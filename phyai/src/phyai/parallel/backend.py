"""Op enum + Backend Protocol.

Backend authors implement this Protocol. Capability is a pure predicate
(``can_handle(...) -> bool``) — there is no ``score()`` method; priority
is the Registry's job.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist

from phyai.parallel.state import Mode
from phyai.parallel.topology import Topology


class Op(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    ALL_TO_ALL = "all_to_all"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"


@runtime_checkable
class Backend(Protocol):
    """Backend protocol. Pure capability + execute, no internal mode flags.

    ``can_handle`` is called only on Dispatcher cache miss; the hot path
    is the cache lookup. ``mesh_name`` / ``group`` carry the group identity
    so stateful backends (for example pynccl's per-``(mesh, group)``
    communicators) can decline groups they never attached; ``None`` means a
    capability probe (``registry.validate``) — answer for the general case.
    """

    name: str

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
    ) -> bool: ...

    def supports_capture(self) -> bool: ...

    def close(self) -> None:
        """Release anything the backend built outside torch.distributed.

        Called by :func:`phyai.parallel.shutdown` before the torch process
        group is destroyed. Backends that only wrap torch.distributed have
        nothing to release and return immediately.
        """
        ...

    def execute(
        self,
        *,
        op: Op,
        pg: dist.ProcessGroup,
        **kwargs: object,
    ) -> torch.Tensor | None:
        """Run ``op`` on ``pg``.

        ``kwargs`` carry the op's tensors and parameters (``input``,
        ``output``, ``dim``, ``reduce_op``, ...) plus two routing keys every
        op passes: ``_mesh_name`` and ``_group``, which stateful backends use
        to find the communicator they attached for that group.
        """
        ...


__all__ = ["Backend", "Op", "Topology"]
