"""GlooBackend — ``torch.distributed.*`` for gloo process groups.

Sibling of :class:`NcclBackend`. Reasons gloo gets its own backend rather
than living inside ``NcclBackend``:

  - **Different API surface.** gloo does not support
    ``all_gather_into_tensor`` / ``reduce_scatter_tensor`` (the "tensor"
    variants); we have to use the list-based APIs (``all_gather``,
    ``reduce_scatter``) instead.
  - **Different scope.** gloo is the natural CPU collective backend and
    is also used as the bootstrap channel for PyNCCL's unique-id exchange.
    Keeping it isolated lets us register / un-register it independently.
  - **Different capture story.** gloo does not support cuda-graph capture
    (its work is on the host); ``supports_capture`` returns False.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from phyai.parallel.backend import Op, Topology
from phyai.parallel.backends.rank import group_rank_to_global
from phyai.parallel.state import Mode


def _backend_of(pg: dist.ProcessGroup) -> str:
    try:
        return dist.get_backend(pg)
    except Exception:
        return "?"


class GlooBackend:
    """Backend for gloo process groups."""

    name = "gloo"

    _OPS = {
        Op.ALL_REDUCE,
        Op.ALL_GATHER,
        Op.REDUCE_SCATTER,
        Op.BROADCAST,
        Op.ALL_TO_ALL,
        Op.SEND,
        Op.RECV,
        Op.BARRIER,
    }

    def __init__(self) -> None:
        self._handlers: dict[Op, callable] = {
            Op.ALL_REDUCE: self._all_reduce,
            Op.ALL_GATHER: self._all_gather,
            Op.REDUCE_SCATTER: self._reduce_scatter,
            Op.BROADCAST: self._broadcast,
            Op.ALL_TO_ALL: self._all_to_all,
            Op.SEND: self._send,
            Op.RECV: self._recv,
            Op.BARRIER: self._barrier,
        }

    def can_handle(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        pg: dist.ProcessGroup | None = None,
        mesh_name: str | None = None,
        group: str | None = None,
        **extra: object,
    ) -> bool:
        if op not in self._OPS:
            return False
        # Gloo is host-side; it makes no sense for a cuda-graph capture.
        if mode == Mode.GRAPH_CAPTURING:
            return False
        # Probe time (pg=None): be permissive so registry validation finds
        # at least one candidate.
        if pg is not None and _backend_of(pg) != dist.Backend.GLOO:
            return False
        return True

    def supports_capture(self) -> bool:
        return False

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        op: Op,
        pg: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor | None:
        handler = self._handlers.get(op)
        if handler is None:
            raise NotImplementedError(f"GlooBackend.execute: op={op}")
        return handler(pg, **kwargs)

    # ------------------------------------------------------------------
    # primitives — list-based variants for gloo
    # ------------------------------------------------------------------

    def _all_reduce(self, pg, *, input, output, reduce_op, **_):
        output.copy_(input)
        dist.all_reduce(output, op=reduce_op, group=pg)
        return output

    def _all_gather(self, pg, *, input, output, dim, **_):
        x = input.contiguous()
        ws = dist.get_world_size(group=pg)
        if dim == 0:
            # Build per-rank views into `output`. ``all_gather`` writes
            # into the buffer list passed in; we then copy back any
            # views that didn't share storage (in practice they do).
            chunk_views = list(output.chunk(ws, dim=0))
            buffers = [v.contiguous() for v in chunk_views]
            dist.all_gather(buffers, x, group=pg)
            for v, b in zip(chunk_views, buffers):
                if v.data_ptr() != b.data_ptr():
                    v.copy_(b)
            return output
        # dim != 0: gather into a stacked buffer, then reshape.
        stacked = torch.empty(
            (ws,) + tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
        )
        per_rank = [stacked[i] for i in range(ws)]
        dist.all_gather(per_rank, x, group=pg)
        out = stacked.movedim(0, dim).contiguous().flatten(dim, dim + 1)
        output.copy_(out)
        return output

    def _reduce_scatter(self, pg, *, input, output, dim, reduce_op, **_):
        x = input.contiguous()
        ws = dist.get_world_size(group=pg)
        chunks = [c.contiguous() for c in x.chunk(ws, dim=dim)]
        dist.reduce_scatter(output, chunks, op=reduce_op, group=pg)
        return output

    def _broadcast(self, pg, *, input, output, src, **_):
        output.copy_(input)
        dist.broadcast(output, src=group_rank_to_global(pg, src), group=pg)
        return output

    def _all_to_all(self, pg, *, input, output, in_splits, out_splits, **_):
        x = input.contiguous()
        ws = dist.get_world_size(group=pg)
        if in_splits is None:
            in_chunks = list(x.chunk(ws, dim=0))
        else:
            in_chunks = list(x.split(in_splits, dim=0))
        if out_splits is None:
            out_chunks = list(output.chunk(ws, dim=0))
        else:
            out_chunks = list(output.split(out_splits, dim=0))
        in_chunks = [c.contiguous() for c in in_chunks]
        out_buffers = [torch.empty_like(c) for c in out_chunks]
        dist.all_to_all(out_buffers, in_chunks, group=pg)
        for view, buf in zip(out_chunks, out_buffers):
            view.copy_(buf)
        return output

    def _send(self, pg, *, input, dst, **_):
        dist.send(input.contiguous(), dst=group_rank_to_global(pg, dst), group=pg)
        return None

    def _recv(self, pg, *, output, src, **_):
        dist.recv(output, src=group_rank_to_global(pg, src), group=pg)
        return output

    def _barrier(self, pg, **_):
        dist.barrier(group=pg)
        return None
