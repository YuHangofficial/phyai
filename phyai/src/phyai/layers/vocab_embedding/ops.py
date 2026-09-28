"""Custom-op wrapper for the masked vocab-parallel embedding lookup.

Two reasons this is wrapped as ``torch.library.custom_op`` rather than called
inline from the layer's ``forward``:

* **Graph capture stability.** Dynamo / ``torch.compile`` see an opaque op and
  do not re-trace the masking + gather path on every call.

* **Backend pluggability.** The default implementation forwards to the Triton
  kernel on CUDA and to a pure-PyTorch fallback elsewhere; a future CUDA
  graph or low-bit-weight kernel can replace either path without disturbing
  the layer code.

The shape contract is ``out.shape == input_ids.shape + (weight.shape[1],)``;
positions whose id falls outside ``[shard_start, shard_end)`` read as zero.
"""

from __future__ import annotations

import torch
from torch import Tensor

from phyai.kernel.call import CallSite


_EMBEDDING_CALL: CallSite | None = None


def get_embedding_call() -> CallSite:
    """The bound call site for the masked lookup.

    There is exactly one lookup in the process, so the binding is a singleton
    rather than per-layer state -- the op is a free function registered with
    ``torch.library``, with no module to hang it off.
    """

    global _EMBEDDING_CALL
    if _EMBEDDING_CALL is None:
        _EMBEDDING_CALL = CallSite("embedding", role="vocab.lookup")
    return _EMBEDDING_CALL


@torch.library.custom_op("phyai::masked_embedding_lookup", mutates_args=())
def _masked_embedding_lookup_op(
    input_ids: Tensor,
    weight: Tensor,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    handle = get_embedding_call().select(
        device=weight.device,
        dtype={
            "input": input_ids.dtype,
            "weight": weight.dtype,
            "output": weight.dtype,
        },
        dims={
            "tokens": input_ids.numel(),
            "embedding_dim": weight.shape[-1],
        },
        attrs={"shard_start": shard_start, "shard_end": shard_end},
    )
    return handle.execute(input_ids, weight, int(shard_start), int(shard_end))


@_masked_embedding_lookup_op.register_fake
def _(input_ids: Tensor, weight: Tensor, shard_start: int, shard_end: int) -> Tensor:
    out_shape = (*input_ids.shape, weight.shape[1])
    return torch.empty(out_shape, dtype=weight.dtype, device=weight.device)


def masked_embedding_lookup(
    input_ids: Tensor,
    weight: Tensor,
    *,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    """Gather ``weight[input_ids - shard_start]`` for in-shard positions, else 0.

    Out-of-shard positions return zero rows so that an all-reduce across TP
    ranks recovers the global embedding without an explicit second pass.
    """
    return torch.ops.phyai.masked_embedding_lookup.default(
        input_ids, weight, int(shard_start), int(shard_end)
    )


__all__ = ["get_embedding_call", "masked_embedding_lookup"]
