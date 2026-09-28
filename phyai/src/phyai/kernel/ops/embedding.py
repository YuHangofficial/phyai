"""Define sharded vocabulary embedding implementations."""

from __future__ import annotations

from phyai.kernel.facts import lib, dtype, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of

EMBEDDING = OpSpec(
    name="embedding",
    dims=("tokens", "embedding_dim"),
    dtypes=("input", "weight"),
    attributes=("shard_start", "shard_end"),
    signature="(input_ids, weight, shard_start, shard_end) -> Tensor",
    doc="Embedding lookup masked to this rank's vocabulary shard.",
)

FLOATS = frozenset({"bf16", "fp16", "fp32"})

TRITON_EMBEDDING = all_of(
    lib.has("phyai_kernel"),
    device.vendor == "nvidia",
    dtype.weight.in_(FLOATS),
)


def _triton(facts, params):
    from phyai_kernel import masked_embedding_lookup

    return masked_embedding_lookup


def _torch(facts, params):
    import torch

    def execute(input_ids, weight, shard_start, shard_end):
        mask = (input_ids >= shard_start) & (input_ids < shard_end)
        local = torch.where(mask, input_ids - shard_start, torch.zeros_like(input_ids))
        out = torch.nn.functional.embedding(local, weight)
        return out.masked_fill(~mask.unsqueeze(-1), 0)

    return execute


def register(catalog: Catalog) -> None:
    catalog.register_op(EMBEDDING)
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.embedding",
                op="embedding",
                priority=Priority.OPTIMIZED,
                when=TRITON_EMBEDDING,
                prepare=_triton,
                metadata={"package": "phyai-kernel"},
            ),
            Impl(
                kernel_id="torch.embedding",
                op="embedding",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.weight.is_set(),
                prepare=_torch,
                metadata={"package": "torch"},
            ),
        )
    )


__all__ = ["EMBEDDING", "register"]
