"""Parallel linear layers with declarative kernel selection.

Quick start::

    import torch.distributed as dist
    import phyai.parallel as P
    import phyai.layers.linear as L
    from phyai.engine_config import ParallelConfig, DenseParallelConfig

    dist.init_process_group("nccl")
    P.init(ParallelConfig(dense=DenseParallelConfig(tp_size=8)))

    qkv = L.QKVParallelLinear(
        hidden_size=4096, head_dim=128, num_heads=32, num_kv_heads=8,
        group="attention_tp", spec=L.Bf16Spec(),
    )
    o_proj = L.RowParallelLinear(
        in_features=4096, out_features=4096,
        group="attention_tp", sequence_parallel=True,
        spec=L.Fp8Spec(granularity=L.Granularity.PER_CHANNEL),
    )
"""

from __future__ import annotations

from phyai.layers.linear.backend import Granularity, LinearFn, LinearKernel
from phyai.layers.linear.layers import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    resolve_linear_kernel,
)
from phyai.layers.linear.spec import ActivationView, Bf16Spec, Fp8Spec, Nvfp4Spec
from phyai.layers.quant import AllocationRequest, WeightSpec


__all__ = [
    # layers
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "resolve_linear_kernel",
    # specs
    "Bf16Spec",
    "Fp8Spec",
    "Nvfp4Spec",
    "ActivationView",
    "AllocationRequest",
    "WeightSpec",
    "Granularity",
    # execution contract (selection lives in phyai.kernel)
    "LinearFn",
    "LinearKernel",
    # backends
]
