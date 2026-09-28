"""Define fused causal depthwise convolution implementations."""

from __future__ import annotations

from phyai.kernel.facts import lib, dtype, shape, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of

CAUSAL_CONV = OpSpec(
    name="causal_conv",
    dims=("tokens", "channels", "kernel"),
    dtypes=("input",),
    attributes=("activation", "split"),
    signature="(x, weight, split_sizes) -> tuple[Tensor, ...]",
    doc="Causal conv1d + SiLU + split, as used by the Qwen3.5 mixer.",
)

FLOATS = frozenset({"bf16", "fp16", "fp32"})

TRITON_CONV = all_of(
    lib.has("phyai_kernel"),
    device.vendor == "nvidia",
    dtype.input.in_(FLOATS),
    shape.kernel.between(1, 8),
)


def _triton(facts, params):
    from phyai_kernel import causal_conv1d_silu_split_qkv

    return causal_conv1d_silu_split_qkv


def _torch(facts, params):
    import torch

    def execute(x, weight, split_sizes):
        # Convert between token-major and Conv1d channel-major layouts.
        channels = x.shape[-1]
        kernel = weight.shape[-1]
        y = torch.nn.functional.conv1d(
            x.transpose(1, 2),
            weight,
            bias=None,
            stride=1,
            padding=kernel - 1,
            groups=channels,
        )[:, :, : x.shape[1]]
        y = torch.nn.functional.silu(y).transpose(1, 2)
        return torch.split(y, tuple(int(size) for size in split_sizes), dim=-1)

    return execute


def register(catalog: Catalog) -> None:
    catalog.register_op(CAUSAL_CONV)
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.causal_conv",
                op="causal_conv",
                priority=Priority.OPTIMIZED,
                when=TRITON_CONV,
                prepare=_triton,
                metadata={"package": "phyai-kernel"},
            ),
            Impl(
                kernel_id="torch.causal_conv",
                op="causal_conv",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch,
                metadata={"package": "torch"},
            ),
        )
    )


__all__ = ["CAUSAL_CONV", "register"]
