"""GEMM backend Protocol.

Selection used to live here: a ``KernelProbe`` query packet, a ``can_handle``
predicate per backend, and a registry that ordered the survivors. All of that
now lives in :mod:`phyai.kernel` — each GEMM implementation is one catalog row
whose eligibility is a declarative predicate over typed facts, and the row
binds the concrete function directly.

What is left is the execution contract: a backend is a named collection of
matmul entry points, each taking the layer (for its weight and scale tensors),
the activations, and the bias.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch

from phyai.layers.quant.granularity import Granularity


#: The signature every GEMM entry point shares.
LinearFn = Callable[
    [torch.nn.Module, torch.Tensor, "torch.Tensor | None"], torch.Tensor
]


@runtime_checkable
class LinearKernel(Protocol):
    """A named collection of matmul entry points.

    There is no ``can_handle`` and no ``apply`` dispatch. Which function runs
    is decided by the catalog row that was selected, and that row holds a
    reference to the function itself — so "capability said yes, then execution
    raised on an unhandled format" is no longer expressible.
    """

    name: str


__all__ = ["Granularity", "LinearFn", "LinearKernel"]
