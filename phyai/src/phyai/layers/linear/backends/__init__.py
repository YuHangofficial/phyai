"""GEMM entry points, one module per backend.

Each module holds plain functions with the ``(layer, x, bias) -> Tensor``
signature. They do not register themselves and they do not decide when they
run: :mod:`phyai.kernel.ops.gemm` declares one catalog row per storage format,
naming the entry point and the conditions under which it is eligible.

That separation is the point. These modules used to hold a class each, with a
``can_handle`` method that tested ``spec_id.startswith(...)`` and an ``apply``
method that re-dispatched on the same strings — so capability could answer yes
and execution could then raise. Eligibility now lives in one declarative place
and each function is reached only for the format it was written for.

Nothing is re-exported here. Importing a backend costs an ``import torch`` at
minimum and an ``import flashinfer`` at worst, and a ``prepare`` in the catalog
imports exactly the one it needs, on first use.
"""

from __future__ import annotations

__all__: list[str] = []
