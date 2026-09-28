"""Op-agnostic core of the spec abstraction.

A :class:`WeightSpec` knows the storage format of a weight tensor — its
dtype, scale layout, and any per-format setup. It does *not* know which
op consumes the weight (linear, embedding, MoE), which kernel runs, or
how the activation should be pre-processed. Op-specific concerns live on
separate Protocols, e.g. :class:`phyai.layers.quant.linear.LinearActivationQuant`.

The contract between a layer and its spec is :class:`AllocationRequest`.
Layers build the request from their own shape conventions (linear's
``(N, K)``, embedding's ``(V_per, D)``, …) and hand it over; specs only
ever see ``weight_shape`` plus a list of per-logical-matrix widths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch
import torch.nn as nn

from phyai.kernel.types import PhysicalSignature


@dataclass(frozen=True)
class AllocationRequest:
    """Op-agnostic "where & how big" packet handed to :meth:`WeightSpec.allocate`.

    ``weight_shape`` is the *local* (per-rank, post-fuse) shape. For a
    fused matmul that's ``(sum(logical_widths), in_per_rank)``; for a
    sharded vocab embedding it's ``(num_embeddings_per_partition, embedding_dim)``.

    ``logical_widths`` is the per-sub-matrix breakdown along the fused
    dim — ``[w]`` for a single unfused matrix, ``[gate, up]`` for a
    merged MLP, etc. ``fused_dim`` says which dim the widths sum over;
    both linear and embedding use ``0``.

    ``device`` is the device every parameter the spec allocates lands
    on. The layer fills it from its own resolved device (see each
    layer's ``device`` kwarg), which in turn defaults to
    :attr:`phyai.engine_config.EngineConfig.device`.

    ``extras`` is the controlled escape hatch for op-specific config a
    spec may want to look at without polluting the core fields. Prefer
    a typed Protocol over stuffing things here.
    """

    weight_shape: tuple[int, ...]
    logical_widths: list[int]
    fused_dim: int = 0
    params_dtype: torch.dtype = torch.bfloat16
    device: str | torch.device | None = None
    extras: Mapping[str, object] = field(default_factory=dict)


class WeightSpec(Protocol):
    """Op-agnostic core every spec satisfies.

    Implementations register parameters on ``layer`` (typically
    ``layer.weight`` plus any scales / zero points) and may attach
    spec-flavoured metadata such as ``layer.logical_widths``. They must
    NOT write op-specific names like ``layer.input_size_per_partition``
    — that's the layer's job.
    """

    spec_id: str
    weight_dtype: torch.dtype

    @property
    def physical_signature(self) -> PhysicalSignature: ...

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None: ...

    def process_after_loading(self, layer: nn.Module) -> None: ...


def physical_signature_for_spec(spec: object) -> PhysicalSignature:
    """Best-effort bridge for third-party specs that predate the property.

    Built-in specs expose a typed property; this fallback keeps custom
    WeightSpec implementations usable while they migrate.
    """

    value = getattr(spec, "physical_signature", None)
    if isinstance(value, PhysicalSignature):
        return value
    spec_id = str(getattr(spec, "spec_id", "bf16")).lower()
    if spec_id == "bf16":
        return PhysicalSignature(format="bf16", storage_dtype="bf16")
    if spec_id.startswith("fp8"):
        granularity = spec_id.removeprefix("fp8_")
        block_shape = None
        block_match = re.fullmatch(r"block_(\d+)_(\d+)", granularity)
        if block_match:
            block_shape = (int(block_match.group(1)), int(block_match.group(2)))
            granularity = "block"
        return PhysicalSignature(
            format="fp8_e4m3",
            granularity=granularity,
            block_shape=block_shape,
            storage_dtype="fp8_e4m3",
        )
    if spec_id.startswith("nvfp4"):
        layout = "128x4" if "128x4" in spec_id else "linear"
        return PhysicalSignature(format="nvfp4", layout=layout, storage_dtype="uint8")
    return PhysicalSignature(format=spec_id)


__all__ = ["AllocationRequest", "WeightSpec", "physical_signature_for_spec"]
