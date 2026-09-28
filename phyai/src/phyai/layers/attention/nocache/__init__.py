"""`phyai.layers.attention.nocache` — stateless prefill attention.

ViT-style attention, no KV cache, no per-layer state. The
:class:`Attention` layer takes already-projected Q/K/V and returns an
attention output in the same layout. Backends:

* ``"flashinfer"`` (default) — single-prefill / batch-prefill ragged.
* ``"sdpa"`` — :func:`torch.nn.functional.scaled_dot_product_attention`,
  optionally compiled.
* ``"eager"`` — pure PyTorch reference.

Sibling stacks: :mod:`phyai.layers.attention.ar` and
:mod:`phyai.layers.attention.diffusion` (paged-KV attention).
"""

from __future__ import annotations

from phyai.layers.attention.nocache.backends import (
    EagerAttentionBackend,
    EagerAttentionPlan,
    FlashInferAttentionBackend,
    FlashInferAttentionPlan,
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)
from phyai.layers.attention.nocache.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.nocache.layer import Attention


__all__ = [
    "Attention",
    "AttentionBackend",
    "AttentionLayerProto",
    "AttnCtx",
    "AttnMetadata",
    "AttnPlanHandle",
    "EagerAttentionBackend",
    "EagerAttentionPlan",
    "FlashInferAttentionBackend",
    "FlashInferAttentionPlan",
    "SdpaAttentionBackend",
    "SdpaAttentionPlan",
]
