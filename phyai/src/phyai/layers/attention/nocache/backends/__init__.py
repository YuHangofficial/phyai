"""Concrete no-cache attention backend implementations.

Each ``<vendor>.py`` module owns one backend class (and the plan
handle types it consumes). The kernel catalog
(:mod:`phyai.kernel.ops.attention`) constructs these classes directly;
this package only surfaces them through the names below.

Three backend names — ``"sdpa"`` / ``"flashinfer"`` / ``"eager"``.
"""

from __future__ import annotations

from phyai.layers.attention.nocache.backends.eager import (
    EagerAttentionBackend,
    EagerAttentionPlan,
)
from phyai.layers.attention.nocache.backends.flashinfer import (
    FlashInferAttentionBackend,
    FlashInferAttentionPlan,
)
from phyai.layers.attention.nocache.backends.sdpa import (
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)


__all__ = [
    "EagerAttentionBackend",
    "EagerAttentionPlan",
    "FlashInferAttentionBackend",
    "FlashInferAttentionPlan",
    "SdpaAttentionBackend",
    "SdpaAttentionPlan",
]
