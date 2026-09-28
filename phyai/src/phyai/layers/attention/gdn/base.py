"""Types for FlashInfer-backed Gated Delta Net.

The metadata lifecycle is the shared
:mod:`phyai.layers.attention.contract`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from phyai.layers.attention.contract import (
    AttentionBackendBase,
    AttnPlanHandleBase,
    BaseAttnMetadata,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode


@dataclass(frozen=True)
class GatedDeltaNetMetadata(BaseAttnMetadata):
    """Host-side description of one GDN step."""

    cu_seqlens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode == AttnMode.IDLE:
            return
        if self.mode == AttnMode.MIXED:
            raise NotImplementedError("GatedDeltaNet does not support MIXED mode.")
        if self.mode == AttnMode.PREFILL and self.cu_seqlens is None:
            raise ValueError("GatedDeltaNetMetadata: PREFILL requires cu_seqlens.")
        if self.mode == AttnMode.DECODE and self.layout != AttnLayout.PADDED_4D:
            raise ValueError("GatedDeltaNetMetadata: DECODE requires PADDED_4D layout.")


class GatedDeltaNetPlanHandle(AttnPlanHandleBase):
    """Backend-private per-step state for GDN."""


@dataclass(frozen=True)
class GatedDeltaNetCtx:
    """Per-call GDN context owned by the model runner.

    ``state`` is either a per-batch state tensor or a state pool. Decode
    selects pool mode when ``state_indices`` is present. Prefill accepts only
    per-batch initial state and writes its final state to ``output_state``.
    ``output`` optionally supplies a preallocated kernel output buffer.
    """

    backend: "GatedDeltaNetBackend"
    plan: GatedDeltaNetPlanHandle
    mode: AttnMode
    layout: AttnLayout
    cu_seqlens: torch.Tensor | None = None
    state: torch.Tensor | None = None
    output_state: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None
    output_state_indices: torch.Tensor | None = None
    output: torch.Tensor | None = None


@runtime_checkable
class GatedDeltaNetLayerProto(Protocol):
    """Static configuration read by a GDN backend."""

    num_query_heads: int
    num_key_heads: int
    num_value_heads: int
    num_state_heads: int
    head_dim: int
    scale: float
    use_qk_l2norm: bool


class GatedDeltaNetBackend(
    AttentionBackendBase[GatedDeltaNetMetadata, GatedDeltaNetPlanHandle]
):
    """ABC for Gated Delta Net kernel backends."""

    @abstractmethod
    def forward(
        self,
        layer: GatedDeltaNetLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ctx: GatedDeltaNetCtx,
    ) -> torch.Tensor:
        """Run one prefill or decode GDN step."""


__all__ = [
    "GatedDeltaNetBackend",
    "GatedDeltaNetCtx",
    "GatedDeltaNetLayerProto",
    "GatedDeltaNetMetadata",
    "GatedDeltaNetPlanHandle",
]
