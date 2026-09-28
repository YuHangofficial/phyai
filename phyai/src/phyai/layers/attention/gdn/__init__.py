"""Gated Delta Net with backend dispatch."""

from __future__ import annotations

from phyai.layers.attention.gdn.backends import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
    FlashQlaGatedDeltaNetBackend,
    FlashQlaGatedDeltaNetPlan,
)
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn.layer import GatedDeltaNet


__all__ = [
    "FlaGatedDeltaNetBackend",
    "FlaGatedDeltaNetPlan",
    "FlashInferGatedDeltaNetBackend",
    "FlashInferGatedDeltaNetPlan",
    "FlashQlaGatedDeltaNetBackend",
    "FlashQlaGatedDeltaNetPlan",
    "GatedDeltaNet",
    "GatedDeltaNetBackend",
    "GatedDeltaNetCtx",
    "GatedDeltaNetLayerProto",
    "GatedDeltaNetMetadata",
    "GatedDeltaNetPlanHandle",
]
