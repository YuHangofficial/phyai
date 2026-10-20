"""Build temporal and spatial position IDs for Cosmos3 visual/action inputs."""

from __future__ import annotations

import math

import torch


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """3-D (t, h, w) ids for vision tokens; spatial ids reset per frame.

    With fps modulation, the temporal axis is rescaled by
    ``base_fps/base_tcf`` over ``fps/tcf`` so videos at different frame rates
    share a comparable temporal coordinate. Flattened t-major.
    """
    fps_modulation = enable_fps_modulation and fps is not None
    if fps_modulation:
        tps = fps / temporal_compression_factor
        effective_base_tcf = (
            base_temporal_compression_factor
            if base_temporal_compression_factor is not None
            else temporal_compression_factor
        )
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (
            ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + int(temporal_offset)
            + start_frame_offset
        )
    h_index = (
        torch.arange(grid_h, dtype=torch.long)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )
    if fps_modulation:
        mrope_ids = torch.stack(
            [t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0
        )
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_offset = math.floor(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


__all__ = ["compute_mrope_position_ids_vision"]
