"""Request schemas shared by all Cosmos3 replica executors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Cosmos3T2VRequest:
    """One tokenized Cosmos3 video or video-with-sound request."""

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    video_shape: tuple[int, int, int]
    fps: float = 24.0
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    noise: torch.Tensor | None = None
    seed: int = 42
    cond_latents: torch.Tensor | None = None
    cond_frame_indexes: tuple[int, ...] = ()
    sound_frames: int | None = None
    sound_dim: int = 64
    sound_latent_fps: float = 25.0


def pixel_to_latent_shape(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal: int = 4,
    spatial: int = 16,
) -> tuple[int, int, int]:
    """Convert a pixel grid to the VAE latent grid."""
    return (num_frames - 1) // temporal + 1, height // spatial, width // spatial


__all__ = ["Cosmos3T2VRequest", "pixel_to_latent_shape"]
