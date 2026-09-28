"""Action request schema for the Cosmos3 policy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import torch


ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")


@dataclass
class Cosmos3ActionRequest:
    """One policy, forward-dynamics, or inverse-dynamics request."""

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    video_shape: tuple[int, int, int]
    mode: str
    domain_id: int
    action_chunk: int
    raw_action_dim: int
    action_dim: int = 64
    cond_video_latents: torch.Tensor | None = None
    cond_video_pixels: torch.Tensor | None = None
    cond_action: torch.Tensor | None = None
    cond_frame_indexes: tuple[int, ...] | None = None
    fps: float = 24.0
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    seed: int = 42


__all__ = ["ACTION_MODES", "Cosmos3ActionRequest"]
