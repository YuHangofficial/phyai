"""Requests for independently executing the Cosmos3 video/action stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Cosmos3VLMRequest:
    """Replace the active prompt's K/V; optionally move UND weights to CPU.

    This is the UND text tower. Visual observations enter through VAE encoding.
    The engine retains one prompt at a time, and returns its real token count.
    """

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    offload_after: bool = False


@dataclass
class Cosmos3EncodeRequest:
    """Encode clips [B,3,T,H,W] in [-1,1] to normalized VAE latents.

    Clips may have different spatial/temporal sizes. Encoding uses the posterior
    mean. Only the encoder and quantization convolution are loaded onto the GPU.
    """

    pixels: tuple[torch.Tensor, ...]
    offload_after: bool = False


@dataclass
class Cosmos3DenoiseRequest:
    """Denoise visual streams and optional actions with the active prompt.

    Each visual stream is [B,C,T,H,W], with a [B,T] mask (True means noisy).
    GEN positions [3,B,S] follow the concatenated visual patches, then actions.
    Supplying positions explicitly supports goal/history grids and their clocks.
    Actions are [B,L,D]; the first state row can stay clean through its row mask.
    Invalid action dimensions are zeroed. All clean values are restored per step.
    Sampling uses native UniPC without CFG; every step evaluates the full DiT.
    """

    vision: tuple[torch.Tensor, ...]
    video_noisy_masks: tuple[torch.Tensor, ...]
    gen_position_ids: torch.Tensor
    action: torch.Tensor | None = None
    action_noisy_mask: torch.Tensor | None = None
    action_valid_mask: torch.Tensor | None = None
    domain_id: int = 0
    video_role_embeddings: tuple[torch.Tensor, ...] | None = None
    num_inference_steps: int = 10
    seed: int = 42
    offload_after: bool = False


@dataclass
class Cosmos3DecodeRequest:
    """Decode normalized [B,C,T,H,W] latents to [B,3,T,H,W] pixels in [0,1].

    Only the decoder and post-quantization convolution are staged onto the GPU.
    """

    latents: torch.Tensor
    offload_after: bool = False


Cosmos3StageRequest = (
    Cosmos3VLMRequest
    | Cosmos3EncodeRequest
    | Cosmos3DenoiseRequest
    | Cosmos3DecodeRequest
)


__all__ = [
    "Cosmos3VLMRequest",
    "Cosmos3EncodeRequest",
    "Cosmos3DenoiseRequest",
    "Cosmos3DecodeRequest",
    "Cosmos3StageRequest",
]
