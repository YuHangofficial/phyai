"""Runner-input forward batches for the pi0.5 vision/LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VisionForwardBatch:
    """Payload for one vision-tower call.

    A vision runner is captured once at fixed shape ``(N, C, H, W)`` —
    ``N`` images of ``C`` channels at the configured spatial size — and
    replayed once per request batch. The scheduler builds one
    :class:`VisionForwardBatch` per replay and concatenates the
    runner's outputs along the token axis afterwards.

    Fields
    ------
    pixel_values:
        Float tensor of shape ``(N, C, H, W)``: ``N`` images per call,
        ``C`` channels, spatial dims ``H == W == image_size``. dtype is
        the engine's ``params_dtype`` so the captured graph never
        inserts a runtime cast.
    """

    pixel_values: torch.Tensor


@dataclass
class LLMForwardBatch:
    """Per-call inputs for one prefix forward pass through the LLM backbone.

    This payload carries **only** the tensors that vary per forward
    call. Everything else the prefix forward needs — the
    :class:`KVCachePool`, the planned flashinfer attention wrapper,
    and the attention-metadata buffers (``cu_seqlens_q`` /
    ``paged_kv_*``) — is owned by the LLM runner and refreshed once
    per inference via its ``plan_inference()`` hook. The captured
    graph reads those runner-owned buffers directly, so passing them
    on the batch would be ignored.

    Per-sample padding layout (graph-friendly fixed shapes)
    -------------------------------------------------------
    Every sample is padded to ``n_per_sample`` tokens
    (``= image_token_count + tokenizer_max_length``). Real tokens land
    in the leading prefix of each sample's slot range; padding rows
    fill the tail and write their K/V to the sentinel slot at index 0
    of the cache pool (so the writes are harmless — the slot is never
    in any read-side ``paged_kv_indices``).

    Fields
    ------
    hidden_states:
        ``(B * n_per_sample, hidden_size)`` packed prefix embeddings,
        sample-major. Padding rows hold whatever the embed step left
        there (typically zeros).
    position_ids:
        ``(B * n_per_sample,)`` int32 — RoPE positions per token.
        Padding rows can carry any in-range value (0 is the simple
        default); attention never reads their key, so the RoPE phase
        on padded q rows is unobserved.
    write_indices:
        ``(B * n_per_sample,)`` int64 — per-token slot index where the
        layer's K/V is written. Real tokens point at their allocated
        slot; padding tokens point at the pool's sentinel slot
        (typically 0).
    """

    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    write_indices: torch.Tensor


__all__ = ["LLMForwardBatch", "VisionForwardBatch"]
