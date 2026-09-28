"""Declarative attention masks with backend-owned lowering.

VLA models carry three mask families that plain ``causal`` cannot express,
and every in-tree model used to invent its own answer (a hand-rolled
per-row Python loop, zeroed embeddings, refusing padded batches, or
hand-written scheduler index arithmetic):

* **lengths** — per-row valid KV length (right-padded batches);
* **keys** — per-row boolean KV set (scattered columns, e.g. "attend the
  visual tokens only", query-independent);
* **segments** — block-causal structure over the KV sequence (prefix-LM,
  pi0's prefix | state | action blocks), openpi's ``make_attn_mask``
  formalism at segment granularity.

:class:`AttnMask` declares the structure once; the lowering to a kernel's
native form is owned here and consumed by the backends:

* dense boolean mask (:meth:`AttnMask.dense`) — SDPA / eager, any family;
* per-row KV packing (:func:`pack_kv`) — FlashInfer varlen, lengths/keys.

Semantics (openpi ``make_attn_mask``): a query row may attend a key
position iff ``block_ok AND kv_valid`` —

    ``allowed[b, i, j] = (cumsum_ar[q_pos(i)] >= cumsum_ar[j]) & valid[b, j]``

where ``q_pos(i) = i + (S_kv - S_q)`` aligns queries with the *trailing*
keys (the subsystem-wide append-prefill convention), the per-token ``ar``
flag is each segment's flag on its first token and ``False`` inside, and
``valid`` comes from ``lengths`` / ``keys``. Tokens inside one segment
attend each other bidirectionally; a segment attends every earlier
segment. Within-segment causal attention is what the layer's plain
``causal=True`` path is for — a layer constructed causal cannot also take
``segments``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


def causal_block_mask(
    S_q: int,
    S_kv: int,
    device: torch.device,
    *,
    causal: bool,
    sliding_window: int | None,
) -> torch.Tensor | None:
    """Bool mask ``(S_q, S_kv)``: True = attend.

    Returns ``None`` when no masking is needed (full non-causal). The
    layer's ``__init__`` rejects ``sliding_window`` with ``causal=False``,
    so the only shapes here are causal / causal+SWA. Append-prefill
    alignment is ``q_pos[i] = i + (S_kv - S_q)``.
    """
    if not causal and sliding_window is None:
        return None
    i = torch.arange(S_q, device=device).unsqueeze(1)
    j = torch.arange(S_kv, device=device).unsqueeze(0)
    q_pos = i + (S_kv - S_q)
    mask = q_pos >= j
    if sliding_window is not None:
        mask = mask & (q_pos - j < sliding_window)
    return mask


@lru_cache(maxsize=64)
def _segments_block_mask(
    segments: tuple[tuple[int, bool], ...],
    S_q: int,
    S_kv: int,
    device_str: str,
) -> torch.Tensor:
    """Cached block mask for one segment layout on one device.

    Segment layouts are per-model constants and the mask is shape-pure,
    so one build serves every layer and every step (the sglang vision
    stack caches its patch masks the same way).
    """
    device = torch.device(device_str)
    ar = torch.zeros(S_kv, dtype=torch.int32, device=device)
    start = 0
    for length, flag in segments:
        if flag:
            ar[start] = 1
        start += length
    cumsum = torch.cumsum(ar, dim=0)
    q_pos = torch.arange(S_kv - S_q, S_kv, device=device)
    return cumsum[q_pos].unsqueeze(1) >= cumsum.unsqueeze(0)


@dataclass(frozen=True)
class AttnMask:
    """Declarative attention mask; see the module docstring for semantics.

    ``segments`` composes with either KV-validity family (padding inside
    a segmented sequence); ``seq_lens_kv`` and ``key_mask`` are mutually
    exclusive (``key_mask`` subsumes lengths).
    """

    segments: tuple[tuple[int, bool], ...] | None = None
    seq_lens_kv: torch.Tensor | None = None
    key_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.segments is None and self.seq_lens_kv is None and self.key_mask is None:
            raise ValueError(
                "AttnMask needs at least one of segments / seq_lens_kv / key_mask."
            )
        if self.seq_lens_kv is not None and self.key_mask is not None:
            raise ValueError(
                "AttnMask: seq_lens_kv and key_mask are mutually exclusive — "
                "key_mask already encodes per-row validity."
            )
        if self.segments is not None:
            normalized = tuple((int(n), bool(flag)) for n, flag in self.segments)
            if not normalized or any(n <= 0 for n, _ in normalized):
                raise ValueError(
                    f"AttnMask.segments must be non-empty (length, ar_flag) "
                    f"pairs with positive lengths; got {self.segments!r}."
                )
            object.__setattr__(self, "segments", normalized)
        if self.seq_lens_kv is not None and self.seq_lens_kv.ndim != 1:
            raise ValueError(
                f"AttnMask.seq_lens_kv must be 1-D (batch,), got "
                f"{tuple(self.seq_lens_kv.shape)}."
            )
        if self.key_mask is not None:
            if self.key_mask.ndim != 2:
                raise ValueError(
                    f"AttnMask.key_mask must be 2-D (batch, S_kv), got "
                    f"{tuple(self.key_mask.shape)}."
                )
            if self.key_mask.dtype != torch.bool:
                object.__setattr__(self, "key_mask", self.key_mask.bool())

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_lengths(cls, seq_lens_kv: torch.Tensor) -> "AttnMask":
        """Per-row valid KV length (right-padded batches)."""
        return cls(seq_lens_kv=seq_lens_kv)

    @classmethod
    def from_key_mask(cls, key_mask: torch.Tensor) -> "AttnMask":
        """Per-row boolean KV set — scattered columns allowed."""
        return cls(key_mask=key_mask)

    @classmethod
    def from_segments(
        cls,
        segments,
        *,
        seq_lens_kv: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> "AttnMask":
        """Block-causal structure, optionally combined with KV validity."""
        return cls(segments=tuple(segments), seq_lens_kv=seq_lens_kv, key_mask=key_mask)

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    @property
    def kind(self) -> str:
        """Selection fact: the hardest-to-lower part names the mask."""
        if self.segments is not None:
            return "segments"
        if self.key_mask is not None:
            return "keys"
        return "lengths"

    # ------------------------------------------------------------------ #
    # Lowerings                                                          #
    # ------------------------------------------------------------------ #

    def kv_valid(
        self, batch_size: int, S_kv: int, device: torch.device
    ) -> torch.Tensor | None:
        """Per-row KV validity ``(B, S_kv)`` bool, or ``None`` if unrestricted."""
        if self.key_mask is not None:
            if self.key_mask.shape != (batch_size, S_kv):
                raise ValueError(
                    f"AttnMask.key_mask shape {tuple(self.key_mask.shape)} does "
                    f"not match (batch, S_kv) = ({batch_size}, {S_kv})."
                )
            return self.key_mask.to(device)
        if self.seq_lens_kv is not None:
            if self.seq_lens_kv.numel() != batch_size:
                raise ValueError(
                    f"AttnMask.seq_lens_kv has {self.seq_lens_kv.numel()} rows, "
                    f"batch has {batch_size}."
                )
            lens = self.seq_lens_kv.to(device)
            return torch.arange(S_kv, device=device)[None, :] < lens[:, None]
        return None

    def dense(
        self,
        S_q: int,
        S_kv: int,
        batch_size: int,
        device: torch.device,
        *,
        causal: bool,
        sliding_window: int | None,
    ) -> torch.Tensor | None:
        """Combined bool mask, broadcastable against ``(B, H, S_q, S_kv)``.

        The block part comes from ``segments`` when present (which the
        layer only allows on ``causal=False`` layers), otherwise from the
        layer's ``causal`` / ``sliding_window`` flags; the KV-validity
        part is ANDed in per row. Returns ``(S_q, S_kv)`` when nothing is
        per-row, ``(B, 1, S_q, S_kv)`` otherwise.
        """
        if self.segments is not None:
            total = sum(n for n, _ in self.segments)
            if total != S_kv:
                raise ValueError(
                    f"AttnMask.segments cover {total} tokens but S_kv={S_kv}."
                )
            if S_q > S_kv:
                raise ValueError(
                    f"AttnMask.segments: S_q={S_q} exceeds the {S_kv} declared "
                    f"positions."
                )
            block = _segments_block_mask(self.segments, S_q, S_kv, str(device))
        else:
            block = causal_block_mask(
                S_q, S_kv, device, causal=causal, sliding_window=sliding_window
            )

        valid = self.kv_valid(batch_size, S_kv, device)
        if valid is None:
            return block
        per_row = valid[:, None, None, :]
        if block is None:
            return per_row
        return block[None, None, :, :] & per_row


def pack_rows(
    valid: torch.Tensor,
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Compact padded rows down to the valid ones, batched.

    ``valid`` is ``(B, S)`` bool; each tensor is ``(B, S, ...)``. Returns
    ``(cu_seqlens, index, packed_tensors)`` where each packed tensor is
    ``(N_kept, ...)`` and ``cu_seqlens`` is the int32 varlen offsets
    FlashInfer's ragged wrapper consumes. ``index`` scatters a packed
    result back into the padded layout
    (``padded.flatten(0, 1).index_copy_(0, index, packed)``). One gather
    replaces the per-row Python loop this pattern used to be implemented
    with.
    """
    for tensor in tensors:
        if valid.shape != tensor.shape[:2]:
            raise ValueError(
                f"pack_rows: valid shape {tuple(valid.shape)} does not match "
                f"tensor batch shape {tuple(tensor.shape[:2])}."
            )
    counts = valid.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(valid.shape[0] + 1, dtype=torch.int32, device=valid.device)
    torch.cumsum(counts, dim=0, out=cu_seqlens[1:])
    index = valid.flatten().nonzero(as_tuple=False).squeeze(1)
    packed = tuple(t.flatten(0, 1).index_select(0, index) for t in tensors)
    return cu_seqlens, index, packed


__all__ = ["AttnMask", "causal_block_mask", "pack_rows"]
