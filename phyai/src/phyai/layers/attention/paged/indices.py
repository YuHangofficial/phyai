"""Central index builders for the paged attention stack.

Schedulers used to hand-write the KV-slot arithmetic per model — pi0 and
pi05 each carried a private copy of the padded-write builder, and pi0's
three-block visibility structure (prefix | state | actions) lived as two
bespoke arange/searchsorted functions. The structure they all encode is
one shape:

    per sample b, the visible slots are
        [ prefix slots: contiguous ``real_lens[b]`` rows packed from
          ``prefix_slot_base`` ]
      ++ [ suffix slots: the first ``suffix_visible`` rows of the sample's
           fixed-stride suffix block at ``suffix_slot_base`` ]

Block-causal masks are *decomposed* into per-pass visibility sets rather
than materialized (each pass is a dense rectangle — no masked-out FLOPs),
so a model declares its passes by what each one may see:

* pi05 expert joint pass: ``suffix_visible = chunk`` (prefix + own chunk);
* pi0 state pass: ``suffix_visible = 1`` (prefix + the state token);
* pi0 action pass: ``suffix_visible = 1 + chunk`` (everything).

Every builder is sync-free (pass ``n_full`` when the host already knows
the total so the one ``int(cu[-1])`` read is skipped) and directly
consumable by :class:`~phyai.layers.attention.paged.PagedAttnMetadata`.
"""

from __future__ import annotations

import torch


def padded_write_indices(
    real_lens: torch.Tensor,
    *,
    n_per_sample: int,
    slot_base: int,
    sentinel_slot: int = 0,
) -> torch.Tensor:
    """KV-pool slot index per padded token, ``(B * n_per_sample,)`` int64.

    Real token ``b * n_per_sample + j`` (with ``j < real_lens[b]``) writes
    to ``slot_base + cu_real[b] + j``; padding rows write to
    ``sentinel_slot`` (typically 0). Directly consumable by
    :meth:`KVCachePool.write_kv`.

    The same ``j < real_lens[b]`` mask handles inter-sample padding too:
    setting ``real_lens[b] = 0`` for unused samples routes every one of
    their rows to the sentinel slot.
    """
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_real = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_real[1:] = torch.cumsum(real64, 0)

    j = torch.arange(n_per_sample, dtype=torch.int64, device=device).unsqueeze(0)
    real_at_b = real64.unsqueeze(1)
    cu_at_b = cu_real[:-1].unsqueeze(1)
    is_real = j < real_at_b
    real_slot = slot_base + cu_at_b + j
    write = torch.where(
        is_real, real_slot, torch.full_like(real_slot, int(sentinel_slot))
    )
    return write.flatten().to(torch.int64)


def suffix_pos_ids(
    real_lens: torch.Tensor,
    count: int,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """RoPE positions for suffix tokens, ``(B * count,)`` int32.

    Sample ``b``'s suffix rows sit at positions
    ``real_lens[b] + offset + [0 .. count)`` so the joint attention K
    layout (cached prefix + fresh suffix) sees one coherent position
    axis per sample. ``offset`` skips suffix rows that precede this
    pass's queries (pi0's action pass sits one past the state token).
    Padded samples (``real_lens[b] == 0``) get positions from ``offset``
    — fine because they self-attend only over their own suffix.
    """
    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1) + int(offset)
    j = torch.arange(count, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def visibility_indices(
    real_lens: torch.Tensor,
    *,
    prefix_slot_base: int,
    suffix_slot_base: int,
    suffix_stride: int,
    suffix_visible: int,
    n_full: int | None = None,
) -> torch.Tensor:
    """Per-sample interleaved slot list for one attention pass, int32.

    Output layout per sample:
    ``[prefix_b0 (real_lens[0] slots), suffix_b0 (suffix_visible slots),
       prefix_b1 (real_lens[1] slots), suffix_b1 (...), ...]``
    concatenated end-to-end — ``N_full = sum(real_lens) +
    B * suffix_visible`` rows. Prefix slots are the packed rows at
    ``prefix_slot_base + cu_real[b]``; suffix slots are the first
    ``suffix_visible`` rows of the sample's block at
    ``suffix_slot_base + b * suffix_stride``.

    Padded samples (``real_lens[b] == 0``) contribute only suffix slots —
    those rows self-attend within their own suffix, with no real-prefix
    participation.

    ``n_full`` may be supplied when the caller already knows it on the
    host, to skip the blocking ``int(cu_full[-1])`` device-to-host read.
    """
    if suffix_visible < 0 or suffix_stride < suffix_visible:
        raise ValueError(
            f"suffix_visible={suffix_visible} must be within the suffix "
            f"stride ({suffix_stride})."
        )
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)

    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)

    full_lens = real64 + suffix_visible
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(full_lens, 0)
    if n_full is None:
        n_full = int(cu_full[-1])

    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg

    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    suffix_slot = suffix_slot_base + seg_id * suffix_stride + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


__all__ = ["padded_write_indices", "suffix_pos_ids", "visibility_indices"]
