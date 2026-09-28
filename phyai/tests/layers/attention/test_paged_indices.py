"""The central paged index builders against their frozen predecessors.

pi0 and pi05 schedulers used to hand-write this arithmetic (pi0/pi05 each
carried a copy of the padded-write builder; pi0's state/action visibility
were two bespoke functions). The legacy implementations below are frozen
verbatim as the regression reference — the central builders must be
bit-identical on randomized layouts, padding included.
"""

from __future__ import annotations

import pytest
import torch

from phyai.layers.attention.paged.indices import (
    padded_write_indices,
    suffix_pos_ids,
    visibility_indices,
)


# --------------------------------------------------------------------- #
# Frozen legacy implementations (verbatim from the pi0/pi05 schedulers) #
# --------------------------------------------------------------------- #


def legacy_prefix_padded_write_indices(
    real_lens, *, n_per_sample, prefix_slot_base, sentinel_slot=0
):
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_real = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_real[1:] = torch.cumsum(real64, 0)
    j = torch.arange(n_per_sample, dtype=torch.int64, device=device).unsqueeze(0)
    real_at_b = real64.unsqueeze(1)
    cu_at_b = cu_real[:-1].unsqueeze(1)
    is_real = j < real_at_b
    real_slot = prefix_slot_base + cu_at_b + j
    write = torch.where(
        is_real, real_slot, torch.full_like(real_slot, int(sentinel_slot))
    )
    return write.flatten().to(torch.int64)


def legacy_suffix_pos_ids(real_lens, chunk_size):
    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1)
    j = torch.arange(chunk_size, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def legacy_action_pos_ids(real_lens, chunk_size):
    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1) + 1
    j = torch.arange(chunk_size, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def legacy_joint_paged_kv_indices(
    real_lens, chunk_size, *, prefix_slot_base, suffix_slot_base, n_full=None
):
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    full_lens = real64 + chunk_size
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
    suffix_slot = suffix_slot_base + seg_id * chunk_size + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


def legacy_pi0_state_paged_kv_indices(
    real_lens, suffix_len, *, prefix_slot_base, suffix_slot_base
):
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    total_lens = real64 + 1
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(total_lens, 0)
    n_full = int(cu_full[-1])
    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg
    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    state_slot = suffix_slot_base + seg_id * suffix_len
    return torch.where(is_prefix, prefix_slot, state_slot).to(torch.int32)


def legacy_pi0_action_paged_kv_indices(
    real_lens, suffix_len, *, prefix_slot_base, suffix_slot_base
):
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    total_lens = real64 + suffix_len
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(total_lens, 0)
    n_full = int(cu_full[-1])
    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg
    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    suffix_slot = suffix_slot_base + seg_id * suffix_len + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


# --------------------------------------------------------------------- #
# Randomized equivalence                                                #
# --------------------------------------------------------------------- #


def random_lens(B: int, high: int, *, zero_rows: bool, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, high + 1, (B,), generator=g, dtype=torch.int32)
    if zero_rows:
        lens[torch.rand(B, generator=g) < 0.3] = 0
    return lens


CASES = [(3, 16), (32, 200)]  # a small and a large randomized layout


@pytest.mark.parametrize("B,high", CASES)
@pytest.mark.parametrize("zero_rows", [False, True])
def test_padded_write_indices_matches_legacy(B, high, zero_rows):
    lens = random_lens(B, high, zero_rows=zero_rows, seed=B * high)
    new = padded_write_indices(
        lens, n_per_sample=high + 4, slot_base=17, sentinel_slot=0
    )
    old = legacy_prefix_padded_write_indices(
        lens, n_per_sample=high + 4, prefix_slot_base=17, sentinel_slot=0
    )
    assert torch.equal(new, old)


@pytest.mark.parametrize("B,high", CASES)
def test_suffix_pos_ids_cover_all_three_legacy_builders(B, high):
    lens = random_lens(B, high, zero_rows=True, seed=B + high)
    chunk = 10
    assert torch.equal(suffix_pos_ids(lens, chunk), legacy_suffix_pos_ids(lens, chunk))
    assert torch.equal(
        suffix_pos_ids(lens, chunk, offset=1), legacy_action_pos_ids(lens, chunk)
    )
    # pi0's state positions: one row per sample at position real_len.
    assert torch.equal(suffix_pos_ids(lens, 1), lens.to(torch.int32))


@pytest.mark.parametrize("B,high", CASES)
@pytest.mark.parametrize("zero_rows", [False, True])
def test_visibility_matches_legacy_pi05_joint(B, high, zero_rows):
    lens = random_lens(B, high, zero_rows=zero_rows, seed=B * 3 + high)
    chunk = 12
    n_full = int(lens.sum()) + B * chunk
    new = visibility_indices(
        lens,
        prefix_slot_base=100,
        suffix_slot_base=9000,
        suffix_stride=chunk,
        suffix_visible=chunk,
        n_full=n_full,
    )
    old = legacy_joint_paged_kv_indices(
        lens, chunk, prefix_slot_base=100, suffix_slot_base=9000, n_full=n_full
    )
    assert torch.equal(new, old)


@pytest.mark.parametrize("B,high", CASES)
@pytest.mark.parametrize("zero_rows", [False, True])
def test_visibility_matches_legacy_pi0_state_and_action(B, high, zero_rows):
    lens = random_lens(B, high, zero_rows=zero_rows, seed=B * 5 + high)
    chunk = 10
    suffix_len = 1 + chunk  # state token + action chunk
    state_new = visibility_indices(
        lens,
        prefix_slot_base=100,
        suffix_slot_base=9000,
        suffix_stride=suffix_len,
        suffix_visible=1,
    )
    state_old = legacy_pi0_state_paged_kv_indices(
        lens, suffix_len, prefix_slot_base=100, suffix_slot_base=9000
    )
    assert torch.equal(state_new, state_old)

    action_new = visibility_indices(
        lens,
        prefix_slot_base=100,
        suffix_slot_base=9000,
        suffix_stride=suffix_len,
        suffix_visible=suffix_len,
    )
    action_old = legacy_pi0_action_paged_kv_indices(
        lens, suffix_len, prefix_slot_base=100, suffix_slot_base=9000
    )
    assert torch.equal(action_new, action_old)


def test_visibility_rejects_window_beyond_stride():
    with pytest.raises(ValueError, match="within the suffix"):
        visibility_indices(
            torch.tensor([3]),
            prefix_slot_base=0,
            suffix_slot_base=10,
            suffix_stride=4,
            suffix_visible=5,
        )
