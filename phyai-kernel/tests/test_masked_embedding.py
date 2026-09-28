"""Numerical-equivalence tests for the Triton masked_embedding_lookup kernel.

Reference is the obvious three-pass formulation:
``F.embedding(where(mask, ids - start, 0), W).masked_fill_(~mask, 0)``.

The kernel does no reductions and no fp32 promotion, so for fp32 weights we
expect bit-exact equality with the reference. For fp16/bf16 we still expect
bit-exact since the only operation is an indexed copy + zero-fill.

Test grid covers:
* shapes that span small (decode-like) and large (prefill-like) batches
* embedding dims at common LLM hidden sizes plus an awkward non-pow2
* dtypes: fp32, fp16, bf16
* shard configurations: full coverage, partial coverage, empty shard
* edge cases: empty input, all in-range, all out-of-range, mixed
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import phyai_kernel


# --------------------------------------------------------------------------- #
# Reference                                                                   #
# --------------------------------------------------------------------------- #


def _ref_masked_embedding_lookup(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    shard_start: int,
    shard_end: int,
) -> torch.Tensor:
    """Three-pass reference: masked gather then zero-out misses."""
    mask = (input_ids >= shard_start) & (input_ids < shard_end)
    local_ids = torch.where(mask, input_ids - shard_start, torch.zeros_like(input_ids))
    out = F.embedding(local_ids, weight)
    out = out.masked_fill(~mask.unsqueeze(-1), 0)
    return out


# --------------------------------------------------------------------------- #
# Test grid                                                                   #
# --------------------------------------------------------------------------- #

_INPUT_SHAPES = [
    (1,),  # decode, single token
    (8192,),  # long prefill
    (2, 17, 64),  # 3D batch (B, S, ...)
]
_DIMS = [127, 4096]  # non-pow2 and Llama / Qwen2 14B
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _make_inputs(
    shape: tuple[int, ...],
    embedding_dim: int,
    *,
    v_per_rank: int,
    dtype: torch.dtype,
    seed: int = 0xC0DE,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        v_per_rank, embedding_dim, device="cuda", dtype=dtype, generator=g
    )
    return weight


# --------------------------------------------------------------------------- #
# Tests — full and partial shard coverage                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", _INPUT_SHAPES)
@pytest.mark.parametrize("embedding_dim", _DIMS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_masked_embedding_full_shard_matches_reference(
    shape: tuple[int, ...], embedding_dim: int, dtype: torch.dtype
):
    """Single rank holding the entire vocab — every id is in-range."""
    V = 1024
    weight = _make_inputs(shape, embedding_dim, v_per_rank=V, dtype=dtype)
    ids = torch.randint(0, V, shape, device="cuda", dtype=torch.int64)

    expected = _ref_masked_embedding_lookup(ids, weight, 0, V)
    actual = phyai_kernel.masked_embedding_lookup(ids, weight, 0, V)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # No reductions, no fp32 promotion -> bit-exact for every dtype.
    assert torch.equal(
        actual, expected
    ), f"mismatch at shape={shape} D={embedding_dim} dtype={dtype}"


@pytest.mark.parametrize("shape", [(256,), (4, 128)])
@pytest.mark.parametrize("embedding_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_embedding_partial_shard_zeros_out_of_range(
    shape: tuple[int, ...], embedding_dim: int, dtype: torch.dtype
):
    """Rank holds [256, 512); ids span the full vocab -> about half are zeroed."""
    V_global = 1024
    shard_start, shard_end = 256, 512
    V_per_rank = shard_end - shard_start
    weight = _make_inputs(shape, embedding_dim, v_per_rank=V_per_rank, dtype=dtype)
    ids = torch.randint(0, V_global, shape, device="cuda", dtype=torch.int64)

    expected = _ref_masked_embedding_lookup(ids, weight, shard_start, shard_end)
    actual = phyai_kernel.masked_embedding_lookup(ids, weight, shard_start, shard_end)

    assert torch.equal(actual, expected)

    # Sanity: out-of-shard positions are exactly zero.
    out_of_shard = (ids < shard_start) | (ids >= shard_end)
    if out_of_shard.any():
        assert (actual[out_of_shard] == 0).all()


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_edge_cases_empty_all_in_all_out_zero_width_shard_and_int32_ids():
    weight = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    empty = phyai_kernel.masked_embedding_lookup(
        torch.empty((0,), device="cuda", dtype=torch.int64), weight, 0, 32
    )
    assert empty.shape == (0, 64) and empty.dtype == torch.float32
    # Every id in range reproduces the table; every id out of range is zero.
    assert torch.equal(
        phyai_kernel.masked_embedding_lookup(
            torch.arange(32, device="cuda"), weight, 0, 32
        ),
        weight,
    )
    outside = torch.tensor([0, 1, 2, 100, 200, 300], device="cuda", dtype=torch.int64)
    assert (phyai_kernel.masked_embedding_lookup(outside, weight, 10, 20) == 0).all()
    # An empty shard (start == end) means every position is out-of-range.
    none = torch.randn(0, 64, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 1000, (16,), device="cuda", dtype=torch.int64)
    assert (phyai_kernel.masked_embedding_lookup(ids, none, 100, 100) == 0).all()
    # int32 ids are accepted.
    ids32 = torch.randint(0, 32, (8,), device="cuda", dtype=torch.int32)
    assert torch.equal(
        phyai_kernel.masked_embedding_lookup(ids32, weight, 0, 32),
        _ref_masked_embedding_lookup(ids32.long(), weight, 0, 32),
    )


def test_output_follows_input_shape_and_non_contiguous_ids_are_handled():
    V = 64
    weight = torch.randn(V, 16, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, V, (2, 3, 4), device="cuda", dtype=torch.int64)
    out = phyai_kernel.masked_embedding_lookup(ids, weight, 0, V)
    assert out.shape == (2, 3, 4, 16)
    assert torch.equal(out, _ref_masked_embedding_lookup(ids, weight, 0, V))
    strided = torch.randint(0, V, (16, 4), device="cuda", dtype=torch.int64)[::2]
    assert not strided.is_contiguous()
    assert torch.equal(
        phyai_kernel.masked_embedding_lookup(strided, weight, 0, V),
        _ref_masked_embedding_lookup(strided, weight, 0, V),
    )


@pytest.mark.parametrize(
    "weight, ids, shard, message",
    [
        (torch.randn(32, 16), torch.randint(0, 32, (4,)), (0, 32), "must live on CUDA"),
        (
            torch.randn(16, device="cuda"),
            torch.randint(0, 16, (4,), device="cuda"),
            (0, 16),
            "weight must be 2D",
        ),
        (
            torch.randn(16, 8, device="cuda"),
            torch.zeros(4, device="cuda"),
            (0, 16),
            "input_ids dtype",
        ),
        (
            torch.randn(16, 8, device="cuda"),
            torch.zeros(4, device="cuda", dtype=torch.int64),
            (10, 5),
            "shard_end",
        ),
    ],
)
def test_invalid_arguments_are_rejected(weight, ids, shard, message):
    with pytest.raises(RuntimeError, match=message):
        phyai_kernel.masked_embedding_lookup(ids, weight, *shard)
