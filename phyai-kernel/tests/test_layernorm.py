"""Numerical-equivalence tests for the Triton LayerNorm kernel.

Validates :func:`phyai_kernel.triton.layernorm` against
:func:`torch.nn.functional.layer_norm` across the dtype x hidden_size
matrix relevant to SigLIP / BERT / ViT.
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel


# --------------------------------------------------------------------------- #
# Reference                                                                    #
# --------------------------------------------------------------------------- #


def _ref_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias, eps=eps
    )


# --------------------------------------------------------------------------- #
# Shape x dtype x bias matrix                                                  #
# --------------------------------------------------------------------------- #


# SigLIP-tiny, PaliGemma SigLIP, an awkward non-power-of-two width, the
# single-block boundary and a width that forces the two-pass kernel.
_HIDDEN_SIZES = [384, 1152, 3584, 8192, 12288]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_layernorm_matches_reference(hidden_size, dtype, with_bias):
    torch.manual_seed(0)
    n_rows = 17  # awkward to exercise masked tail
    x = (torch.randn(n_rows, hidden_size, device="cuda") * 0.5).to(dtype)
    # SigLIP stores weight/bias in the activation dtype; mirror that.
    weight = (torch.randn(hidden_size, device="cuda") * 0.1 + 1.0).to(dtype)
    bias = (
        (torch.randn(hidden_size, device="cuda") * 0.02).to(dtype)
        if with_bias
        else None
    )
    eps = 1e-5

    out = phyai_kernel.layernorm(x, weight, bias, eps)
    ref = _ref_layernorm(x, weight, bias, eps)

    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_layernorm_flattens_higher_rank_input_and_honours_out():
    torch.manual_seed(1)
    B, S, D = 2, 8, 1152
    x = (torch.randn(B, S, D, device="cuda") * 0.5).to(torch.bfloat16)
    weight = (torch.randn(D, device="cuda") * 0.1 + 1.0).to(torch.bfloat16)
    bias = (torch.randn(D, device="cuda") * 0.02).to(torch.bfloat16)
    out = phyai_kernel.layernorm(x, weight, bias, 1e-5)
    assert out.shape == (B, S, D)
    torch.testing.assert_close(
        out, _ref_layernorm(x, weight, bias, 1e-5), atol=2e-2, rtol=2e-2
    )

    flat = x.reshape(-1, D)
    buffer = torch.empty_like(flat)
    returned = phyai_kernel.layernorm(flat, weight, None, 1e-5, out=buffer)
    assert returned.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(
        returned, _ref_layernorm(flat, weight, None, 1e-5), atol=2e-2, rtol=2e-2
    )


def test_layernorm_validates_device_and_shapes():
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.layernorm(torch.randn(2, 64), torch.randn(64))
    x = torch.randn(2, 64, device="cuda").to(torch.bfloat16)
    w = torch.randn(64, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="weight"):
        phyai_kernel.layernorm(x, torch.randn(32, device="cuda").to(torch.bfloat16))
    with pytest.raises(RuntimeError, match="bias"):
        phyai_kernel.layernorm(x, w, torch.randn(32, device="cuda").to(torch.bfloat16))
    with pytest.raises(RuntimeError, match="`out` must match"):
        phyai_kernel.layernorm(
            x,
            w,
            None,
            1e-5,
            out=torch.empty(2, 32, device="cuda", dtype=torch.bfloat16),
        )
