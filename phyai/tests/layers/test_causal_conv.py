"""Layer-level tests for :class:`phyai.layers.CausalConv1d`.

Runs on CUDA; the torch reference math below is the semantic contract
whichever catalog row selection picks must match.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.conv import CausalConv1d


def make_layer(**overrides) -> CausalConv1d:
    kwargs = dict(
        channels=12,
        kernel_size=4,
        split_sizes=(4, 4, 4),
        device="cuda",
    )
    kwargs.update(overrides)
    layer = CausalConv1d(kwargs.pop("channels"), kwargs.pop("kernel_size"), **kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.weight.normal_(0.0, 0.2)
    return layer


def reference(x: torch.Tensor, weight: torch.Tensor, split_sizes) -> tuple:
    channels = x.shape[-1]
    kernel = weight.shape[-1]
    y = F.conv1d(x.transpose(1, 2), weight, padding=kernel - 1, groups=channels)[
        :, :, : x.shape[1]
    ]
    y = F.silu(y).transpose(1, 2)
    return torch.split(y, split_sizes, dim=-1)


def test_matches_the_reference_math() -> None:
    layer = make_layer()
    x = torch.randn(2, 6, 12, device="cuda")
    query, key, value = layer(x)
    ref_q, ref_k, ref_v = reference(x, layer.weight, (4, 4, 4))
    torch.testing.assert_close(query, ref_q)
    torch.testing.assert_close(key, ref_k)
    torch.testing.assert_close(value, ref_v)
    assert query.shape == (2, 6, 4)


def test_the_convolution_is_causal() -> None:
    """Perturbing a later position must not change earlier outputs."""

    layer = make_layer()
    x = torch.randn(1, 8, 12, device="cuda")
    poked = x.clone()
    poked[:, 5:] += 1.0
    base = torch.cat(layer(x), dim=-1)
    changed = torch.cat(layer(poked), dim=-1)
    torch.testing.assert_close(base[:, :5], changed[:, :5])
    assert not torch.allclose(base[:, 5:], changed[:, 5:])


def test_a_noncontiguous_input_is_handled() -> None:
    """The mixer feeds a ``split()`` view; the layer owns the copy."""

    layer = make_layer()
    wide = torch.randn(2, 6, 24, device="cuda")
    x = wide.split((12, 12), dim=-1)[0]
    assert not x.is_contiguous()
    out = torch.cat(layer(x), dim=-1)
    ref = torch.cat(layer(x.contiguous()), dim=-1)
    torch.testing.assert_close(out, ref)


def test_split_sizes_must_sum_to_channels() -> None:
    with pytest.raises(ValueError, match="must sum to"):
        CausalConv1d(12, 4, split_sizes=(4, 4, 8), device="cuda")


def test_prefix_attaches_checkpoint_plumbing() -> None:
    layer = CausalConv1d(
        8, 4, split_sizes=(4, 2, 2), device="cuda", prefix="model.layers.0.conv1d"
    )
    assert layer.weight.shape == (8, 1, 4)
    assert layer.weight.hf_keys == [("model.layers.0.conv1d.weight", None)]
