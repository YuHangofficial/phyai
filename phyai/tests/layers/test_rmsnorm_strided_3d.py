"""Strided 3-D RMSNorm parity, fallback, and CUDA graph replay."""

import torch
import pytest
from phyai.parallel.state import graph_capture
from phyai.layers.layer_norm import RMSNorm, GemmaRMSNorm


@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel", "torch"])
@pytest.mark.parametrize("tokens,heads", [(17, 8), (583, 32), (721, 8)])
def test_strided_3d_matches_contiguous_path(backend, tokens, heads):
    torch.manual_seed(932)
    norm = RMSNorm(128, backend=backend, device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(torch.randn_like(norm.weight))
    packed = torch.randn(tokens, 48, 128, device="cuda", dtype=torch.bfloat16)
    value = packed[:, :heads]
    original = packed.clone()
    expected = norm(value)
    actual = norm(value, preserve_3d_layout=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(packed, original, atol=0, rtol=0)


def test_strided_3d_cpu_fallback_and_invalid_contract():
    norm = RMSNorm(32, backend="flashinfer", device="cpu", dtype=torch.float32)
    value = torch.randn(5, 6, 32)[:, :3]
    torch.testing.assert_close(
        norm(value, preserve_3d_layout=True), norm(value), atol=0, rtol=0
    )
    for invalid in (value[0], torch.randn(5, 3, 64)[..., ::2]):
        with pytest.raises(ValueError, match="preserve_3d_layout"):
            norm(invalid, preserve_3d_layout=True)
    with pytest.raises(ValueError, match="no residual"):
        norm(value, value.clone(), preserve_3d_layout=True)


@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel", "torch"])
@pytest.mark.parametrize(
    "norm_cls,options",
    [
        (GemmaRMSNorm, {}),
        (RMSNorm, {"cast_before_affine": True}),
        (RMSNorm, {"elementwise_affine": False}),
    ],
)
def test_strided_3d_permuted_rows_preserve_norm_options(backend, norm_cls, options):
    torch.manual_seed(934)
    norm = norm_cls(
        128, backend=backend, device="cuda", dtype=torch.bfloat16, **options
    )
    if norm.elementwise_affine:
        norm.weight.copy_(torch.randn_like(norm.weight))
    value = torch.randn(3, 5, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    expected = norm(value)
    actual = norm(value, preserve_3d_layout=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_strided_3d_cuda_graph_observes_new_values():
    torch.manual_seed(933)
    norm = RMSNorm(128, backend="flashinfer", device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(17, 48, 128, device="cuda", dtype=torch.bfloat16)
    value = packed[:, 32:40]
    with graph_capture():
        for _ in range(3):
            norm(value, preserve_3d_layout=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with graph_capture(), torch.cuda.graph(graph):
        captured = norm(value, preserve_3d_layout=True)
    for _ in range(3):
        packed.copy_(torch.randn_like(packed))
        expected = norm(value)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    graph.reset()
