"""Numerical-equivalence tests for the Triton AdaRMSNorm kernel.

Validates :func:`phyai_kernel.adarmsnorm` against:

* an eager torch reference that mirrors lerobot ``PiGemmaRMSNorm`` semantics
  (``forward(x, cond) -> (out, gate)`` with ``out = normed * (1 + scale)
  + shift`` and ``gate = chunk(modulation, 3, dim=-1)[2]``),
* the same reference wrapped with ``torch.compile(mode="reduce-overhead")``
  — both because that's a realistic alternative the user would otherwise
  reach for, and because ``torch.compile``'s fp32-reduction fusions can
  diverge slightly from naive eager and the kernel must match either
  within tolerance.

Test grid covers:

* hidden sizes from 256 (Gemma head_dim) up through 8192 (single-block
  boundary) and 12288 (forces the two-pass kernel),
* per-batch broadcast (``x=(B, S, D)``, ``cond=(B, cond_dim)``) and
  per-token broadcast (``x=(B*S, D)``, ``cond=(B*S, cond_dim)``),
* fp16 / bf16 / fp32.
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel
import phyai_kernel.triton.ada_rms_norm as triton_adarms_mod


# --------------------------------------------------------------------------- #
# Reference (mirrors lerobot ``PiGemmaRMSNorm.forward`` exactly)              #
# --------------------------------------------------------------------------- #


def _ref_adarmsnorm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager torch reference. Modulation already broadcast-shaped vs ``x``."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = xf * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


def _broadcast_modulation(x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
    """Mirror lerobot's ``unsqueeze(1)`` for 3-D ``x`` x 2-D ``modulation``."""
    if x.dim() == 3 and modulation.dim() == 2:
        return modulation.unsqueeze(1)
    return modulation


# Pre-compiled torch reference, lazily initialised so tests that don't need
# it skip the compile cost.
_compiled_ref: dict[str, callable] = {}


def _ref_adarmsnorm_compiled(x: torch.Tensor, modulation: torch.Tensor, eps: float):
    """Same math, but routed through ``torch.compile``."""
    key = "default"
    fn = _compiled_ref.get(key)
    if fn is None:
        fn = torch.compile(_ref_adarmsnorm, mode="reduce-overhead", dynamic=True)
        _compiled_ref[key] = fn
    return fn(x, modulation, eps)


# --------------------------------------------------------------------------- #
# Shapes                                                                      #
# --------------------------------------------------------------------------- #


# Gemma head_dim, the pi0.5 action-expert width, the single-block boundary
# and a width that forces the two-pass kernel.
_HIDDEN_SIZES = [256, 1024, 8192, 12288]
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _tols(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return (5e-5, 5e-5)
    if dtype == torch.bfloat16:
        return (2e-2, 2e-2)
    return (1e-3, 1e-3)


def _make_inputs(
    *,
    leading: tuple[int, ...],
    cond_leading: tuple[int, ...],
    hidden: int,
    cond_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(x, modulation)``.

    ``modulation`` is allocated with shape ``cond_leading + (3 * hidden,)``;
    the test caller picks ``cond_leading`` to exercise the broadcast pattern
    (``cond_leading = leading[:1]`` for per-batch, or ``cond_leading =
    leading`` for per-token).
    """
    torch.manual_seed(0xC0DE * (hidden + sum(leading)) + cond_dim)
    x = torch.randn(*leading, hidden, device="cuda", dtype=dtype) * 0.5
    # Modulation in the same dtype as x — most realistic since
    # ``self.dense`` runs in the activation dtype during inference.
    modulation = (
        torch.randn(*cond_leading, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    )
    return x, modulation


# --------------------------------------------------------------------------- #
# Per-token mapping (cond_leading == x_leading)                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_adarmsnorm_per_token_matches_reference_in_2d_and_3d(
    hidden: int, dtype: torch.dtype
):
    """``x=(N, D)`` with ``modulation=(N, 3D)`` and ``x=(B, S, D)`` with
    ``modulation=(B, S, 3D)``: a 1:1 mapping in both layouts."""
    rtol, atol = _tols(dtype)
    for leading in ((17,), (2, 7)):
        x, mod = _make_inputs(
            leading=leading,
            cond_leading=leading,
            hidden=hidden,
            cond_dim=hidden,
            dtype=dtype,
        )
        expected_out, expected_gate = _ref_adarmsnorm(x, mod, 1e-6)
        actual_out, actual_gate = phyai_kernel.adarmsnorm(x, mod, 1e-6)
        torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Per-batch broadcast (``cond_leading=(B,)`` -> broadcasts across S)            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_adarmsnorm_3d_broadcast_over_seq(hidden: int, dtype: torch.dtype):
    """``x=(B, S, D)`` x ``modulation=(B, 3D)``: pi0.5 action-expert pattern.

    The kernel infers ``group_size = S`` and broadcasts each modulation row
    across the sequence axis. Gate output shape mirrors the broadcast-shaped
    modulation, ``(B, 1, D)``, so ``residual + out * gate`` lifts correctly.
    """
    B, S = 3, 13
    x = torch.randn(B, S, hidden, device="cuda", dtype=dtype) * 0.5
    modulation = torch.randn(B, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    # Reference: unsqueeze for 3-D x.
    expected_out, expected_gate = _ref_adarmsnorm(x, modulation.unsqueeze(1), eps=1e-6)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(
        x, modulation.unsqueeze(1), eps=1e-6
    )
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)
    assert actual_gate.shape == (B, 1, hidden)


# --------------------------------------------------------------------------- #
# Match torch.compile reference                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden", [1024])  # torch.compile is slow to warm up
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adarmsnorm_matches_torch_compile(hidden: int, dtype: torch.dtype):
    """The Triton kernel matches a ``torch.compile``'d reference at the same tols."""
    x = torch.randn(2, 7, hidden, device="cuda", dtype=dtype) * 0.5
    modulation = torch.randn(2, 1, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    eps = 1e-6
    # Compile reference, then run twice — second call uses the cached graph.
    expected_out, expected_gate = _ref_adarmsnorm_compiled(x, modulation, eps)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(x, modulation, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_adarmsnorm_edge_cases_explicit_out_zero_rows_and_block_boundary():
    hidden = 1024
    x = torch.randn(4, 6, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    modulation = (
        torch.randn(4, 1, 3 * hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    out = torch.empty_like(x)
    gate = torch.empty(4, 1, hidden, device="cuda", dtype=torch.bfloat16)
    ret_out, ret_gate = phyai_kernel.adarmsnorm(
        x, modulation, eps=1e-6, out=out, gate_out=gate
    )
    assert (
        ret_out.data_ptr() == out.data_ptr() and ret_gate.data_ptr() == gate.data_ptr()
    )
    expected_out, expected_gate = _ref_adarmsnorm(x, modulation, 1e-6)
    torch.testing.assert_close(ret_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ret_gate, expected_gate, rtol=2e-2, atol=2e-2)

    empty_out, empty_gate = phyai_kernel.adarmsnorm(
        torch.empty(0, hidden, device="cuda", dtype=torch.float16),
        torch.empty(0, 3 * hidden, device="cuda", dtype=torch.float16),
        eps=1e-6,
    )
    assert empty_out.shape == (0, hidden) and empty_gate.shape == (0, hidden)

    # The two-pass kernel matches the single-block one at the threshold.
    threshold = triton_adarms_mod._SINGLE_BLOCK_MAX
    for n_cols in (threshold, threshold + 256):
        x = torch.randn(4, n_cols, device="cuda", dtype=torch.float16) * 0.5
        modulation = (
            torch.randn(4, 3 * n_cols, device="cuda", dtype=torch.float16) * 0.1
        )
        expected_out, expected_gate = _ref_adarmsnorm(x, modulation, 1e-6)
        actual_out, actual_gate = phyai_kernel.adarmsnorm(x, modulation, 1e-6)
        torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "x, mod, kwargs, message",
    [
        (torch.randn(2, 64), torch.randn(2, 192), {}, "must live on CUDA"),
        (
            torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 100, device="cuda", dtype=torch.bfloat16),
            {},
            "modulation last dim",
        ),
        # N_total must be a multiple of N_mod.
        (
            torch.randn(7, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 192, device="cuda", dtype=torch.bfloat16),
            {},
            "non-zero multiple",
        ),
        (
            torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 192, device="cuda", dtype=torch.bfloat16),
            {"out": torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)},
            "`out` must match",
        ),
        (
            torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 192, device="cuda", dtype=torch.bfloat16),
            {"gate_out": torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)},
            "`gate_out` must have shape",
        ),
    ],
)
def test_adarmsnorm_validates_its_arguments(x, mod, kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        phyai_kernel.adarmsnorm(x, mod, **kwargs)


# --------------------------------------------------------------------------- #
# Module-level smoke (phyai.layers.AdaRMSNorm wraps the kernel)                #
# --------------------------------------------------------------------------- #


def test_phyai_layers_adarmsnorm_module_matches_reference_on_both_backends():
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    hidden, cond_dim = 1024, 1024

    def build(backend: str) -> AdaRMSNorm:
        return (
            AdaRMSNorm(
                hidden_size=hidden,
                cond_dim=cond_dim,
                eps=1e-6,
                backend=backend,
                prefix="m.l0.in",
            )
            .cuda()
            .to(torch.bfloat16)
        )

    kernel = build("phyai-kernel")
    torch_backend = build("torch")
    # Override zero-init so the test exercises a non-trivial modulation, and
    # sync weights so the two backends see identical inputs to the math.
    with torch.no_grad():
        kernel.dense.weight.normal_(0.0, 0.05)
        kernel.dense.bias.normal_(0.0, 0.05)
        torch_backend.dense.weight.copy_(kernel.dense.weight)
        torch_backend.dense.bias.copy_(kernel.dense.bias)

    x = torch.randn(3, 11, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    cond = torch.randn(3, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    actual_out, actual_gate = kernel(x, cond)
    # ReplicatedLinear.forward returns (y, optional_bias).
    modulation = kernel.dense(cond)[0].unsqueeze(1)
    expected_out, expected_gate = _ref_adarmsnorm(
        x, modulation, kernel.variance_epsilon
    )
    torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=2e-2, atol=2e-2)
    assert actual_gate.shape == (3, 1, hidden)
    out_t, gate_t = torch_backend(x, cond)
    torch.testing.assert_close(actual_out, out_t, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_gate, gate_t, rtol=2e-2, atol=2e-2)


def test_phyai_layers_adarmsnorm_weight_loader_keys():
    """AdaRMSNorm exposes its inner ``dense.weight`` / ``dense.bias`` to
    the generic safetensors loader via ``param.hf_keys`` (the new weight
    loading API replacing the old ``placements()`` method)."""
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    layer = AdaRMSNorm(
        hidden_size=64, cond_dim=64, prefix="m.l0.input_layernorm", backend="torch"
    )
    keys: set[tuple[str, str]] = set()
    for name, param in layer.named_parameters():
        for hf_key, _shard_id in getattr(param, "hf_keys", []):
            keys.add((hf_key, name))
    assert keys == {
        ("m.l0.input_layernorm.dense.weight", "dense.weight"),
        ("m.l0.input_layernorm.dense.bias", "dense.bias"),
    }
    with pytest.raises(ValueError, match="flashinfer"):
        AdaRMSNorm(hidden_size=64, cond_dim=64, backend="flashinfer", prefix="x")


# --------------------------------------------------------------------------- #
# Precomputed-modulation path (forward(x, modulation=...) == forward(x, cond))  #
# --------------------------------------------------------------------------- #


def _make_adarms(hidden: int, cond_dim: int) -> "object":
    from phyai.layers import AdaRMSNorm

    layer = (
        AdaRMSNorm(
            hidden_size=hidden,
            cond_dim=cond_dim,
            eps=1e-6,
            backend="phyai-kernel",
            prefix="model.layers.0.input_layernorm",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        layer.dense.weight.normal_(0.0, 0.05)
        layer.dense.bias.normal_(0.0, 0.05)
    return layer


def test_phyai_layers_adarmsnorm_precomputed_modulation_matches_the_dense_path():
    """``forward(x, modulation=row)`` (one precomputed row broadcast over all
    tokens) equals ``forward(x, cond)`` projecting that row per token, in 2-D
    and 3-D, and the ``(1, D)`` gate broadcasts through ``torch.addcmul`` like a
    full gate would. Near-, not bit-exact: the projection runs at a different
    row count.
    """
    pytest.importorskip("phyai.layers")
    hidden = cond_dim = 1024
    chunk, n_steps = 50, 10
    layer = _make_adarms(hidden, cond_dim)
    conds = torch.randn(n_steps, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    mod = layer.project_modulation(conds)
    assert tuple(mod.shape) == (n_steps, 3 * hidden)

    for i in (0, n_steps - 1):
        x = torch.randn(chunk, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
        out_idx, gate_idx = layer(x, modulation=mod[i : i + 1])
        out_ref, gate_ref = layer(x, conds[i : i + 1].expand(chunk, -1))
        torch.testing.assert_close(out_idx, out_ref, rtol=2e-2, atol=2e-2)
        assert gate_idx.shape == (1, hidden)
        torch.testing.assert_close(
            gate_idx.expand(chunk, -1).contiguous(), gate_ref, rtol=2e-2, atol=2e-2
        )
        residual = torch.randn(chunk, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
        torch.testing.assert_close(
            torch.addcmul(residual, out_idx, gate_idx),
            torch.addcmul(residual, out_idx, gate_idx.expand(chunk, -1)),
            rtol=0,
            atol=0,
        )

    x3 = torch.randn(2, 7, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    out_idx, gate_idx = layer(x3, modulation=mod[1:2])
    out_ref, _ = layer(x3, conds[1].reshape(1, 1, cond_dim).expand(2, 7, -1))
    torch.testing.assert_close(out_idx, out_ref, rtol=2e-2, atol=2e-2)
    assert out_idx.shape == (2, 7, hidden) and gate_idx.shape == (1, 1, hidden)
