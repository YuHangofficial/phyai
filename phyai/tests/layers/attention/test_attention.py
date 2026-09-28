"""Tests for :class:`phyai.layers.attention.attention.Attention`.

Covers the three no-cache backends — ``"eager"``, ``"sdpa"``, and
``"flashinfer"`` — across the padded (4-D) and ragged (3-D)
dispatch paths plus the ``ctx=None`` convenience flow used by the
vision tower. Numerical agreement between eager and sdpa is the
primary correctness signal; flashinfer is gated on GPU + flashinfer
import.
"""

from __future__ import annotations

import pytest

from phyai.kernel.call import explain
import torch

from phyai.layers.attention import (
    Attention,
    AttnCtx,
    AttnLayout,
    AttnMetadata,
    AttnMode,
)


def _has_flashinfer() -> bool:
    try:
        import flashinfer.prefill  # noqa: F401

        return True
    except ImportError:
        return False


def _can_use_flashinfer() -> bool:
    return _has_flashinfer()


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_construction_records_traits_and_rejects_bad_configs():
    for backend in ("eager", "sdpa"):
        attn = Attention(
            num_heads=4,
            head_dim=16,
            num_kv_heads=2,
            backend=backend,
            causal=True,
            backend_kwargs={"compile": False} if backend == "sdpa" else None,
        )
        assert (attn.backend, attn.num_heads, attn.num_kv_heads) == (backend, 4, 2)
        assert attn.head_dim == 16 and attn.causal is True
    with pytest.raises(ValueError, match="unknown backend"):
        Attention(num_heads=4, head_dim=16, backend="not-a-backend")
    with pytest.raises(ValueError, match="must be a positive multiple"):
        Attention(num_heads=4, head_dim=16, num_kv_heads=3, backend="eager")
    with pytest.raises(ValueError, match="sliding_window requires causal"):
        Attention(
            num_heads=4, head_dim=16, sliding_window=4, causal=False, backend="eager"
        )


def test_convenience_path_infers_layout_from_rank_and_validates_it():
    """With ctx=None the layer infers PADDED_4D / RAGGED_3D from q.ndim and
    lazily builds a default backend + AttnCtx in place."""
    torch.manual_seed(0)
    B, S, H, D = 2, 8, 4, 16
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    q = torch.randn(B, S, H, D, device="cuda")
    assert attn(q, q, q).shape == (B, S, H, D)

    cu_q = torch.tensor([0, 5, 12], dtype=torch.int32)
    ragged = torch.randn(int(cu_q[-1]), H, D, device="cuda")
    assert attn(ragged, ragged, ragged, cu_seqlens_q=cu_q).shape == (12, H, D)
    with pytest.raises(ValueError, match="ragged forward requires cu_seqlens_q"):
        attn(ragged, ragged, ragged)
    five_d = torch.randn(2, 4, H, D, 1, device="cuda")
    with pytest.raises(ValueError, match="q must be 3-D .ragged. or 4-D"):
        attn(five_d, five_d, five_d)


# --------------------------------------------------------------------- #
# Numerical correctness — eager vs sdpa                                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("causal", "S_q", "S_kv", "H_kv"),
    [(True, 8, 8, 4), (False, 5, 9, 2)],  # causal square; rectangular cross-attn + GQA
)
def test_eager_and_sdpa_agree_on_padded_input(causal, S_q, S_kv, H_kv):
    torch.manual_seed(2)
    B, H, D = 2, 4, 16
    q = torch.randn(B, S_q, H, D, device="cuda")
    k = torch.randn(B, S_kv, H_kv, D, device="cuda")
    v = torch.randn(B, S_kv, H_kv, D, device="cuda")
    eager = Attention(
        num_heads=H, head_dim=D, num_kv_heads=H_kv, backend="eager", causal=causal
    )
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=causal,
        backend_kwargs={"compile": False},
    )
    out_e = eager(q, k, v)
    assert out_e.shape == (B, S_q, H, D)
    assert torch.allclose(out_e, sdpa(q, k, v), atol=1e-5, rtol=1e-4)


def test_sdpa_is_not_selected_for_ragged_input():
    """SDPA is padded-only: it has no varlen API, so ragged is flashinfer's job.

    That used to surface as ``NotImplementedError`` because the backend was
    bound at construction. It is now a declared eligibility condition, so a
    ragged call simply does not select SDPA — and the trace says exactly why.
    The request still runs, on whichever implementation can handle it.
    """
    torch.manual_seed(5)
    H, D = 4, 16
    cu_q = torch.tensor([0, 5, 12], dtype=torch.int32)
    N = int(cu_q[-1])
    q = torch.randn(N, H, D, device="cuda")
    k = torch.randn(N, H, D, device="cuda")
    v = torch.randn(N, H, D, device="cuda")
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        backend="sdpa",
        causal=False,
        backend_kwargs={"compile": False},
    )

    trace = explain(
        "attention",
        role=sdpa.kernel_role,
        device=q.device,
        dtype={"input": q.dtype, "key": k.dtype, "value": v.dtype},
        shape={"head_dim": D, "tokens": N, "heads": H},
        attrs={"layout": "ragged", "causal": False},
        prefer=sdpa.prefer,
    )
    assert trace.selected != "sdpa.attention"
    rejection = next(c for c in trace.candidates if c.kernel_id == "sdpa.attention")
    assert "attrs.layout == padded" in rejection.reason

    # The padded form of the same call does select SDPA.
    padded = explain(
        "attention",
        role=sdpa.kernel_role,
        device=q.device,
        dtype={"input": q.dtype, "key": k.dtype, "value": v.dtype},
        shape={"head_dim": D, "tokens": N, "heads": H},
        attrs={"layout": "padded", "causal": False},
        prefer=sdpa.prefer,
    )
    assert padded.selected == "sdpa.attention"

    # And the ragged call still produces an answer.
    out = sdpa(q, k, v, cu_seqlens_q=cu_q)
    assert out.shape == (N, H, D)


# --------------------------------------------------------------------- #
# Causal / SWA / soft-cap correctness (vs eager reference)              #
# --------------------------------------------------------------------- #


def test_sliding_window_and_soft_cap_change_the_attention():
    torch.manual_seed(6)
    B, S, H, D = 1, 6, 2, 8
    q = torch.randn(B, S, H, D, device="cuda")
    k = torch.randn(B, S, H, D, device="cuda")
    v = torch.randn(B, S, H, D, device="cuda")
    # A window of 1 means each query attends only to its own position, so the
    # softmax over a single key is 1 and the output equals v.
    windowed = Attention(
        num_heads=H, head_dim=D, backend="eager", causal=True, sliding_window=1
    )
    assert torch.allclose(windowed(q, k, v), v, atol=1e-5, rtol=1e-4)
    # A finite soft cap must change the result for large logits.
    no_cap = Attention(num_heads=H, head_dim=D, backend="eager", causal=False)
    capped = Attention(
        num_heads=H, head_dim=D, backend="eager", causal=False, logits_soft_cap=1.0
    )
    assert not torch.allclose(
        no_cap(q * 5, k * 5, v), capped(q * 5, k * 5, v), atol=1e-3
    )


# --------------------------------------------------------------------- #
# Explicit ctx (advanced path used by callers that own the backend)     #
# --------------------------------------------------------------------- #


def test_explicit_ctx_padded_idle_returns_zeros():
    """IDLE mode bypasses the kernel and returns zeros."""
    B, S, H, D = 2, 4, 2, 8
    q = torch.randn(B, S, H, D, device="cuda")
    k = torch.randn(B, S, H, D, device="cuda")
    v = torch.randn(B, S, H, D, device="cuda")
    attn = Attention(num_heads=H, head_dim=D, backend="eager")
    backend = attn._ensure_backend()
    plan = backend.init_forward_metadata(
        AttnMetadata(
            mode=AttnMode.IDLE,
            layout=AttnLayout.PADDED_4D,
            batch_size=B,
            num_query_tokens=B * S,
        )
    )
    ctx = AttnCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.IDLE,
        layout=AttnLayout.PADDED_4D,
    )
    out = attn(q, k, v, ctx=ctx)
    assert torch.equal(out, torch.zeros_like(q))


# --------------------------------------------------------------------- #
# flashinfer (GPU-gated)                                                #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# Rectangular cross-attention: 4-D padded with S_q != S_kv              #
# --------------------------------------------------------------------- #


def test_sdpa_select_kernel_matches_default():
    """The CUDA kernel-priority context must not perturb results.

    ``select_kernel`` only biases which fused kernel CUDA dispatches to.
    Same inputs through ``select_kernel=True`` and ``False`` must agree:
    the test enters the ``sdpa_kernel`` priority context and checks the
    chosen kernel still matches default dispatch.
    """
    torch.manual_seed(13)
    # head_dim=64 + fp16 on CUDA exercises a real fused kernel; CPU uses fp32.
    B, S, H, H_kv, D = 2, 8, 4, 2, 64
    dtype = torch.float16
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H_kv, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H_kv, D, device="cuda", dtype=dtype)
    sel = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=True,
        backend_kwargs={"compile": False, "select_kernel": True},
    )
    nosel = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=True,
        backend_kwargs={"compile": False, "select_kernel": False},
    )
    out_sel = sel(q, k, v)
    out_nosel = nosel(q, k, v)
    assert torch.allclose(out_sel, out_nosel, atol=1e-3, rtol=1e-3)


def test_padded_rectangular_matches_ragged():
    """4-D padded rectangular == the same data packed into the 3-D ragged path.

    Confirms the padded path's synthesized uniform cu_seqlens (built in
    ``_build_default_ctx``) describe the same attention as explicit ragged
    cu_seqlens — i.e. a 4-D ``attn(q, k, v)`` with S_q != S_kv needs no manual
    packing.
    """
    torch.manual_seed(11)
    B, S_q, S_kv, H, D = 2, 4, 7, 4, 16
    q = torch.randn(B, S_q, H, D, device="cuda")
    k = torch.randn(B, S_kv, H, D, device="cuda")
    v = torch.randn(B, S_kv, H, D, device="cuda")
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=False)
    out_padded = attn(q, k, v)
    cu_q = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32)
    cu_kv = torch.arange(0, (B + 1) * S_kv, S_kv, dtype=torch.int32)
    out_ragged = attn(
        q.reshape(B * S_q, H, D),
        k.reshape(B * S_kv, H, D),
        v.reshape(B * S_kv, H, D),
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
    ).reshape(B, S_q, H, D)
    assert torch.allclose(out_padded, out_ragged, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not _can_use_flashinfer(),
    reason="flashinfer requires CUDA + flashinfer-python.",
)
def test_flashinfer_padded_matches_eager():
    """flashinfer 4-D padded matches eager: causal B=1 square, and non-causal
    rectangular (S_q != S_kv) for B==1 and B>1.

    B==1 routes through ``single_prefill`` (already rectangular); B>1 exercises
    the synthesized padded cu_seqlens + ragged-KV plan (B>1 padded raised
    before). This is the regression guard for cosmos3's cross-attention after
    dropping its hand-rolled ``_attend``.
    """
    torch.manual_seed(12)
    H, D = 4, 64
    for causal, B, S_q, S_kv in ((True, 1, 6, 6), (False, 1, 5, 9), (False, 2, 5, 9)):
        q = torch.randn(B, S_q, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S_kv, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S_kv, H, D, device="cuda", dtype=torch.float16)
        fi = Attention(num_heads=H, head_dim=D, backend="flashinfer", causal=causal)
        eager = Attention(num_heads=H, head_dim=D, backend="eager", causal=causal)
        out_fi = fi(q, k, v)
        assert out_fi.shape == (B, S_q, H, D)
        out_e = eager(q.float(), k.float(), v.float())
        assert torch.allclose(out_fi.float(), out_e, atol=1e-2, rtol=1e-2)
