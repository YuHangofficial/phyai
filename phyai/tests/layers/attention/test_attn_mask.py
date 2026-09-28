"""The declarative attention mask: semantics, lowerings, routing.

The reference for ``segments`` is openpi's ``make_attn_mask``
(.tmp/openpi/src/openpi/models/pi0.py): ``mask[i, j] = cumsum_ar[j] <=
cumsum_ar[i] AND valid[i] AND valid[j]``. The dense lowering must agree
with it elementwise; the varlen packing must agree with the dense path
numerically through real backends.
"""

from __future__ import annotations

import pytest
import torch

from phyai.layers.attention import Attention, AttnMask
from phyai.layers.attention.mask import causal_block_mask, pack_rows


def openpi_reference(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Straight port of openpi's make_attn_mask (bool[B, N] each)."""
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    return attn & valid


def expand_segments(segments) -> torch.Tensor:
    """Per-token ar flags for a segment list: flag on first token only."""
    flags = []
    for length, ar in segments:
        flags.extend([ar] + [False] * (length - 1))
    return torch.tensor(flags, dtype=torch.bool)


# --------------------------------------------------------------------- #
# segments semantics == openpi                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "segments",
    [
        ((4, False), (1, True), (3, True)),  # pi0: prefix | state | actions
        ((5, False), (3, True)),  # pi05: prefix-LM
        ((2, True), (2, True), (2, True)),  # block-causal chain
    ],
)
def test_segments_match_openpi_reference(segments):
    total = sum(n for n, _ in segments)
    mask = AttnMask.from_segments(segments)
    dense = mask.dense(
        total, total, 1, torch.device("cpu"), causal=False, sliding_window=None
    )
    ar = expand_segments(segments).unsqueeze(0)
    ref = openpi_reference(torch.ones(1, total, dtype=torch.bool), ar)[0]
    assert torch.equal(dense, ref)


def test_segments_trailing_alignment_for_short_queries():
    """S_q < S_kv: queries are the trailing rows of the full mask."""
    segments = ((4, False), (1, True), (3, True))
    total, S_q = 8, 3
    mask = AttnMask.from_segments(segments)
    dense = mask.dense(
        S_q, total, 1, torch.device("cpu"), causal=False, sliding_window=None
    )
    ar = expand_segments(segments).unsqueeze(0)
    full = openpi_reference(torch.ones(1, total, dtype=torch.bool), ar)[0]
    assert torch.equal(dense, full[total - S_q :])


def test_segments_compose_with_lengths_like_openpi_input_mask():
    """segments + per-row validity == openpi(input_mask, mask_ar) columns."""
    segments = ((5, False), (3, True))
    total, B = 8, 2
    lens = torch.tensor([6, 8])
    mask = AttnMask.from_segments(segments, seq_lens_kv=lens)
    dense = mask.dense(
        total, total, B, torch.device("cpu"), causal=False, sliding_window=None
    )
    ar = expand_segments(segments).unsqueeze(0).expand(B, -1)
    valid = torch.arange(total)[None, :] < lens[:, None]
    ref = openpi_reference(valid, ar)
    # phyai masks KV validity per column only (invalid *query* rows produce
    # ignored outputs); compare on the valid query rows.
    for b in range(B):
        rows = valid[b]
        assert torch.equal(dense[b, 0][rows], ref[b][rows])


# --------------------------------------------------------------------- #
# lengths / keys lowering                                               #
# --------------------------------------------------------------------- #


def test_lengths_and_keys_lower_to_query_independent_columns():
    dense = AttnMask.from_lengths(torch.tensor([2, 4])).dense(
        3, 4, 2, torch.device("cpu"), causal=False, sliding_window=None
    )
    # Broadcastable against (B, H, S_q, S_kv): rows are query-independent.
    assert dense.shape == (2, 1, 1, 4)
    assert torch.equal(dense[0, 0, 0], torch.tensor([True, True, False, False]))
    assert dense[1].all()

    keys = torch.tensor([[True, False, True, False]])
    dense = AttnMask.from_key_mask(keys).dense(
        2, 4, 1, torch.device("cpu"), causal=False, sliding_window=None
    )
    assert dense.shape == (1, 1, 1, 4)
    assert torch.equal(dense[0, 0, 0], keys[0])


def test_pack_rows_gathers_and_offsets():
    valid = torch.tensor([[True, False, True], [True, True, False]])
    k = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
    cu, index, (packed,) = pack_rows(valid, k)
    assert torch.equal(cu, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert torch.equal(packed.flatten(), torch.tensor([0.0, 2.0, 3.0, 4.0]))
    out = torch.zeros(6, 1, 1)
    out.index_copy_(0, index, packed)
    assert torch.equal(out.flatten(), torch.tensor([0.0, 0, 2, 3, 4, 0]))


# --------------------------------------------------------------------- #
# constraints                                                           #
# --------------------------------------------------------------------- #


def test_mask_constraints_are_enforced():
    with pytest.raises(ValueError, match="at least one"):
        AttnMask()
    with pytest.raises(ValueError, match="mutually exclusive"):
        AttnMask(
            seq_lens_kv=torch.tensor([1]), key_mask=torch.ones(1, 2, dtype=torch.bool)
        )
    causal = Attention(num_heads=2, head_dim=8, causal=True, backend="eager")
    q = torch.randn(1, 4, 2, 8)
    with pytest.raises(ValueError, match="causal=False"):
        causal(q, q, q, mask=AttnMask.from_segments(((2, False), (2, True))))
    plain = Attention(num_heads=2, head_dim=8, causal=False, backend="eager")
    ragged = torch.randn(4, 2, 8)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="padded"):
        plain(
            ragged,
            ragged,
            ragged,
            cu_seqlens_q=cu,
            mask=AttnMask.from_lengths(torch.tensor([4])),
        )


def test_causal_block_mask_keeps_trailing_alignment():
    """The absorbed common.build_padded_mask behaviour."""
    mask = causal_block_mask(
        2, 4, torch.device("cpu"), causal=True, sliding_window=None
    )
    assert torch.equal(
        mask, torch.tensor([[True, True, True, False], [True, True, True, True]])
    )
    assert (
        causal_block_mask(3, 3, torch.device("cpu"), causal=False, sliding_window=None)
        is None
    )


# --------------------------------------------------------------------- #
# end-to-end through the layer (sdpa vs eager)                          #
# --------------------------------------------------------------------- #


def _layer(backend: str, *, causal: bool = False, head_dim: int = 16) -> Attention:
    return Attention(num_heads=4, head_dim=head_dim, causal=causal, backend=backend)


def _qkv(B=2, S=6, H=4, D=16, dtype=torch.float32):
    torch.manual_seed(0)
    return (
        torch.randn(B, S, H, D, dtype=dtype, device="cuda"),
        torch.randn(B, S, H, D, dtype=dtype, device="cuda"),
        torch.randn(B, S, H, D, dtype=dtype, device="cuda"),
    )


MASKS = {
    "lengths": lambda B, S: AttnMask.from_lengths(
        torch.tensor([S - 2] * B, device="cuda")
    ),
    "keys": lambda B, S: AttnMask.from_key_mask(
        torch.rand(B, S, device="cuda") > 0.4,
    ),
    "segments": lambda B, S: AttnMask.from_segments(((S - 2, False), (2, True))),
}


@pytest.mark.parametrize("family", sorted(MASKS))
def test_sdpa_and_eager_agree_on_every_family(family):
    q, k, v = _qkv()
    mask = MASKS[family](q.shape[0], k.shape[1])
    out_sdpa = _layer("sdpa")(q, k, v, mask=mask)
    out_eager = _layer("eager")(q, k, v, mask=mask)
    assert torch.allclose(out_sdpa, out_eager, atol=1e-5, rtol=1e-5)


def test_dense_lowering_matches_manual_masked_softmax_with_and_without_causal():
    q, k, v = _qkv(B=2)
    lens = torch.tensor([4, 6], device="cuda")
    # Non-causal: row attends the valid keys only.
    out = _layer("eager")(q, k, v, mask=AttnMask.from_lengths(lens))
    qh, kh, vh = (
        q[:1].transpose(1, 2),
        k[:1, :4].transpose(1, 2),
        v[:1, :4].transpose(1, 2),
    )
    ref = (torch.softmax((qh @ kh.transpose(-2, -1)) / 4.0, dim=-1) @ vh).transpose(
        1, 2
    )
    assert torch.allclose(out[:1], ref, atol=1e-5, rtol=1e-5)
    # Causal + lengths: row 0 of batch 0 attends key 0 only.
    out = _layer("eager", causal=True)(q, k, v, mask=AttnMask.from_lengths(lens))
    qh, kh = q[0:1, 0:1].transpose(1, 2), k[0:1, 0:1].transpose(1, 2)
    ref = torch.softmax((qh @ kh.transpose(-2, -1)) / 4.0, dim=-1) @ v[
        0:1, 0:1
    ].transpose(1, 2)
    assert torch.allclose(out[0:1, 0:1], ref.transpose(1, 2), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# flashinfer varlen lowering (CUDA)                                     #
# --------------------------------------------------------------------- #


def _has_flashinfer() -> bool:
    try:
        import flashinfer.prefill  # noqa: F401

        return True
    except ImportError:
        return False


flashinfer_only = pytest.mark.skipif(
    not _has_flashinfer(), reason="requires flashinfer-python"
)


def _assert_flashinfer_serves(*, causal: bool, mask_kind: str) -> None:
    """Keep the CUDA tests honest: head_dim 64 must land on flashinfer."""
    from phyai.kernel.call import explain

    trace = explain(
        "attention",
        role="attention",
        device=torch.device("cuda"),
        dtype={"input": torch.float16, "key": torch.float16, "value": torch.float16},
        shape={
            "tokens": 12,
            "kv_tokens": 12,
            "heads": 4,
            "kv_heads": 4,
            "head_dim": 64,
        },
        attrs={"layout": "padded", "causal": causal, "mask_kind": mask_kind},
    )
    assert trace.selected.startswith("flashinfer."), trace.selected


@flashinfer_only
@pytest.mark.parametrize("family", ["lengths", "keys"])
def test_flashinfer_pack_lowering_matches_eager(family):
    q, k, v = _qkv(D=64, dtype=torch.float16)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    mask = MASKS[family](q.shape[0], k.shape[1])
    _assert_flashinfer_serves(causal=False, mask_kind=family)
    out_fi = _layer("flashinfer", head_dim=64)(q, k, v, mask=mask)
    out_ref = _layer("eager", head_dim=64)(
        q.cpu().float(), k.cpu().float(), v.cpu().float(), mask=mask
    )
    # Compare only rows that attend at least one key (fully-masked rows are
    # unspecified for varlen kernels; the eager path zeros them).
    valid = mask.kv_valid(q.shape[0], k.shape[1], torch.device("cpu"))
    attended = valid.any(dim=1)
    assert torch.allclose(
        out_fi.cpu().float()[attended], out_ref[attended], atol=2e-2, rtol=2e-2
    )


@flashinfer_only
def test_flashinfer_causal_lengths_packs_both_sides():
    q, k, v = _qkv(D=64, dtype=torch.float16)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    lens = torch.tensor([4, 6], device="cuda")
    mask = AttnMask.from_lengths(lens)
    _assert_flashinfer_serves(causal=True, mask_kind="lengths")
    out_fi = _layer("flashinfer", causal=True, head_dim=64)(q, k, v, mask=mask)
    out_ref = _layer("eager", causal=True, head_dim=64)(
        q.cpu().float(), k.cpu().float(), v.cpu().float(), mask=mask
    )
    for b, n in enumerate(lens.tolist()):
        assert torch.allclose(
            out_fi[b, :n].cpu().float(), out_ref[b, :n], atol=2e-2, rtol=2e-2
        )
        # Pad rows scatter back as zeros.
        assert out_fi[b, n:].abs().sum().item() == 0


@flashinfer_only
def test_segments_route_away_from_flashinfer():
    """The catalog, not a runtime raise, keeps segments off the varlen path."""
    from phyai.kernel.call import explain

    trace = explain(
        "attention",
        role="attention",
        device=torch.device("cuda"),
        dtype={"input": torch.float16, "key": torch.float16, "value": torch.float16},
        shape={
            "tokens": 12,
            "kv_tokens": 12,
            "heads": 4,
            "kv_heads": 4,
            "head_dim": 64,
        },
        attrs={"layout": "padded", "causal": False, "mask_kind": "segments"},
    )
    flashinfer_rows = [
        c for c in trace.candidates if c.kernel_id.startswith("flashinfer.")
    ]
    assert flashinfer_rows and not any(c.eligible for c in flashinfer_rows)
    assert trace.selected == "sdpa.attention"

    # And the layer path produces sdpa-served output that matches eager.
    q, k, v = _qkv(D=64, dtype=torch.float16)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    mask = MASKS["segments"](q.shape[0], k.shape[1])
    out = _layer("flashinfer", head_dim=64)(q, k, v, mask=mask)  # capability wins
    ref = _layer("eager", head_dim=64)(
        q.cpu().float(), k.cpu().float(), v.cpu().float(), mask=mask
    )
    assert torch.allclose(out.cpu().float(), ref, atol=2e-2, rtol=2e-2)
