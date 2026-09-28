"""VocabParallelEmbedding + ParallelLMHead at ws=1.

Collectives short-circuit when the group size is 1, so construction, weight
allocation, masked lookup and forward run without a real process group. The
vocab shard-bound math is exercised across ranks through the layer at tp=4
and, at the loader level, in ``tests/weights/test_shards.py``. Multi-rank
correctness lives under the gloo / NCCL harnesses.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.vocab_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_to,
)


def test_pad_vocab_to_rounds_up_to_a_tp_multiple():
    assert pad_vocab_to(32000, tp_size=2, multiple=64) == 32000  # already aligned
    assert pad_vocab_to(151700, tp_size=4, multiple=64) == 151808
    assert pad_vocab_to(100, tp_size=2, multiple=128) == 256  # FP8-style alignment
    assert pad_vocab_to(257, tp_size=2, multiple=128) == 512
    with pytest.raises(ValueError):
        pad_vocab_to(100, tp_size=0, multiple=64)


# --------------------------------------------------------------------------- #
# VocabParallelEmbedding                                                      #
# --------------------------------------------------------------------------- #


def test_embedding_tp1_construct_attrs(fake_mesh):
    fake_mesh(tp_size=1)
    layer = VocabParallelEmbedding(
        num_embeddings=100,
        embedding_dim=16,
        params_dtype=torch.float32,
        prefix="embed_tokens",
    )
    assert layer.num_embeddings == 100
    assert layer.num_embeddings_padded == 128  # 100 -> 128 (multiple of 64)
    assert layer.num_embeddings_per_partition == 128
    assert (layer.shard_start, layer.shard_end) == (0, 100)  # end clamped to V
    assert layer.weight.shape == (128, 16)
    assert layer.weight.dtype == torch.float32
    assert layer.weight.hf_keys == [("embed_tokens.weight", None)]
    assert callable(layer.weight.weight_loader)


def test_embedding_tp4_shard_bounds_per_rank(fake_mesh):
    """Across ranks, shard_start/end tile [0, V_padded) and clamp to V."""
    V, D, tp = 151700, 32, 4
    expected_padded = pad_vocab_to(V, tp, multiple=64)
    per_rank = expected_padded // tp
    boundaries = []
    for rank in range(tp):
        fake_mesh(tp_size=tp, rank=rank)
        layer = VocabParallelEmbedding(
            num_embeddings=V, embedding_dim=D, params_dtype=torch.float32
        )
        assert layer.num_embeddings_padded == expected_padded
        assert layer.weight.shape == (per_rank, D)
        boundaries.append((layer.shard_start, layer.shard_end))

    assert boundaries[:3] == [(i * per_rank, (i + 1) * per_rank) for i in range(3)]
    # Rank 3 starts at 3 * per_rank but ends at V: everything past V is padding.
    assert boundaries[3] == (3 * per_rank, V)


def test_embedding_rejects_unsupported_layouts_and_sizes(fake_mesh):
    fake_mesh(tp_size=1)
    with pytest.raises(NotImplementedError, match="vocab_parallel"):
        VocabParallelEmbedding(
            num_embeddings=100, embedding_dim=16, layout="hidden_parallel"
        )
    with pytest.raises(ValueError, match="num_embeddings must be positive"):
        VocabParallelEmbedding(num_embeddings=0, embedding_dim=16)
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        VocabParallelEmbedding(num_embeddings=100, embedding_dim=0)


def test_embedding_tp1_forward_matches_nn_embedding(fake_mesh):
    fake_mesh(tp_size=1)
    V, D = 64, 16
    layer = VocabParallelEmbedding(
        num_embeddings=V, embedding_dim=D, params_dtype=torch.float32
    )
    nn.init.normal_(layer.weight, std=0.05)
    # The loader zero-fills the padding rows (V..V_padded); mirror that here.
    layer.weight.data[V:].zero_()

    ids = torch.randint(0, V, (4, 8), dtype=torch.int64, device="cuda")
    torch.testing.assert_close(
        layer(ids), F.embedding(ids, layer.weight[:V]), atol=0, rtol=0
    )
    # Any leading shape is preserved; an empty batch is legal.
    ids3 = torch.randint(0, V, (2, 3, 4), dtype=torch.int64, device="cuda")
    assert layer(ids3).shape == (2, 3, 4, D)
    assert layer(torch.empty((0,), dtype=torch.int64, device="cuda")).shape == (0, D)


# --------------------------------------------------------------------------- #
# ParallelLMHead                                                              #
# --------------------------------------------------------------------------- #


def test_lmhead_tp1_construct_attrs(fake_mesh):
    fake_mesh(tp_size=1)
    head = ParallelLMHead(
        embedding_dim=16,
        num_embeddings=100,
        params_dtype=torch.float32,
        prefix="lm_head",
    )
    assert head.num_embeddings == 100
    assert head.num_embeddings_padded == 128
    assert head.num_embeddings_per_partition == 128
    assert head.weight.shape == (128, 16)
    assert head.input_size_per_partition == 16
    assert head.output_size_per_partition == 128
    assert head.bias is None
    assert head.weight.hf_keys == [("lm_head.weight", None)]


def test_lmhead_rejects_bias_and_mismatched_tied_weight(fake_mesh):
    fake_mesh(tp_size=1)
    with pytest.raises(NotImplementedError, match="bias"):
        ParallelLMHead(embedding_dim=16, num_embeddings=100, bias=True)
    bogus = nn.Parameter(torch.empty(64, 16), requires_grad=False)
    with pytest.raises(ValueError, match="shape"):
        ParallelLMHead(
            embedding_dim=16,
            num_embeddings=100,
            tied_weight=bogus,
            params_dtype=torch.float32,
        )


def test_lmhead_tp1_forward_matches_F_linear(fake_mesh):
    fake_mesh(tp_size=1)
    head = ParallelLMHead(
        embedding_dim=32, num_embeddings=100, params_dtype=torch.bfloat16
    )
    nn.init.normal_(head.weight, std=0.02)

    x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
    y = head(x)
    torch.testing.assert_close(y, F.linear(x, head.weight), atol=0, rtol=0)
    assert y.shape == (4, 128)  # per-rank V, which is the padded V at tp=1


def test_lmhead_tied_weight_is_the_embedding_parameter(fake_mesh):
    fake_mesh(tp_size=1)
    embed = VocabParallelEmbedding(
        num_embeddings=64, embedding_dim=16, params_dtype=torch.bfloat16
    )
    nn.init.normal_(embed.weight, std=0.02)
    head = ParallelLMHead(
        embedding_dim=16,
        num_embeddings=64,
        tied_weight=embed.weight,
        params_dtype=torch.bfloat16,
    )
    assert head.weight is embed.weight  # the same Parameter, not a copy
    assert head.logical_widths == [embed.num_embeddings_padded]

    # A write through the embedding is visible to the head's forward.
    embed.weight.data.fill_(0.5)
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(head(x), F.linear(x, embed.weight), atol=0, rtol=0)


def test_padding_logits_are_zero_after_load(fake_mesh):
    """The whole point of zero-fill padding: out-of-vocab logits stay 0."""
    fake_mesh(tp_size=1)
    V, D = 100, 8
    head = ParallelLMHead(
        embedding_dim=D, num_embeddings=V, params_dtype=torch.float32, prefix="lm_head"
    )
    disk_w = torch.randn(V, D, dtype=torch.float32)
    head.weight.weight_loader(head.weight, disk_w, None)

    assert torch.equal(head.weight.data[:V].cpu(), disk_w)
    assert torch.all(head.weight.data[V:] == 0)
    x = torch.randn(7, D, dtype=torch.float32, device="cuda")
    assert torch.all(head(x)[:, V:] == 0)


# --------------------------------------------------------------------------- #
# embed_scale (Gemma / PaliGemma scaled embeddings)                           #
# --------------------------------------------------------------------------- #


def test_embedding_scale_construction(fake_mesh):
    fake_mesh(tp_size=1)
    D = 16
    plain = VocabParallelEmbedding(num_embeddings=64, embedding_dim=D)
    assert plain.embed_scale == 1.0
    assert not hasattr(plain, "_embed_scale_t")  # default allocates no buffer
    assert "embed_scale" not in repr(plain)

    scaled = VocabParallelEmbedding(
        num_embeddings=64,
        embedding_dim=D,
        embed_scale=D**0.5,
        params_dtype=torch.float32,
    )
    assert scaled.embed_scale == D**0.5
    assert scaled._embed_scale_t.dtype == torch.float32
    assert f"embed_scale={D**0.5}" in repr(scaled)

    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="embed_scale"):
            VocabParallelEmbedding(num_embeddings=64, embedding_dim=D, embed_scale=bad)


def test_embedding_scale_applies_at_forward_time_only(fake_mesh):
    """Output equals the unscaled lookup times the scale, in the input dtype,
    and the stored weight is never touched (a tied head must see it unscaled)."""
    fake_mesh(tp_size=1)
    V, D = 64, 16
    scale = D**0.5
    plain = VocabParallelEmbedding(
        num_embeddings=V, embedding_dim=D, params_dtype=torch.bfloat16
    )
    scaled = VocabParallelEmbedding(
        num_embeddings=V,
        embedding_dim=D,
        embed_scale=scale,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(plain.weight, std=0.05)
    plain.weight.data[V:].zero_()
    scaled.weight.data.copy_(plain.weight.data)
    weight_before = scaled.weight.data.clone()

    ids = torch.randint(0, V, (4, 8), dtype=torch.int64, device="cuda")
    out = scaled(ids)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, plain(ids) * scale, atol=0, rtol=0)
    torch.testing.assert_close(scaled.weight.data, weight_before, atol=0, rtol=0)


def test_tied_lmhead_sees_unscaled_weight(fake_mesh):
    """HF Gemma semantics: embeddings are scaled by sqrt(D), tied logits are not."""
    fake_mesh(tp_size=1)
    V, D = 64, 16
    scale = D**0.5
    embed = VocabParallelEmbedding(
        num_embeddings=V, embedding_dim=D, embed_scale=scale, params_dtype=torch.float32
    )
    nn.init.normal_(embed.weight, std=0.02)
    embed.weight.data[V:].zero_()
    head = ParallelLMHead(
        embedding_dim=D,
        num_embeddings=V,
        tied_weight=embed.weight,
        params_dtype=torch.float32,
    )

    x = torch.randn(2, D, dtype=torch.float32, device="cuda")
    y = head(x)
    torch.testing.assert_close(y, F.linear(x, embed.weight), atol=0, rtol=0)
    assert not torch.allclose(y, F.linear(x, embed.weight * scale), atol=1e-3)
