"""TransformerBlock: construction contract, family forward smoke, HF-key mapping.

Llama / Qwen / Gemma / Mistral / Phi3 / Olmo all use HF-default norm names
(``input_layernorm`` / ``post_attention_layernorm`` /
``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``), so the
block's defaults cover them with no ``norm_hf_names=`` argument. The override
dict is keyed by these HF default names, not by phyai-internal slot
identifiers; :data:`SIGLIP_NORM_OVERRIDES` is the override example.
"""

from __future__ import annotations

import pytest
import torch

from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    ParallelConfig,
)
from phyai.layers import RotaryEmbedding
from phyai.layers.transformer_block import TransformerBlock
from phyai.parallel.layout import build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import register_mesh


# SigLIP / CLIP: pre-norm with custom HF source names. Keys are the HF
# defaults for the slot; values are the names in the SigLIP checkpoint.
SIGLIP_NORM_OVERRIDES = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
}


def _block(**overrides) -> TransformerBlock:
    kwargs = dict(hidden_size=64, num_heads=4, intermediate_size=128)
    kwargs.update(overrides)
    return TransformerBlock(**kwargs)


def _hf_keys(blk: TransformerBlock) -> set[str]:
    """Collect every HF source key declared by any parameter in ``blk``."""
    return {
        hf_key
        for _, p in blk.named_parameters()
        for hf_key, _shard_id in getattr(p, "hf_keys", ())
    }


# ---------------------------------------------------------------------------
# Construction-time contract
# ---------------------------------------------------------------------------


def test_construction_rejects_bad_shapes_norm_types_and_norm_keys(fake_mesh):
    fake_mesh()
    with pytest.raises(ValueError, match="Unknown norm_type"):
        _block(norm_type="banana")
    with pytest.raises(ValueError, match="not divisible"):
        _block(hidden_size=65)
    # norm_hf_names is keyed by HF default names, never phyai slot names, and
    # sandwich-only keys are unknown to a pre-norm block.
    with pytest.raises(ValueError, match="unknown keys"):
        _block(norm_hf_names={"input_norm": "x"})
    with pytest.raises(ValueError, match="unknown keys"):
        _block(norm_hf_names={"pre_feedforward_layernorm": "anything"})


def test_different_domain_memberships_require_token_redistribution(fake_mesh):
    fake_mesh()
    register_mesh(
        Mesh(
            build_rank_layout(
                ParallelConfig(
                    dense=DenseParallelConfig(tp_size=4),
                    attention=AttentionParallelConfig(dp_size=2),
                )
            )
        )
    )
    with pytest.raises(NotImplementedError, match="token redistribution"):
        _block()


@pytest.mark.parametrize("config", ("cp_size", "decode_cp_size"))
def test_context_parallel_requires_a_model_attention_implementation(fake_mesh, config):
    fake_mesh()
    register_mesh(
        Mesh(
            build_rank_layout(
                ParallelConfig(attention=AttentionParallelConfig(**{config: 2}))
            )
        )
    )
    with pytest.raises(NotImplementedError, match="context-parallel attention"):
        _block()


def test_norm_topology_and_head_dim_attributes(fake_mesh):
    fake_mesh()
    pre = _block(head_dim=32)
    assert pre.head_dim == 32  # explicit head_dim wins over hidden // heads
    assert pre.q_heads_local == 4
    assert isinstance(pre.post_attn_norm, torch.nn.Identity)  # pre-norm: 2 norms
    assert isinstance(pre.post_ff_norm, torch.nn.Identity)
    assert isinstance(pre.q_norm, torch.nn.Identity)  # qk norm off by default
    assert isinstance(pre.k_norm, torch.nn.Identity)

    sandwich = _block(sandwich_norm=True, attn_qk_norm=True, head_dim=16)
    for norm in (
        sandwich.input_norm,
        sandwich.post_attn_norm,
        sandwich.pre_ff_norm,
        sandwich.post_ff_norm,
    ):
        assert not isinstance(norm, torch.nn.Identity)
    assert sandwich.q_norm.weight.shape == (16,)  # Q/K norm acts on head_dim
    assert sandwich.k_norm.weight.shape == (16,)
    s = repr(sandwich)
    assert "hidden_size=64" in s and "sandwich_norm=True" in s
    assert "attn_qk_norm=True" in s


def test_attention_kind_accepts_layer_idx_metadata(fake_mesh):
    """No-cache attention may carry layer_idx (stack-position metadata, ignored).

    The paged kinds still *require* it for KV-pool addressing, but attention
    treats it as optional inert metadata (surfaced in repr).
    """
    fake_mesh()
    blk = _block(layer_idx=3)
    assert blk.layer_idx == 3
    assert "layer_idx=3" in repr(blk)
    assert _block().layer_idx is None
    with pytest.raises(ValueError, match="requires layer_idx"):
        _block(attn_kind="paged")


def test_forward_validates_positions_rank_and_hidden_size(fake_mesh):
    fake_mesh()
    rope = RotaryEmbedding(16, max_position_embeddings=64, backend="eager")
    blk = _block(
        head_dim=16, rope=rope, attn_backend="eager", norm_backend="phyai-kernel"
    )
    with pytest.raises(ValueError, match="positions"):
        blk(torch.randn(2, 8, 64))
    with pytest.raises(ValueError, match="2-D .* or 3-D"):
        blk(torch.randn(2, 4, 8, 64), positions=torch.arange(8))
    with pytest.raises(ValueError, match="hidden_size"):
        blk(torch.randn(2, 8, 32), positions=torch.arange(8))


# ---------------------------------------------------------------------------
# Forward smoke, one row per model family
# ---------------------------------------------------------------------------

_FAMILIES = {
    # Gemma1 / Llama: pre-norm + RMSNorm + RoPE + gated SiLU + GQA + causal.
    "gemma1": dict(mlp_activation="silu", norm_type="rmsnorm", attn_backend="sdpa"),
    # Gemma2: sandwich + GemmaRMSNorm + soft_cap + sliding_window + GeGLU.
    "gemma2": dict(
        sandwich_norm=True,
        attn_sliding_window=8,
        attn_logits_soft_cap=50.0,
        mlp_activation="gelu_tanh",
        norm_type="gemma_rmsnorm",
        attn_backend="eager",
    ),
    # Gemma3: sandwich + GemmaRMSNorm + sliding_window + Q/K norm, no soft_cap.
    "gemma3": dict(
        sandwich_norm=True,
        attn_sliding_window=8,
        attn_qk_norm=True,
        mlp_activation="gelu_tanh",
        norm_type="gemma_rmsnorm",
        attn_backend="eager",
    ),
    # Qwen2 / Qwen2.5: Q/K/V bias but no O bias.
    "qwen2": dict(
        attn_bias=True,
        attn_out_bias=False,
        mlp_activation="silu",
        norm_type="rmsnorm",
        attn_backend="sdpa",
    ),
    # Qwen3: Q/K head_dim norm, no QKV bias.
    "qwen3": dict(
        attn_qk_norm=True,
        mlp_activation="silu",
        norm_type="rmsnorm",
        attn_backend="sdpa",
    ),
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_rope_family_forward_smoke(fake_mesh, family):
    fake_mesh()
    H, head_dim, tokens = 64, 16, 16
    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = _block(
        num_kv_heads=2,
        head_dim=head_dim,
        attn_causal=True,
        rope=rope,
        mlp_gated=True,
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
        **_FAMILIES[family],
    ).cuda()
    rope.cuda()
    assert (blk.qkv_proj.bias is not None) == bool(_FAMILIES[family].get("attn_bias"))
    assert blk.o_proj.bias is None
    assert isinstance(blk.q_norm, torch.nn.Identity) != bool(
        _FAMILIES[family].get("attn_qk_norm")
    )

    x = (torch.randn(2, tokens, H) * 0.05).to(torch.bfloat16).cuda()
    y = blk(x, positions=torch.arange(tokens, device="cuda"))
    assert y.shape == (2, tokens, H)
    assert y.dtype == torch.bfloat16


def _siglip_kwargs(prefix: str = "") -> dict:
    return dict(
        hidden_size=96,
        num_heads=4,
        head_dim=24,
        intermediate_size=256,
        attn_causal=False,
        attn_bias=True,
        rope=None,
        mlp_gated=False,
        mlp_activation="gelu_tanh",
        mlp_bias=True,
        norm_type="layernorm",
        norm_eps=1e-6,
        norm_bias=True,
        norm_backend="phyai-kernel",
        norm_hf_names=SIGLIP_NORM_OVERRIDES,
        attn_out_hf_name="out_proj",
        prefix=prefix,
    )


def test_siglip_style_forward(fake_mesh):
    """SigLIP encoder: pre-norm + LayerNorm(bias) + plain GELU-tanh MLP + non-causal."""
    fake_mesh()
    blk = TransformerBlock(
        **_siglip_kwargs(), attn_backend="sdpa", params_dtype=torch.bfloat16
    ).cuda()
    x = (torch.randn(2, 32, 96) * 0.05).to(torch.bfloat16).cuda()
    assert blk(x).shape == (2, 32, 96)


def test_precompute_rope_pattern_b_matches_pattern_a(fake_mesh):
    """Pattern B (threaded ``cos`` / ``sin``) == Pattern A (per-layer rope).

    Covers the block-level ``cos`` / ``sin`` forward kwargs and the
    ``precompute_rope=True`` branch against the default ``positions`` path,
    using one shared rope instance and identical weights.
    """
    fake_mesh()
    H, head_dim = 64, 16
    rope = RotaryEmbedding(
        head_dim, max_position_embeddings=128, backend="eager"
    ).cuda()

    def build(precompute: bool) -> TransformerBlock:
        return _block(
            num_kv_heads=2,
            head_dim=head_dim,
            attn_causal=True,
            rope=rope,
            precompute_rope=precompute,
            mlp_gated=True,
            mlp_activation="silu",
            norm_type="rmsnorm",
            attn_backend="sdpa",
            norm_backend="phyai-kernel",
            params_dtype=torch.bfloat16,
        ).cuda()

    blk_a = build(False)  # Pattern A: forward(positions=...)
    blk_b = build(True)  # Pattern B: forward(cos=..., sin=...)
    # phyai layers leave weights uninitialized (load_pretrained fills them in
    # production); give blk_a finite values and copy them into blk_b.
    torch.manual_seed(0)
    for p in blk_a.parameters():
        torch.nn.init.normal_(p, std=0.02)
    blk_b.load_state_dict(blk_a.state_dict())

    x = (torch.randn(2, 16, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(16, device="cuda")
    cos, sin = rope.get_cos_sin(pos)
    torch.testing.assert_close(blk_b(x, cos=cos, sin=sin), blk_a(x, positions=pos))


def test_ragged_forward(fake_mesh):
    """2-D ragged input runs through the block and preserves shape."""
    fake_mesh()
    H, head_dim = 64, 16
    rope = RotaryEmbedding(head_dim, max_position_embeddings=64, backend="eager")
    blk = _block(
        head_dim=head_dim,
        rope=rope,
        attn_backend="eager",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    nnz = 24
    x = (torch.randn(nnz, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.cat([torch.arange(12), torch.arange(12)]).to("cuda")
    cu = torch.tensor([0, 12, 24], dtype=torch.int32, device="cuda")
    assert blk(x, positions=pos, cu_seqlens_q=cu).shape == (nnz, H)


# ---------------------------------------------------------------------------
# Param-attached HF-key mapping, exact keys per family
# ---------------------------------------------------------------------------


def test_pre_norm_hf_keys_llama_like(fake_mesh):
    """Llama / Gemma1 / Qwen2 / Mistral convention."""
    fake_mesh()
    blk = _block(
        head_dim=16,
        sandwich_norm=False,
        attn_bias=False,
        mlp_bias=False,
        mlp_gated=True,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.0",
    )
    assert _hf_keys(blk) == {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }


def test_qwen2_hf_keys_with_qkv_bias(fake_mesh):
    """Qwen2: Q/K/V bias appear as separate keys, O has no bias."""
    fake_mesh()
    blk = _block(
        head_dim=16,
        attn_bias=True,
        attn_out_bias=False,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.7",
    )
    keys = _hf_keys(blk)
    assert {
        "model.layers.7.self_attn.q_proj.bias",
        "model.layers.7.self_attn.k_proj.bias",
        "model.layers.7.self_attn.v_proj.bias",
    } <= keys
    assert "model.layers.7.self_attn.o_proj.bias" not in keys


def test_gemma3_hf_keys_sandwich_qk_norm(fake_mesh):
    """Gemma3: four sandwich norms plus q_norm / k_norm (Gemma2 is the same minus q/k)."""
    fake_mesh()
    blk = _block(
        head_dim=16,
        sandwich_norm=True,
        attn_qk_norm=True,
        norm_type="gemma_rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.5",
    )
    keys = _hf_keys(blk)
    assert {
        "model.layers.5.input_layernorm.weight",
        "model.layers.5.post_attention_layernorm.weight",
        "model.layers.5.pre_feedforward_layernorm.weight",
        "model.layers.5.post_feedforward_layernorm.weight",
        "model.layers.5.self_attn.q_norm.weight",
        "model.layers.5.self_attn.k_norm.weight",
    } <= keys


def test_siglip_hf_keys(fake_mesh):
    """SigLIP: layer_norm{1,2} + out_proj + fc1/fc2 + bias on q/k/v/o/fc/norm."""
    fake_mesh()
    blk = TransformerBlock(**_siglip_kwargs(prefix="vision_model.encoder.layers.0"))
    assert _hf_keys(blk) == {
        "vision_model.encoder.layers.0.layer_norm1.weight",
        "vision_model.encoder.layers.0.layer_norm1.bias",
        "vision_model.encoder.layers.0.layer_norm2.weight",
        "vision_model.encoder.layers.0.layer_norm2.bias",
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.bias",
        "vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "vision_model.encoder.layers.0.self_attn.k_proj.bias",
        "vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "vision_model.encoder.layers.0.self_attn.v_proj.bias",
        "vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "vision_model.encoder.layers.0.self_attn.out_proj.bias",
        "vision_model.encoder.layers.0.mlp.fc1.weight",
        "vision_model.encoder.layers.0.mlp.fc1.bias",
        "vision_model.encoder.layers.0.mlp.fc2.weight",
        "vision_model.encoder.layers.0.mlp.fc2.bias",
    }
