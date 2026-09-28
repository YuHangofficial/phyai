"""V-sharded input embedding and tied LM head.

Quick start::

    import phyai.parallel as P
    from phyai.layers.vocab_embedding import VocabParallelEmbedding, ParallelLMHead

    from phyai.engine_config import ParallelConfig, DenseParallelConfig

    P.init(ParallelConfig(dense=DenseParallelConfig(tp_size=8)))

    embed = VocabParallelEmbedding(num_embeddings=151936, embedding_dim=4096)
    lm_head = ParallelLMHead(
        embedding_dim=4096,
        num_embeddings=151936,
        tied_weight=embed.weight,   # share the parameter; no post-hoc mutation
    )

    h = embed(input_ids)             # (..., 4096)
    logits = lm_head(h)              # (..., V_padded // tp_size)

The fused masked-lookup Triton kernel registers itself on import via the
``phyai::masked_embedding_lookup`` custom op. There is no separate
dispatcher to prime; the LM-head matmul uses the process kernel selector.
"""

from __future__ import annotations

# Importing ``ops`` registers the ``phyai::masked_embedding_lookup`` custom op
# so callers don't have to do anything to make Dynamo / torch.compile see it.
from phyai.layers.vocab_embedding import ops as _ops  # noqa: F401
from phyai.layers.vocab_embedding.layers import (
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_to,
)


__all__ = [
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "pad_vocab_to",
]
