"""Unit tests for the shard-loader factories in phyai.weights.shards."""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import _Leg, fused, replicated, sharded, vocab


def test_replicated_copies_full_tensors_and_scalars():
    p = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
    src = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    replicated()(p, src, None)
    torch.testing.assert_close(p.data, src)

    scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
    replicated()(scalar, torch.tensor(3.5), None)  # 0-D source into a singleton
    assert scalar.data.item() == 3.5


def test_sharded_narrows_the_group_rank_slice_along_dim(fake_mesh):
    fake_mesh(tp_size=4, rank=2)
    src = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    p = nn.Parameter(torch.zeros(8, 8), requires_grad=False)
    sharded(dim=0, group="dense_tp", mesh=resolve_mesh("model"))(p, src, None)
    torch.testing.assert_close(p.data, src.narrow(0, 16, 8))

    fake_mesh(tp_size=2, rank=0)
    src = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
    p = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
    sharded(dim=1, group="dense_tp", mesh=resolve_mesh("model"))(p, src, None)
    torch.testing.assert_close(p.data, src.narrow(1, 0, 8))


def test_fused_qkv_layout(fake_mesh):
    """Fused QKV at tp=1: q/k/v each go to their fuse offsets."""
    fake_mesh(tp_size=1)
    mesh = resolve_mesh("model")
    legs = {
        "q": _Leg(offset=0, size=4, dim=0, group="dense_tp", replication_factor=1),
        "k": _Leg(offset=4, size=2, dim=0, group="dense_tp", replication_factor=1),
        "v": _Leg(offset=6, size=2, dim=0, group="dense_tp", replication_factor=1),
    }
    loader = fused(fuse_dim=0, legs=legs, mesh=mesh)
    p = nn.Parameter(torch.zeros(8, 3), requires_grad=False)

    loader(p, torch.full((4, 3), 1.0), "q")
    loader(p, torch.full((2, 3), 2.0), "k")
    loader(p, torch.full((2, 3), 3.0), "v")

    assert torch.all(p.data[0:4] == 1.0)
    assert torch.all(p.data[4:6] == 2.0)
    assert torch.all(p.data[6:8] == 3.0)


def test_vocab_shards_tile_real_rows_and_zero_fill_padding(fake_mesh):
    V, V_padded, D, tp = 100, 128, 4, 4
    per_rank = V_padded // tp  # 32
    src = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    for rank in range(3):  # ranks 0..2 hold real rows only
        fake_mesh(tp_size=tp, rank=rank)
        p = nn.Parameter(torch.zeros(per_rank, D), requires_grad=False)
        vocab(group="dense_tp", mesh=resolve_mesh("model"))(p, src, None)
        torch.testing.assert_close(p.data, src.narrow(0, rank * per_rank, per_rank))

    # Rank 3: rows 96..100 are real, the remaining 28 rows are padding.
    fake_mesh(tp_size=tp, rank=3)
    p = nn.Parameter(torch.full((per_rank, D), 7.0), requires_grad=False)
    vocab(group="dense_tp", mesh=resolve_mesh("model"))(p, src, None)
    torch.testing.assert_close(p.data[:4], src.narrow(0, 96, 4))
    assert torch.all(p.data[4:] == 0)

    # A rank whose shard starts past V is entirely padding.
    fake_mesh(tp_size=4, rank=1)
    p = nn.Parameter(torch.full((32, 2), 9.0), requires_grad=False)
    vocab(group="dense_tp", mesh=resolve_mesh("model"))(p, torch.randn(20, 2), None)
    assert torch.all(p.data == 0)
