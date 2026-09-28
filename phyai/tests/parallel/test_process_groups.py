"""Group ownership and creation order, including nonmember ranks."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    MoeParallelConfig,
    ParallelConfig,
)
from phyai.parallel.process_groups import ProcessGroupPool
from phyai.parallel.layout import build_rank_layout


def test_pool_reuses_groups_and_releases_only_owned_handles(monkeypatch):
    rank = 3  # a nonmember of half the groups: creates them, owns none of them
    world = object()
    created = []
    destroyed = []
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", destroyed.append)

    def new_group(*, ranks, backend):
        handle = object()
        created.append((tuple(ranks), backend, handle))
        return handle if rank in ranks else dist.GroupMember.NON_GROUP_MEMBER

    monkeypatch.setattr(dist, "new_group", new_group)
    layout = build_rank_layout(
        ParallelConfig(
            dense=DenseParallelConfig(tp_size=4),
            attention=AttentionParallelConfig(dp_size=2),
            moe=MoeParallelConfig(ep_size=2),
        )
    )
    pool = ProcessGroupPool(layout, backend="nccl", build_cpu=True)
    expected = [((0, 1, 2, 3), "gloo")]
    for members in ((0, 1), (2, 3), (0, 2), (1, 3)):
        expected.extend((members, backend) for backend in ("nccl", "gloo"))
    assert [(members, backend) for members, backend, _ in created] == expected
    assert pool.get(tuple(range(4)), "nccl") is world
    pool.close()
    pool.close()
    assert destroyed == [
        handle for members, _, handle in reversed(created) if rank in members
    ]
    assert world not in destroyed


def test_partial_group_creation_is_cleaned_up(monkeypatch):
    owned = object()
    destroyed = []
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", destroyed.append)

    def new_group(*, ranks, backend):
        if ranks == [0, 1]:
            return owned
        raise RuntimeError("group creation failed")

    monkeypatch.setattr(dist, "new_group", new_group)
    layout = build_rank_layout(
        ParallelConfig(attention=AttentionParallelConfig(tp_size=2, dp_size=2))
    )
    with pytest.raises(RuntimeError, match="group creation failed"):
        ProcessGroupPool(layout, backend="gloo", build_cpu=False)
    assert destroyed == [owned]


def test_composite_default_group_stands_in_for_every_world_backend(monkeypatch):
    # An external launcher's init_process_group() without a backend yields a
    # composite default group that torch reports as "undefined". It serves
    # both device types, so no second world-sized group may be created.
    world = object()
    created = []
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(dist, "get_backend", lambda: "undefined")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist, "new_group", lambda *, ranks, backend: created.append((ranks, backend))
    )
    layout = build_rank_layout(ParallelConfig(dense=DenseParallelConfig(tp_size=2)))
    pool = ProcessGroupPool(layout, backend="nccl", build_cpu=True)

    assert created == []
    assert pool.get((0, 1), "nccl") is world
    assert pool.get((0, 1), "gloo") is world
