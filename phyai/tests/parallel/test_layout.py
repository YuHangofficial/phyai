"""Golden rank memberships for overlapping parallel group families."""

import pickle

import pytest

from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    MoeParallelConfig,
    OuterParallelConfig,
    ParallelConfig,
)
from phyai.parallel.layout import (
    GROUP_NAMES,
    RankLayout,
    build_rank_layout,
    memberships,
)


def test_cfg_outer_tp_inner():
    layout = build_rank_layout(
        ParallelConfig(
            outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
        )
    )
    assert layout.replica_world_size == 4
    assert layout.groups_for("dense_tp") == ((0, 1), (2, 3))
    assert layout.groups_for("cfg") == ((0, 2), (1, 3))
    assert layout.members_for("cfg", 1) == (1, 3)
    assert layout.group_rank("dense_tp", 3) == 1
    assert dict(memberships(layout, 2))["cfg"] == 1


def test_attention_dp_and_moe_ep_overlap_dense_tp():
    layout = build_rank_layout(
        ParallelConfig(
            dense=DenseParallelConfig(tp_size=8),
            attention=AttentionParallelConfig(dp_size=2),
            moe=MoeParallelConfig(ep_size=4),
        )
    )
    assert layout.groups_for("dense_tp") == (tuple(range(8)),)
    assert layout.groups_for("attention_tp") == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert layout.groups_for("attention_dp") == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert layout.groups_for("moe_tp") == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert layout.groups_for("moe_ep") == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert layout.groups_for("moe_tp_ep") == (tuple(range(8)),)


def test_pipeline_cfg_and_domain_offsets():
    layout = build_rank_layout(
        ParallelConfig(
            outer=OuterParallelConfig(pipeline_size=2, cfg_size=2),
            dense=DenseParallelConfig(tp_size=2),
        )
    )
    assert layout.groups_for("pipeline") == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert layout.groups_for("cfg") == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert layout.groups_for("dense_tp") == ((0, 1), (2, 3), (4, 5), (6, 7))


def test_context_groups_and_phase_subgroups():
    layout = build_rank_layout(
        ParallelConfig(
            attention=AttentionParallelConfig(
                tp_size=4, cp_size=2, dp_size=2, decode_cp_size=2
            )
        )
    )
    assert layout.members_for("attention_tp", 13) == (12, 13, 14, 15)
    assert layout.members_for("attention_cp", 13) == (9, 13)
    assert layout.members_for("attention_dp", 13) == (5, 13)
    assert layout.members_for("attention_decode_cp", 13) == (12, 13)


def test_every_group_family_partitions_all_ranks():
    for world in (1, 2, 4, 8):
        layout = build_rank_layout(ParallelConfig(), world)
        assert set(layout.groups) == set(GROUP_NAMES)
        for groups in layout.groups.values():
            assert sorted(rank for group in groups for rank in group) == list(
                range(world)
            )


def test_missing_duplicate_and_invalid_members_fail():
    for groups in (((0,),), ((0, 1), (1,)), ((0, 2),), ((True, 0),), ((1, 0),)):
        with pytest.raises(ValueError):
            RankLayout(2, {"world": ((0, 1),), "dense_tp": groups})


def test_unknown_group_and_rank_fail_loudly():
    layout = build_rank_layout(ParallelConfig(), 2)
    with pytest.raises(KeyError, match="valid groups"):
        layout.members_for("tp", 0)
    with pytest.raises(ValueError, match="outside"):
        layout.members_for("dense_tp", 2)
    with pytest.raises(ValueError, match="outside"):
        memberships(layout, 2)


def test_layout_snapshots_memberships_and_survives_worker_serialization():
    groups = {"world": ((0, 1),), "dense_tp": ((0, 1),)}
    layout = RankLayout(2, groups)
    groups["dense_tp"] = ((0,), (1,))
    assert layout.members_for("dense_tp", 0) == (0, 1)
    with pytest.raises(TypeError):
        layout.groups["dense_tp"] = ((0,), (1,))
    assert pickle.loads(pickle.dumps(layout)) == layout
