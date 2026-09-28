"""Configuration contracts for overlapping domains within one replica."""

import pickle

import pytest

from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    EngineConfig,
    MoeParallelConfig,
    OuterParallelConfig,
    ParallelConfig,
)


def test_domains_share_one_rank_pool():
    config = ParallelConfig(
        outer=OuterParallelConfig(cfg_size=2),
        dense=DenseParallelConfig(tp_size=8),
        attention=AttentionParallelConfig(dp_size=2),
        moe=MoeParallelConfig(ep_size=4),
    )
    assert config.infer_replica_world_size() == 16
    resolved = config.resolve(16)
    assert resolved.dense.tp_size == 8
    assert resolved.attention.tp_size == 4
    assert resolved.moe.tp_size == 2
    assert pickle.loads(pickle.dumps(config)) == config


def test_partial_domains_use_smallest_compatible_rank_pool():
    config = ParallelConfig(
        attention=AttentionParallelConfig(dp_size=2),
        moe=MoeParallelConfig(ep_size=3),
    )
    assert config.infer_replica_world_size() == 6
    assert config.resolve(12).attention.tp_size == 6


def test_explicit_domain_products_must_agree():
    config = ParallelConfig(
        dense=DenseParallelConfig(tp_size=8),
        attention=AttentionParallelConfig(tp_size=2, dp_size=2),
    )
    with pytest.raises(ValueError, match="different model-scope"):
        config.infer_replica_world_size()
    with pytest.raises(ValueError, match="attention parallel sizes"):
        config.resolve(8)


def test_launcher_world_resolves_omitted_tp_dimensions():
    resolved = ParallelConfig(moe=MoeParallelConfig(ep_size=4)).resolve(8)
    assert resolved.replica_world_size == 8
    assert resolved.dense.tp_size == resolved.attention.tp_size == 8
    assert resolved.moe.tp_size == 2
    with pytest.raises(ValueError, match="do not divide"):
        ParallelConfig(moe=MoeParallelConfig(ep_size=3)).resolve(8)


def test_sequence_parallel_does_not_multiply_world():
    config = ParallelConfig(
        dense=DenseParallelConfig(tp_size=4, sequence_parallel=True)
    )
    assert config.infer_replica_world_size() == 4


def test_decode_cp_subdivides_tp_while_cp_factors_the_scope():
    config = ParallelConfig(
        attention=AttentionParallelConfig(cp_size=2, decode_cp_size=2)
    )
    assert config.infer_replica_world_size() == 4
    config.resolve(8)
    with pytest.raises(ValueError, match="must divide"):
        config.resolve(6)


@pytest.mark.parametrize("value", (0, True, "2"))
def test_sizes_must_be_positive_integers(value):
    with pytest.raises(ValueError, match="tp_size"):
        DenseParallelConfig(tp_size=value)
    with pytest.raises(ValueError, match="replica_world_size"):
        ParallelConfig().resolve(value)


def test_invalid_nested_config_fails_before_launch():
    with pytest.raises(TypeError, match="DenseParallelConfig"):
        ParallelConfig(dense={"tp_size": 2})
    with pytest.raises(ValueError, match="divisible"):
        ParallelConfig(outer=OuterParallelConfig(pipeline_size=3)).resolve(8)


@pytest.mark.parametrize(
    "name", ("PHYAI_TP_SIZE", "PHYAI_EP_SIZE", "PHYAI_REPLICA_COUNT")
)
def test_topology_is_not_overridden_through_environment(monkeypatch, name):
    monkeypatch.setenv(name, "4")
    with pytest.raises(ValueError, match=name):
        EngineConfig.from_env(EngineConfig())
