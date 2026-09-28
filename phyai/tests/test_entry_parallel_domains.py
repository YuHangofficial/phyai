"""The default ``Entry`` parallel contract: declared, implied and rejected domains."""

from __future__ import annotations

import pytest

from phyai.engine import Entry, EntryArgs
from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    MoeParallelConfig,
    OuterParallelConfig,
    ParallelConfig,
)


class _Args(EntryArgs):
    pass


def _plugin(*domains: str) -> type[Entry]:
    class Plugin(Entry):
        name = "contract-test"
        args_cls = _Args
        parallel_domains = frozenset(domains)

        def setup(self, args):
            raise NotImplementedError

        def step(self, request):
            raise NotImplementedError

    return Plugin


def test_multi_rank_outer_dimensions_must_be_declared():
    _plugin().validate_parallel(ParallelConfig())  # single rank: nothing to declare
    parallel = ParallelConfig(
        outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
    )
    with pytest.raises(ValueError, match="'cfg'"):
        _plugin("dense").validate_parallel(parallel)
    _plugin("dense", "cfg").validate_parallel(parallel)


def test_model_scope_needs_one_declared_model_domain():
    with pytest.raises(
        ValueError, match=r"supports parallel domains \['cfg'\].*model_scope"
    ):
        _plugin("cfg").validate_parallel(
            ParallelConfig(dense=DenseParallelConfig(tp_size=2))
        )


def test_undeclared_domains_may_follow_a_declared_tp_layout():
    # attention and moe resolve to the dense_tp members and stay TP-only.
    _plugin("dense").validate_parallel(
        ParallelConfig(dense=DenseParallelConfig(tp_size=4))
    )
    # Spelling out the same TP size is redundant, not a new layout.
    _plugin("dense").validate_parallel(
        ParallelConfig(
            dense=DenseParallelConfig(tp_size=4),
            attention=AttentionParallelConfig(tp_size=4),
        )
    )


def test_undeclared_domain_with_its_own_layout_is_rejected():
    # dense resolves to tp=4 while the declared attention domain uses tp=2 x dp=2.
    with pytest.raises(ValueError, match="'dense': 'tp groups differ"):
        _plugin("attention").validate_parallel(
            ParallelConfig(attention=AttentionParallelConfig(tp_size=2, dp_size=2))
        )
    # Expert parallelism is more than TP, whatever the resulting members.
    with pytest.raises(ValueError, match="'moe': 'uses more than"):
        _plugin("dense").validate_parallel(
            ParallelConfig(
                dense=DenseParallelConfig(tp_size=4), moe=MoeParallelConfig(ep_size=2)
            )
        )
    # Sequence parallelism is a dense feature the plugin did not declare.
    with pytest.raises(ValueError, match="'dense': 'uses more than"):
        _plugin("attention").validate_parallel(
            ParallelConfig(
                attention=AttentionParallelConfig(tp_size=2),
                dense=DenseParallelConfig(sequence_parallel=True),
            )
        )
