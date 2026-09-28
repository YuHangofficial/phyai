"""Validation tests for Cosmos3 parallel topology limits."""

from __future__ import annotations

import pytest
import torch

from phyai.engine_config import (
    ParallelConfig,
    OuterParallelConfig,
    DenseParallelConfig,
    AttentionParallelConfig,
    MoeParallelConfig,
)
from phyai.models.cosmos3.main_cosmos3 import Cosmos3Entry
from phyai.models.cosmos3.main_cosmos3_policy import Cosmos3PolicyEntry
from phyai.models.cosmos3.guidance import combine_cfg


@pytest.mark.parametrize("entry", (Cosmos3Entry, Cosmos3PolicyEntry))
def test_cfg_parallel_rejects_more_than_two_branches(entry: type) -> None:
    with pytest.raises(ValueError, match="cfg_size=1 or 2"):
        entry.validate_parallel(ParallelConfig(outer=OuterParallelConfig(cfg_size=3)))


@pytest.mark.parametrize("entry", (Cosmos3Entry, Cosmos3PolicyEntry))
def test_cosmos_plugins_declare_supported_domains(entry: type) -> None:
    assert entry.parallel_domains == frozenset({"cfg", "dense", "attention"})
    parallel = ParallelConfig(
        outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
    )
    entry.validate_parallel(parallel)
    assert parallel.infer_replica_world_size() == 4


@pytest.mark.parametrize("entry", (Cosmos3Entry, Cosmos3PolicyEntry))
@pytest.mark.parametrize(
    "parallel",
    (
        ParallelConfig(moe=MoeParallelConfig(ep_size=2)),
        ParallelConfig(dense=DenseParallelConfig(sequence_parallel=True)),
        ParallelConfig(attention=AttentionParallelConfig(cp_size=2)),
    ),
)
def test_undeclared_domains_are_rejected_by_the_default_contract(
    entry: type, parallel: ParallelConfig
) -> None:
    with pytest.raises(ValueError):
        entry.validate_parallel(parallel)


@pytest.mark.parametrize("guidance_scale", (0.0, 1.0))
def test_disabled_cfg_returns_conditional_value(guidance_scale: float) -> None:
    cond = torch.tensor([1.0, 2.0])
    uncond = torch.tensor([10.0, 20.0])

    assert combine_cfg(cond, uncond, guidance_scale) is cond


def test_enabled_cfg_combines_conditional_and_unconditional_values() -> None:
    cond = torch.tensor([1.0, 2.0])
    uncond = torch.tensor([10.0, 20.0])

    actual = combine_cfg(cond, uncond, 2.0)

    assert torch.equal(actual, torch.tensor([-8.0, -16.0]))
