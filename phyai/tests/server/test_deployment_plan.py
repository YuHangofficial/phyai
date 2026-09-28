"""Tests for deployment placement of complete replica_count."""

from __future__ import annotations

import pytest
from phyai.engine_config import ParallelConfig, DenseParallelConfig, OuterParallelConfig

from phyai.server.deployment import (
    DeploymentPlan,
    NodeResources,
    normalize_device_indices,
)


def test_deployment_places_complete_replicas_in_resource_order():
    plan = DeploymentPlan.build(
        ParallelConfig(
            outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
        ),
        2,
        (
            NodeResources(0, (0, 1, 2), "node-a"),
            NodeResources(1, (0, 1, 2, 3, 4), "node-b"),
        ),
        output_rank=1,
    )

    assert plan.replica_world_size == 4
    assert [worker.device_index for worker in plan.workers_for_replica(0)] == [
        0,
        1,
        2,
        0,
    ]
    assert [worker.replica_rank for worker in plan.workers_for_replica(1)] == [
        0,
        1,
        2,
        3,
    ]
    assert [worker.worker_id for worker in plan.workers if worker.is_output_rank] == [
        1,
        5,
    ]
    assert plan.workers[3].group_rank("cfg") == 1
    assert plan.workers[3].group_rank("dense_tp") == 1


def test_build_uses_only_the_slots_it_needs_and_rejects_invalid_plans():
    plan = DeploymentPlan.build(ParallelConfig(), 2, (NodeResources(0, (0, 1, 2)),))
    assert [worker.device_index for worker in plan.workers] == [0, 1]

    with pytest.raises(ValueError, match="needs 3 worker slots"):
        DeploymentPlan.build(ParallelConfig(), 3, (NodeResources(0, (0, 1)),))
    with pytest.raises(ValueError, match="output_rank"):
        DeploymentPlan.build(
            ParallelConfig(dense=DenseParallelConfig(tp_size=2)),
            1,
            (NodeResources(0, (0, 1)),),
            output_rank=2,
        )
    with pytest.raises(ValueError, match="contiguous"):
        DeploymentPlan.build(ParallelConfig(), 1, (NodeResources(1, (0,)),))


def test_device_entries_normalize_to_unique_indices():
    assert normalize_device_indices((0, "3", " 2 "), owner="devices") == (0, 3, 2)
    assert NodeResources(0, ("1", 0)).devices == (1, 0)
    with pytest.raises(ValueError, match="unique"):
        NodeResources(0, (1, "1"))


@pytest.mark.parametrize("device", (-1, "0,1", "GPU-a"))
def test_non_index_device_entries_are_rejected(device):
    with pytest.raises(ValueError, match="non-negative device indices"):
        NodeResources(0, (device,))


def test_multi_node_replica_requires_the_root_node_address():
    nodes_without_addresses = (NodeResources(0, (0, 1)), NodeResources(1, (0, 1)))
    with pytest.raises(ValueError, match="root node 0 has no address"):
        DeploymentPlan.build(
            ParallelConfig(dense=DenseParallelConfig(tp_size=4)),
            1,
            nodes_without_addresses,
        )

    # Only the root needs an address; a single-node replica needs none.
    DeploymentPlan.build(
        ParallelConfig(dense=DenseParallelConfig(tp_size=4)),
        1,
        (NodeResources(0, (0, 1), "node-a"), NodeResources(1, (0, 1))),
    )
    DeploymentPlan.build(
        ParallelConfig(dense=DenseParallelConfig(tp_size=2)),
        2,
        (NodeResources(0, (0, 1)), NodeResources(1, (0, 1))),
    )
