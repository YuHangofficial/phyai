"""Mesh placement: device-count fallback, per-group topology, world group, aliases."""

from __future__ import annotations

import pytest

from phyai.engine_config import DenseParallelConfig, OuterParallelConfig, ParallelConfig
from phyai.parallel import topology as topology_module
from phyai.parallel.backend import Topology
from phyai.parallel.layout import build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.parallel.topology import (
    PlacementEntry,
    fallback_topology,
    topology_from_entries,
)
from phyai.utils import nvml
from phyai.utils.nvml import physical_device_index


def test_fallback_topology_fills_nodes_with_visible_devices(monkeypatch):
    monkeypatch.setattr(topology_module.torch.cuda, "device_count", lambda: 4)
    topology = fallback_topology(8)
    assert not topology.is_single_node
    assert (topology.n_nodes, topology.n_gpus_per_node) == (2, 4)
    # NVLink cannot be known without a probe: only a lone rank is "connected".
    assert not topology.is_full_nvlink

    monkeypatch.setattr(topology_module.torch.cuda, "device_count", lambda: 8)
    assert fallback_topology(8).is_single_node
    assert fallback_topology(1).is_full_nvlink


def _entry(
    node: str, physical: int | None, clique: str | None = None
) -> PlacementEntry:
    return PlacementEntry(node=node, physical=physical, clique=clique)


def test_single_node_topology_asks_nvml_about_every_gpu():
    asked: list[list[int]] = []

    def check(ids):
        asked.append(list(ids))
        return True

    topology = topology_from_entries(
        [_entry("host-a", 2), _entry("host-a", 6)], nvlink_check=check
    )
    assert topology == Topology(
        is_full_nvlink=True, is_single_node=True, n_nodes=1, n_gpus_per_node=2
    )
    assert asked == [[2, 6]]
    assert not topology_from_entries(
        [_entry("host-a", 0), _entry("host-a", 1)], nvlink_check=lambda ids: False
    ).is_full_nvlink


def test_multi_node_topology_counts_nodes_and_uses_fabric_cliques():
    def never(ids):
        pytest.fail("NVML must not be consulted across hosts")

    entries = [
        _entry("host-a", 0, "fabric:1"),
        _entry("host-a", 1, "fabric:1"),
        _entry("host-b", 0, "fabric:1"),
        _entry("host-b", 1, "fabric:1"),
    ]
    topology = topology_from_entries(entries, nvlink_check=never)
    assert (topology.n_nodes, topology.n_gpus_per_node) == (2, 2)
    assert not topology.is_single_node
    # One NVSwitch fabric domain spanning both hosts (NVL72 style).
    assert topology.is_full_nvlink

    split = entries[:3] + [_entry("host-b", 1, "fabric:2")]
    assert not topology_from_entries(split, nvlink_check=never).is_full_nvlink
    no_fabric = [_entry("host-a", 0), _entry("host-b", 0)]
    assert not topology_from_entries(no_fabric, nvlink_check=never).is_full_nvlink


def test_cpu_ranks_never_claim_nvlink():
    def never(ids):
        pytest.fail("no NVML query for CPU ranks")

    topology = topology_from_entries(
        [_entry("host-a", None), _entry("host-a", None)], nvlink_check=never
    )
    assert topology.is_single_node
    assert not topology.is_full_nvlink


def test_physical_index_follows_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert physical_device_index(0) == 3
    assert physical_device_index(1) == 5


def _cfg2_tp2_mesh(rank: int) -> Mesh:
    layout = build_rank_layout(
        ParallelConfig(
            outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
        )
    )
    return Mesh(layout, rank=rank)


def test_topology_is_answered_per_group_from_one_placement_probe(monkeypatch):
    # cfg=2 x tp=2 over two hosts: each dense_tp group is node-local while the
    # cfg groups and the world span both hosts. One gather answers all three.
    asked: list[list[int]] = []

    def check(ids):
        asked.append(list(ids))
        return True

    monkeypatch.setattr(nvml, "nvlink_fully_connected", check)
    monkeypatch.setattr(topology_module.torch.cuda, "device_count", lambda: 8)
    mesh = _cfg2_tp2_mesh(rank=1)

    # Before the probe every answer is the device-count guess for the group.
    assert mesh.topology("dense_tp") == fallback_topology(2)
    assert mesh.placement is None

    mesh.set_placement(
        [
            _entry("host-a", 0),
            _entry("host-a", 1),
            _entry("host-b", 0),
            _entry("host-b", 1),
        ]
    )
    assert mesh.topology("dense_tp") == Topology(
        is_full_nvlink=True, is_single_node=True, n_nodes=1, n_gpus_per_node=2
    )
    assert asked == [[0, 1]]
    assert mesh.topology("cfg") == Topology(
        is_full_nvlink=False, is_single_node=False, n_nodes=2, n_gpus_per_node=1
    )
    assert mesh.topology() == Topology(
        is_full_nvlink=False, is_single_node=False, n_nodes=2, n_gpus_per_node=2
    )
    # Cached per group: the NVML check ran once.
    mesh.topology("dense_tp")
    assert asked == [[0, 1]]
    with pytest.raises(ValueError, match="placement entries"):
        mesh.set_placement([_entry("host-a", 0)])


def test_distinct_groups_skip_world_and_aliases():
    mesh = _cfg2_tp2_mesh(rank=3)
    # dense_tp, attention_tp, moe_tp and moe_tp_ep all cover the same ranks;
    # the layout names dense_tp first, and world is implied by rank/size.
    assert mesh.distinct_groups() == ("cfg", "dense_tp")
    assert repr(mesh) == "Mesh(name='model', rank=3/4, cfg=1/2, dense_tp=1/2)"

    single = Mesh(build_rank_layout(ParallelConfig()))
    assert single.distinct_groups() == ()
    assert repr(single) == "Mesh(name='model', rank=0/1)"


def test_world_group_membership_with_and_without_peers():
    mesh = Mesh(build_rank_layout(ParallelConfig(), 2), rank=1)
    assert mesh.group_size("world") == 2
    assert mesh.group_rank("world") == 1
    assert mesh.group_members("world") == (0, 1)

    single = Mesh(build_rank_layout(ParallelConfig()))
    assert single.group_size("world") == 1
    assert single.group_rank("world") == 0
    assert single.group_members("world") == (0,)
    with pytest.raises(RuntimeError, match="one member"):
        single.group("dense_tp")


def test_unknown_group_and_handle_fail_loudly():
    mesh = Mesh(build_rank_layout(ParallelConfig(), 2), rank=0)

    with pytest.raises(KeyError, match="valid groups"):
        mesh.group_size("transformer_tp")
    with pytest.raises(ValueError, match="unknown parallel groups"):
        Mesh(build_rank_layout(ParallelConfig(), 2), process_groups={"tp": object()})
    with pytest.raises(RuntimeError, match="no host group"):
        mesh.cpu_group("dense_tp")
