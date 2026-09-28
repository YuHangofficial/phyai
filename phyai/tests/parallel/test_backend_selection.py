"""PyNCCL capability answers by group identity (no NCCL library is loaded)."""

from __future__ import annotations

import torch

from phyai.parallel.backend import Op, Topology
from phyai.parallel.backends.pynccl import PyNCCLBackend
from phyai.parallel.state import Mode


def _probe(**overrides):
    kwargs = dict(
        op=Op.ALL_REDUCE,
        mode=Mode.GRAPH_CAPTURING,
        nbytes=1024,
        dtype=torch.bfloat16,
        world_size=2,
        topology=Topology(
            is_full_nvlink=True, is_single_node=True, n_nodes=1, n_gpus_per_node=8
        ),
    )
    kwargs.update(overrides)
    return kwargs


def test_pynccl_serves_only_attached_groups_under_capture():
    backend = PyNCCLBackend()
    backend._comms[("model", "dense_tp")] = object()  # stand-in for an attached comm

    assert backend.can_handle(**_probe(mesh_name="model", group="dense_tp"))
    # A group attach() never built (world here) or another mesh must fall
    # through to NcclBackend instead of failing later inside _comm_for.
    assert not backend.can_handle(**_probe(mesh_name="model", group="world"))
    assert not backend.can_handle(**_probe(mesh_name="other", group="dense_tp"))
    # No identity is registry.validate's capability probe: stay general.
    assert backend.can_handle(**_probe())
    # Eager mode belongs to NcclBackend, even for an attached group.
    assert not backend.can_handle(
        **_probe(mode=Mode.EAGER, mesh_name="model", group="dense_tp")
    )
