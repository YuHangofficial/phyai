"""``Registry.validate`` must prove a universal fallback, never a fast path."""

from __future__ import annotations

import pytest
import torch

from phyai.parallel.backend import Op, Topology
from phyai.parallel.backends import GlooBackend, NcclBackend
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.registry import WORST_CASE_TOPOLOGY, Registry
from phyai.parallel.state import Mode

DGX = Topology(is_full_nvlink=True, is_single_node=True, n_nodes=1, n_gpus_per_node=8)


class _NvlinkIsland:
    """Stand-in for a custom all-reduce: one host, full NVLink, 2/4/6/8 ranks."""

    name = "island"

    def can_handle(self, *, op, mode, nbytes, dtype, world_size, topology, **extra):
        return (
            topology.is_single_node
            and topology.is_full_nvlink
            and world_size in (2, 4, 6, 8)
        )

    def supports_capture(self):
        return True

    def close(self):
        return None

    def execute(self, *, op, pg, **kwargs):
        raise AssertionError("probe-only backend")


class _SizeWhitelist(_NvlinkIsland):
    """Topology-agnostic, but only for 2, 4 or 8 ranks."""

    name = "whitelist"

    def can_handle(self, *, op, mode, nbytes, dtype, world_size, topology, **extra):
        return world_size in (2, 4, 8)


def _probe(**overrides):
    kwargs = dict(
        op=Op.ALL_REDUCE,
        mode=Mode.EAGER,
        nbytes=1024,
        dtype=torch.bfloat16,
        world_size=8,
        topology=DGX,
    )
    kwargs.update(overrides)
    return kwargs


def test_an_nvlink_fast_path_alone_is_not_a_fallback():
    assert not WORST_CASE_TOPOLOGY.is_single_node
    assert not WORST_CASE_TOPOLOGY.is_full_nvlink
    registry = Registry()
    registry.register(_NvlinkIsland())
    # It serves a DGX-shaped probe, which is what the old check asked ...
    assert registry.has(**_probe())
    # ... but on any machine validation must refuse to call it the fallback.
    with pytest.raises(NoBackendError, match="8-rank group under the worst-case"):
        registry.validate(group_sizes={8})
    registry.register(NcclBackend())
    registry.validate(group_sizes={3, 8, 16})


def test_size_whitelists_are_probed_per_actual_group_size():
    registry = Registry()
    registry.register(_SizeWhitelist())
    # Single-rank groups need no collective; the rest of this mesh is covered,
    # so the whitelist is a valid fallback for it.
    registry.validate(group_sizes={1})
    registry.validate(group_sizes={1, 2, 4})
    # Sizes are not monotonic: passing 8 says nothing about 3 or 16.
    with pytest.raises(NoBackendError, match="3-rank"):
        registry.validate(group_sizes={2, 3})
    with pytest.raises(NoBackendError, match="16-rank"):
        registry.validate(group_sizes={8, 16})


def test_gloo_only_registry_is_checked_in_eager_mode_only():
    registry = Registry()
    registry.register(GlooBackend())
    registry.validate(group_sizes={2, 5})


def test_preferred_names_must_be_registered():
    registry = Registry()
    registry.register(NcclBackend(), prefer_for={Op.ALL_REDUCE})
    registry.validate(group_sizes={2})
    registry._prefer[Op.ALL_GATHER] = ["pynccl"]
    with pytest.raises(NoBackendError, match="prefer_for"):
        registry.validate(group_sizes={2})
