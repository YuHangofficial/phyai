"""Runtime view of the named communication groups of one replica."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch.distributed as dist

from phyai.parallel.layout import WORLD, RankLayout
from phyai.parallel.topology import (
    PlacementEntry,
    Topology,
    fallback_topology,
    topology_from_entries,
)
from phyai.utils import nvml

if TYPE_CHECKING:
    from phyai.parallel.process_groups import ProcessGroupPool


class Mesh:
    """Process-local access to the named groups of one replica.

    A mesh creates no process groups. :func:`phyai.parallel.init` builds them
    through a ``ProcessGroupPool`` and hands them in; a mesh constructed
    without handles (single rank, tests) still answers every size and rank
    question from the layout, and exposes torch's default group as ``world``
    when one exists.
    """

    def __init__(
        self,
        layout: RankLayout,
        *,
        rank: int = 0,
        process_groups: dict[str, dist.ProcessGroup] | None = None,
        cpu_groups: dict[str, dist.ProcessGroup] | None = None,
        pool: ProcessGroupPool | None = None,
        name: str = "model",
    ) -> None:
        if not 0 <= rank < layout.replica_world_size:
            raise ValueError(
                f"rank {rank} is outside [0, {layout.replica_world_size})."
            )
        self.layout = layout
        self.replica_world_size = layout.replica_world_size
        self.rank = rank
        self.name = name
        self._members = {
            group: layout.members_for(group, rank) for group in layout.groups
        }
        self._process_groups = dict(process_groups or {})
        self._cpu_groups = dict(cpu_groups or {})
        unknown = (set(self._process_groups) | set(self._cpu_groups)) - set(
            self._members
        )
        if unknown:
            raise ValueError(
                f"unknown parallel groups {sorted(unknown)!r}; "
                f"valid: {self.group_names!r}."
            )
        self._pool = pool
        self._placement: tuple[PlacementEntry, ...] | None = None
        self._topologies: dict[str, Topology] = {}

    # -- membership ----------------------------------------------------------

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self._members)

    def group_members(self, name: str) -> tuple[int, ...]:
        try:
            return self._members[name]
        except KeyError as error:
            raise KeyError(
                f"unknown parallel group {name!r}; valid groups: {self.group_names!r}."
            ) from error

    def group_size(self, name: str) -> int:
        return len(self.group_members(name))

    def group_rank(self, name: str) -> int:
        return self.group_members(name).index(self.rank)

    def distinct_groups(self) -> tuple[str, ...]:
        """Multi-rank groups with pairwise different members, in layout order.

        Aliases (``attention_tp`` when it equals ``dense_tp``, say) and the
        implicit ``world`` group are left out, so the result is the shortest
        description of where this rank sits, which is what process titles
        and ``repr`` want.
        """
        seen: set[tuple[int, ...]] = set()
        distinct: list[str] = []
        for name, members in self._members.items():
            if name == WORLD or len(members) <= 1 or members in seen:
                continue
            seen.add(members)
            distinct.append(name)
        return tuple(distinct)

    # -- process groups ------------------------------------------------------

    def group(self, name: str) -> dist.ProcessGroup:
        members = self.group_members(name)
        process_group = self._process_groups.get(name)
        if process_group is not None:
            return process_group
        if name == WORLD and dist.is_initialized():
            # The launcher owns the default group; it is the world group of
            # a mesh that was handed no explicit handle for it.
            return dist.group.WORLD
        if len(members) <= 1:
            raise RuntimeError(f"group {name!r} has one member and no ProcessGroup.")
        raise RuntimeError(f"group {name!r} was not initialized for rank {self.rank}.")

    def cpu_group(self, name: str) -> dist.ProcessGroup:
        """Host (gloo) group built alongside the device group of ``name``."""
        self.group_members(name)
        try:
            return self._cpu_groups[name]
        except KeyError as error:
            raise RuntimeError(
                f"group {name!r} has no host group; init builds them for NCCL "
                "meshes with pynccl enabled."
            ) from error

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._process_groups.clear()
        self._cpu_groups.clear()

    # -- placement -----------------------------------------------------------

    def set_placement(self, entries: Sequence[PlacementEntry]) -> None:
        """Record the probed placement of every replica rank."""
        if len(entries) != self.replica_world_size:
            raise ValueError(
                f"expected {self.replica_world_size} placement entries, "
                f"got {len(entries)}."
            )
        self._placement = tuple(entries)
        self._topologies.clear()

    @property
    def placement(self) -> tuple[PlacementEntry, ...] | None:
        return self._placement

    def topology(self, group: str = WORLD) -> Topology:
        """Coarse placement of ``group``'s members.

        Derived from the probed placement once :meth:`set_placement` ran, so
        a node-local TP group inside a multi-node replica reports single-node.
        Without a probe it is a device-count guess for the group's size.
        """
        members = self.group_members(group)
        topology = self._topologies.get(group)
        if topology is None:
            if self._placement is None:
                topology = fallback_topology(len(members))
            else:
                topology = topology_from_entries(
                    [self._placement[rank] for rank in members],
                    nvlink_check=nvml.nvlink_fully_connected,
                )
            self._topologies[group] = topology
        return topology

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"rank={self.rank}/{self.replica_world_size}"]
        parts.extend(
            f"{name}={self.group_rank(name)}/{self.group_size(name)}"
            for name in self.distinct_groups()
        )
        return f"Mesh({', '.join(parts)})"


__all__ = ["Mesh"]
