"""Placement facts of a replica and the coarse ``Topology`` derived from them.

Placement is probed, not configured: every rank contributes where it runs
(host identity, NVML device index, NVLink fabric domain) and the aggregate
answers the questions a collective backend asks about a group, such as
"single node?" or "fully NVLink-connected?". Keeping the per-rank entries
lets :class:`~phyai.parallel.mesh.Mesh` answer them for any group without
another round of communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from phyai.utils import nvml


@dataclass(frozen=True)
class Topology:
    """Static topology hints visible to ``Backend.can_handle``.

    Coarse-grained on purpose; backends that need finer detail (for example
    custom all-reduce kernels) should run their own probes.
    """

    is_full_nvlink: bool
    is_single_node: bool
    n_nodes: int
    n_gpus_per_node: int


@dataclass(frozen=True)
class PlacementEntry:
    """Where one rank runs.

    ``physical`` is the rank's NVML device index (``None`` for CPU ranks or
    when NVML is unavailable); ``clique`` is the NVLink fabric domain
    identity (``None`` when the device is not on an NVSwitch fabric).
    """

    node: str
    physical: int | None
    clique: str | None


def fallback_topology(size: int) -> Topology:
    """Placement guess from the visible device count alone.

    Used when no process group exists (single rank) or when the runtime probe
    failed. Assumes ``size`` ranks fill nodes of ``device_count`` GPUs. NVLink
    is unknown without a probe, so only a lone rank counts as fully connected.
    """
    try:
        visible_gpus = max(torch.cuda.device_count(), 1)
    except Exception:
        visible_gpus = 1
    node_count = max(1, (size + visible_gpus - 1) // visible_gpus)
    gpus_per_node = (size + node_count - 1) // node_count
    return Topology(
        is_full_nvlink=size == 1,
        is_single_node=node_count == 1,
        n_nodes=node_count,
        n_gpus_per_node=gpus_per_node,
    )


def topology_from_entries(
    entries: Sequence[PlacementEntry],
    *,
    nvlink_check: Callable[[Sequence[int]], bool],
) -> Topology:
    """Aggregate the placement of one group's members into a ``Topology``.

    Pure function shared by :meth:`Mesh.topology` and its tests. A single-node
    group asks ``nvlink_check`` about its members' physical devices; a
    multi-node group counts as fully connected only when every member sits on
    the same NVSwitch fabric domain.
    """
    per_node: dict[str, int] = {}
    for entry in entries:
        per_node[entry.node] = per_node.get(entry.node, 0) + 1
    n_nodes = len(per_node)
    single_node = n_nodes == 1
    physical = [entry.physical for entry in entries]
    if any(index is None for index in physical):
        # CPU ranks (gloo) or a rank without NVML: NVLink is not a property
        # of this group.
        full_nvlink = False
    elif single_node:
        full_nvlink = bool(nvlink_check([int(index) for index in physical]))
    else:
        cliques = {entry.clique for entry in entries}
        full_nvlink = len(cliques) == 1 and None not in cliques
    return Topology(
        is_full_nvlink=full_nvlink,
        is_single_node=single_node,
        n_nodes=n_nodes,
        n_gpus_per_node=max(per_node.values()),
    )


def local_placement(device: torch.device) -> PlacementEntry:
    """This rank's placement entry for ``device``.

    NVML lookups degrade to ``None`` instead of raising: the caller gathers
    the entry collectively, and a rank that skipped the gather would leave
    the others blocked in it.
    """
    physical: int | None = None
    clique: str | None = None
    if device.type == "cuda" and nvml.nvml_available():
        try:
            physical = nvml.physical_device_index(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            clique = nvml.fabric_clique(physical)
        except Exception:  # noqa: BLE001 - degrade to "unknown", never skip the gather
            physical, clique = None, None
    return PlacementEntry(node=nvml.node_identity(), physical=physical, clique=clique)


def probe_placement(
    group: dist.ProcessGroup | None, *, device: torch.device
) -> tuple[PlacementEntry, ...]:
    """Gather every rank's :class:`PlacementEntry` with one ``all_gather_object``.

    NVML cannot see other hosts, which is why the gather comes first. Must be
    called collectively by every rank of ``group``, like ``init`` itself.
    """
    local = local_placement(device)
    gathered: list[PlacementEntry | None] = [None] * dist.get_world_size(group)
    # NCCL's all_gather_object stages through the *current* device; pin it to
    # the probed device so a standalone init(device="cuda:K") does not put the
    # world communicator on whatever happened to be current.
    if device.type == "cuda":
        with torch.cuda.device(device):
            dist.all_gather_object(gathered, local, group=group)
    else:
        dist.all_gather_object(gathered, local, group=group)
    return tuple(gathered)


__all__ = [
    "PlacementEntry",
    "Topology",
    "fallback_topology",
    "local_placement",
    "probe_placement",
    "topology_from_entries",
]
