"""Deployment policy and placement of complete model replicas.

Deployment decides where complete model replicas run — inline in the caller
process, on locally spawned workers, or on externally launched ranks — and
maps replicas onto node and device resources. It sits between the public
:class:`phyai.engine.Engine` facade and the executor implementations; model
parallelism remains owned by ``ParallelConfig``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from phyai.engine_config import ParallelConfig
from phyai.parallel.layout import RankLayout, build_rank_layout, memberships
from phyai.server.dispatcher import RequestDispatcher
from phyai.server.executor import ExternalExecutor, InlineExecutor


DeploymentMode = Literal["auto", "local", "external"]


def normalize_device_indices(
    devices: Sequence[int | str],
    *,
    owner: str,
) -> tuple[int, ...]:
    """Normalize device entries to logical torch device indices.

    Managed workers inherit the launching process's full device visibility
    and bind by index (sglang/vllm style), so a device slot is an ordinal
    into that visible ordering — ``0`` means ``cuda:0`` *as this process
    sees it*. Physical GPU selection stays a shell-level concern: set
    ``CUDA_VISIBLE_DEVICES`` on the launching process, then pass indices
    into that mask here. Decimal strings are accepted for CLI convenience.
    """
    normalized: list[int] = []
    for device in devices:
        index: int | None = None
        if isinstance(device, int) and not isinstance(device, bool):
            index = device
        elif isinstance(device, str) and device.strip().isdecimal():
            index = int(device.strip())
        if index is None or index < 0:
            raise ValueError(
                f"{owner} must contain non-negative device indices, got "
                f"{device!r}. Select physical GPUs with CUDA_VISIBLE_DEVICES "
                "on the launching process; device entries index into that "
                "visible ordering."
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{owner} must be unique, got {tuple(normalized)!r}.")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Placement and request-routing policy outside model parallelism."""

    mode: DeploymentMode = "auto"
    replica_count: int = 1
    devices: tuple[int, ...] = ()
    output_rank: int = 0
    process_config: Any = None
    auto_start: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "local", "external"):
            raise ValueError(f"unknown deployment mode {self.mode!r}.")
        if (
            not isinstance(self.replica_count, int)
            or isinstance(self.replica_count, bool)
            or self.replica_count < 1
        ):
            raise ValueError(
                f"replica_count must be a positive int, got {self.replica_count!r}."
            )
        if not isinstance(self.auto_start, bool):
            raise TypeError("auto_start must be a bool.")
        devices = normalize_device_indices(self.devices, owner="devices")
        object.__setattr__(self, "devices", devices)
        if self.mode == "external" and devices:
            raise ValueError("external deployment placement belongs to the launcher.")
        if self.mode == "external" and self.replica_count != 1:
            raise ValueError("external deployment represents exactly one replica.")
        if self.mode == "external" and self.process_config is not None:
            raise ValueError("external deployment lifecycle belongs to the launcher.")
        if (
            not isinstance(self.output_rank, int)
            or isinstance(self.output_rank, bool)
            or self.output_rank < 0
        ):
            raise ValueError("output_rank must be a non-negative int.")

    @classmethod
    def local(
        cls,
        devices: Sequence[int | str] = (),
        *,
        replica_count: int = 1,
        process_config: Any = None,
        output_rank: int = 0,
        auto_start: bool = True,
    ) -> "DeploymentConfig":
        return cls(
            mode="local",
            replica_count=replica_count,
            devices=tuple(devices),
            output_rank=output_rank,
            process_config=process_config,
            auto_start=auto_start,
        )

    @classmethod
    def external(
        cls,
        *,
        output_rank: int = 0,
    ) -> "DeploymentConfig":
        return cls(
            mode="external",
            output_rank=output_rank,
        )


@dataclass(frozen=True, slots=True)
class NodeResources:
    """Visible accelerator slots and rendezvous address for one node."""

    rank: int
    devices: tuple[int, ...]
    address: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 0
        ):
            raise ValueError(
                f"node rank must be a non-negative int, got {self.rank!r}."
            )
        devices = normalize_device_indices(
            self.devices, owner=f"node {self.rank} devices"
        )
        object.__setattr__(self, "devices", devices)
        if self.address is not None:
            address = str(self.address).strip()
            if not address or "\0" in address:
                raise ValueError("node address must be None or a non-empty string.")
            object.__setattr__(self, "address", address)


@dataclass(frozen=True, slots=True)
class WorkerPlacement:
    """Identity and device assignment for one replica rank.

    ``device_index`` is the torch device ordinal this worker binds
    (``cuda:{device_index}``). Workers inherit the launching process's
    device visibility unchanged, so the index means the same GPU in every
    process — the property CUDA-IPC tensor transport relies on.
    """

    worker_id: int
    replica_id: int
    replica_rank: int
    node_rank: int
    device_index: int
    group_ranks: tuple[tuple[str, int], ...]
    is_output_rank: bool

    def group_rank(self, group: str) -> int:
        try:
            return dict(self.group_ranks)[group]
        except KeyError as error:
            raise KeyError(f"unknown parallel group {group!r}.") from error


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Validated mapping of independent replicas onto resource slots."""

    parallel: ParallelConfig
    replica_count: int
    replica_world_size: int
    layout: RankLayout
    nodes: tuple[NodeResources, ...]
    workers: tuple[WorkerPlacement, ...]

    @classmethod
    def build(
        cls,
        parallel: ParallelConfig,
        replica_count: int,
        nodes: Sequence[NodeResources],
        *,
        output_rank: int = 0,
    ) -> "DeploymentPlan":
        if not isinstance(parallel, ParallelConfig):
            raise TypeError(
                "DeploymentPlan.build expects a ParallelConfig, "
                f"got {type(parallel).__name__}."
            )
        world = parallel.infer_replica_world_size()
        layout = build_rank_layout(parallel, world)
        if (
            not isinstance(replica_count, int)
            or isinstance(replica_count, bool)
            or replica_count < 1
        ):
            raise ValueError(
                f"replica_count must be a positive int, got {replica_count!r}."
            )
        if (
            not isinstance(output_rank, int)
            or isinstance(output_rank, bool)
            or not 0 <= output_rank < world
        ):
            raise ValueError(
                f"output_rank must be in [0, {world}), got {output_rank!r}."
            )
        normalized_nodes = tuple(nodes)
        if not all(isinstance(node, NodeResources) for node in normalized_nodes):
            raise TypeError("nodes must contain only NodeResources instances.")
        ordered_nodes = tuple(sorted(normalized_nodes, key=lambda node: node.rank))
        actual_ranks = tuple(node.rank for node in ordered_nodes)
        if actual_ranks != tuple(range(len(ordered_nodes))):
            raise ValueError(
                f"node ranks must be contiguous from zero, got {actual_ranks!r}."
            )
        slots = [
            (node.rank, device_index)
            for node in ordered_nodes
            for device_index in node.devices
        ]
        required = replica_count * world
        if len(slots) < required:
            raise ValueError(
                f"deployment needs {required} worker slots but only {len(slots)} "
                "devices were provided."
            )
        workers: list[WorkerPlacement] = []
        for worker_id, (node_rank, device_index) in enumerate(slots[:required]):
            replica_id, replica_rank = divmod(worker_id, world)
            workers.append(
                WorkerPlacement(
                    worker_id=worker_id,
                    replica_id=replica_id,
                    replica_rank=replica_rank,
                    node_rank=node_rank,
                    device_index=device_index,
                    group_ranks=memberships(layout, replica_rank),
                    is_output_rank=replica_rank == output_rank,
                )
            )
        for replica_id in range(replica_count):
            members = [w for w in workers if w.replica_id == replica_id]
            node_ranks = {w.node_rank for w in members}
            root_node = ordered_nodes[members[0].node_rank]
            if len(node_ranks) > 1 and root_node.address is None:
                # Every rank of the replica rendezvous at the root node; without
                # its address each node would wait on its own loopback — a hang,
                # not an error — so refuse the plan here.
                raise ValueError(
                    f"replica {replica_id} spans nodes {sorted(node_ranks)} but "
                    f"root node {root_node.rank} has no address; set "
                    "NodeResources.address for multi-node replicas."
                )
        return cls(
            parallel, replica_count, world, layout, ordered_nodes, tuple(workers)
        )

    @property
    def worker_count(self) -> int:
        return len(self.workers)

    def workers_for_replica(self, replica_id: int) -> tuple[WorkerPlacement, ...]:
        if not 0 <= replica_id < self.replica_count:
            raise ValueError(
                f"replica id {replica_id} is outside [0, {self.replica_count})."
            )
        return tuple(
            worker for worker in self.workers if worker.replica_id == replica_id
        )

    def workers_for_node(self, node_rank: int) -> tuple[WorkerPlacement, ...]:
        if not 0 <= node_rank < len(self.nodes):
            raise ValueError(f"node rank {node_rank} is outside this plan.")
        return tuple(worker for worker in self.workers if worker.node_rank == node_rank)


def _rank_environment() -> tuple[int, int] | None:
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None and world_size is None:
        return None
    if rank is None or world_size is None:
        raise ValueError("RANK and WORLD_SIZE must be provided together.")
    try:
        parsed = int(rank), int(world_size)
    except ValueError as error:
        raise ValueError("RANK and WORLD_SIZE must be integer values.") from error
    if parsed[0] < 0 or parsed[1] < 1 or parsed[0] >= parsed[1]:
        raise ValueError(f"invalid external rank environment rank/world={parsed!r}.")
    return parsed


def choose_executor_mode(replica_world_size: int, deployment: DeploymentConfig) -> str:
    """Resolve execution mode without importing worker process modules."""
    rank_environment = _rank_environment()
    if deployment.mode == "external":
        if rank_environment is None and replica_world_size > 1:
            raise ValueError(
                "external deployment requires launcher-provided RANK/WORLD_SIZE."
            )
        if rank_environment is not None and rank_environment[1] != replica_world_size:
            raise ValueError(
                f"launcher WORLD_SIZE={rank_environment[1]} does not match the "
                f"model replica world_size={replica_world_size}."
            )
        return "external"
    if rank_environment is not None and rank_environment[1] > 1:
        raise ValueError(
            "a rank launcher is already active; choose DeploymentConfig.external()."
        )
    if deployment.mode == "local" or deployment.devices:
        return "local"
    if deployment.mode != "auto":
        raise AssertionError(f"unhandled deployment mode {deployment.mode!r}.")
    if deployment.replica_count == 1 and replica_world_size == 1:
        return "inline"
    return "local"


def build_dispatcher(
    core_factory: Any,
    args: Any,
    replica_world_size: int,
    deployment: DeploymentConfig,
) -> tuple[str, RequestDispatcher]:
    """Build one dispatcher over the selected complete-replica executors.

    Lifecycle is intentionally owned by :class:`phyai.engine.Engine`; this
    function only chooses and wires the backend.
    """
    mode = choose_executor_mode(replica_world_size, deployment)
    if mode == "inline":
        executors = (InlineExecutor(core_factory, args, replica_id=0),)
    elif mode == "external":
        rank_environment = _rank_environment()
        rank = 0 if rank_environment is None else rank_environment[0]
        executors = (
            ExternalExecutor(
                core_factory,
                args,
                world_size=replica_world_size,
                rank=rank,
                output_rank=deployment.output_rank,
            ),
        )
    else:
        from phyai.server.local import build_local_executors

        executors = build_local_executors(
            args,
            replica_count=deployment.replica_count,
            devices=deployment.devices,
            process_config=deployment.process_config,
            output_rank=deployment.output_rank,
            core_factory=core_factory,
        )

    return mode, RequestDispatcher(executors)


__all__ = [
    "DeploymentConfig",
    "DeploymentMode",
    "DeploymentPlan",
    "NodeResources",
    "WorkerPlacement",
    "build_dispatcher",
    "choose_executor_mode",
    "normalize_device_indices",
]
