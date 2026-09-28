"""Construct managed local replica executors from model and deployment plans."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from phyai.server.executor import ReplicaExecutor, managed_executors
from phyai.server.engine_worker import engine_worker_factory
from phyai.server.deployment import (
    DeploymentPlan,
    NodeResources,
    normalize_device_indices,
)
from phyai.server.worker_supervisor import WorkerSupervisor, WorkerSupervisorConfig


def _engine_config(engine_args: Any) -> Any:
    config = getattr(engine_args, "config", None)
    if config is None:
        raise ValueError("managed execution requires an explicit EngineConfig.")
    return config


def _resolve_local_devices(
    devices: Sequence[int | str] | None,
    required: int,
    device_type: str,
) -> tuple[int, ...]:
    if devices:
        selected = normalize_device_indices(devices, owner="devices")
    elif device_type == "cuda":
        # Workers inherit this process's device visibility, so the pool is
        # simply every visible index. device_count() resolves through NVML,
        # honours a shell-level CUDA_VISIBLE_DEVICES mask, and does not
        # initialize a CUDA context in the launching process.
        selected = tuple(range(torch.cuda.device_count()))
    else:
        selected = tuple(range(required))
    if len(selected) < required:
        raise ValueError(
            f"deployment needs {required} worker slots but only "
            f"{len(selected)} devices are visible: {selected!r}."
        )
    return tuple(selected[:required])


def build_local_executors(
    engine_args: Any,
    *,
    replica_count: int,
    devices: Sequence[int | str] = (),
    process_config: WorkerSupervisorConfig | None = None,
    output_rank: int = 0,
    core_factory: Callable[..., Any] | None = None,
) -> tuple[ReplicaExecutor, ...]:
    """Build local single-replica executor views without starting workers.

    ``core_factory`` (default ``EngineCore``) is instantiated inside each
    spawned worker, so it must be importable by reference.
    """
    if process_config is not None and not isinstance(
        process_config, WorkerSupervisorConfig
    ):
        raise TypeError(
            "local process_config must be a WorkerSupervisorConfig, got "
            f"{type(process_config).__name__}."
        )
    config = process_config or WorkerSupervisorConfig()
    if config.node_rank != 0:
        raise ValueError("local deployment requires process_config.node_rank=0.")
    engine_config = _engine_config(engine_args)
    device_type = torch.device(engine_config.device.target).type
    parallel = engine_config.parallel
    replica_world_size = parallel.infer_replica_world_size()
    required = replica_count * replica_world_size
    selected = _resolve_local_devices(devices, required, device_type)
    plan = DeploymentPlan.build(
        parallel,
        replica_count,
        (NodeResources(0, selected),),
        output_rank=output_rank,
    )
    supervisor = WorkerSupervisor(
        plan,
        engine_worker_factory(engine_args, core_factory=core_factory),
        config,
    )
    return managed_executors(
        supervisor,
        replica_count=replica_count,
        rank_count=replica_world_size,
    )


__all__ = ["build_local_executors"]
