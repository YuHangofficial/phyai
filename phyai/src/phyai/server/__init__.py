"""Managed local worker execution primitives.

The package is intentionally lazy: importing ``phyai.server`` must stay cheap
for single-card runs, so submodules are pulled in only when one of their
symbols is requested.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


_LAZY: dict[str, str] = {
    "MultiprocessExecutor": "phyai.server.executor",
    "EngineWorkerConfig": "phyai.server.engine_worker",
    "engine_worker_factory": "phyai.server.engine_worker",
    "DeploymentPlan": "phyai.server.deployment",
    "DeploymentConfig": "phyai.server.deployment",
    "NodeResources": "phyai.server.deployment",
    "WorkerPlacement": "phyai.server.deployment",
    "normalize_device_indices": "phyai.server.deployment",
    "LifecycleState": "phyai.server.lifecycle",
    "WorkerFactorySpec": "phyai.server.worker_supervisor",
    "WorkerRuntime": "phyai.server.worker_supervisor",
    "WorkerSupervisor": "phyai.server.worker_supervisor",
    "WorkerSupervisorConfig": "phyai.server.worker_supervisor",
}

if TYPE_CHECKING:
    from phyai.server.executor import MultiprocessExecutor
    from phyai.server.deployment import (
        DeploymentConfig,
        DeploymentPlan,
        NodeResources,
        WorkerPlacement,
        normalize_device_indices,
    )
    from phyai.server.engine_worker import EngineWorkerConfig, engine_worker_factory
    from phyai.server.lifecycle import LifecycleState
    from phyai.server.worker_supervisor import (
        WorkerFactorySpec,
        WorkerRuntime,
        WorkerSupervisor,
        WorkerSupervisorConfig,
    )


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "DeploymentPlan",
    "DeploymentConfig",
    "EngineWorkerConfig",
    "LifecycleState",
    "MultiprocessExecutor",
    "NodeResources",
    "WorkerFactorySpec",
    "WorkerPlacement",
    "WorkerRuntime",
    "WorkerSupervisor",
    "WorkerSupervisorConfig",
    "engine_worker_factory",
    "normalize_device_indices",
]
