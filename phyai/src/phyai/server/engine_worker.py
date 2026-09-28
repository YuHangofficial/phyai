"""Worker factory that constructs one in-process Engine replica rank."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable

from phyai.server.deployment import WorkerPlacement
from phyai.server.worker_supervisor import WorkerFactorySpec


ENGINE_WORKER_FACTORY = "phyai.server.engine_worker:create_engine_worker_runtime"


def worker_device_target(target: str, device_index: int) -> str:
    """Rebind a CUDA target to this worker's placement-assigned device index.

    Managed placement owns device assignment: a bare ``"cuda"`` and any
    explicit ``"cuda:N"`` both land on ``cuda:{device_index}``. Workers keep
    the parent's full device visibility, so the index names the same GPU in
    every process. Non-CUDA targets pass through untouched.
    """
    import torch  # noqa: PLC0415

    if torch.device(target).type != "cuda":
        return target
    return f"cuda:{device_index}"


@dataclass(frozen=True, slots=True)
class EngineWorkerConfig:
    """Engine arguments for one spawned replica rank.

    ``core_factory`` builds the in-process core from the (device-rebound)
    engine args; ``None`` means ``phyai.engine.EngineCore``. It crosses the
    ``spawn`` boundary by reference, so it must be an importable module-level
    class or function.
    """

    engine_args: Any
    core_factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        config = getattr(self.engine_args, "config", None)
        if config is None:
            raise ValueError(
                "multi-process Engine workers require an explicit EngineConfig."
            )


class EngineWorkerRuntime:
    def __init__(
        self, placement: WorkerPlacement, worker_config: EngineWorkerConfig
    ) -> None:
        import torch  # noqa: PLC0415

        core_factory = worker_config.core_factory
        if core_factory is None:
            from phyai.engine import EngineCore  # noqa: PLC0415

            core_factory = EngineCore
        outer_args = worker_config.engine_args
        outer_config = outer_args.config
        target = worker_device_target(
            outer_config.device.target, placement.device_index
        )
        self._device_is_cuda = torch.device(target).type == "cuda"
        if self._device_is_cuda:
            # The spawned worker inherits the parent's environment, and
            # EngineConfig.from_env re-reads PHYAI_DEVICE inside EngineCore;
            # pin it so an inherited value cannot override the placement.
            os.environ["PHYAI_DEVICE"] = target
        engine_config = replace(
            outer_config,
            device=replace(outer_config.device, target=target),
        )
        # Workers are already placed and ranked by WorkerSupervisor. Build
        # the in-process core directly so the public Engine facade cannot
        # recursively launch another worker group.
        self._engine = core_factory(replace(outer_args, config=engine_config))
        self._placement = placement
        self._torch = torch

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self._engine.args.plugin,
            "worker_id": self._placement.worker_id,
            "replica_id": self._placement.replica_id,
            "replica_rank": self._placement.replica_rank,
            "groups": dict(self._placement.group_ranks),
            "replica_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "device": self._engine.config.device.target,
        }

    def execute(self, payload: Any) -> Any:
        result = self._engine.step(payload)
        if not self._placement.is_output_rank:
            return None
        if self._device_is_cuda:
            # The result crosses the process boundary as CUDA-IPC views of
            # this worker's memory; the parent must never observe writes
            # that are still in flight on this device.
            self._torch.cuda.synchronize()
        return result

    def close(self) -> None:
        self._engine.close()


def create_engine_worker_runtime(
    placement: WorkerPlacement, args: EngineWorkerConfig
) -> EngineWorkerRuntime:
    """Construct an Engine only after the spawned worker owns its GPU."""
    if not isinstance(args, EngineWorkerConfig):
        raise TypeError(f"expected EngineWorkerConfig, got {type(args).__name__}.")
    return EngineWorkerRuntime(placement, args)


def engine_worker_factory(
    engine_args: Any, *, core_factory: Callable[..., Any] | None = None
) -> WorkerFactorySpec:
    """Build the importable factory specification used by process supervisors."""
    return WorkerFactorySpec(
        factory=ENGINE_WORKER_FACTORY,
        args=EngineWorkerConfig(engine_args, core_factory),
    )


__all__ = [
    "ENGINE_WORKER_FACTORY",
    "EngineWorkerConfig",
    "EngineWorkerRuntime",
    "create_engine_worker_runtime",
    "engine_worker_factory",
    "worker_device_target",
]
