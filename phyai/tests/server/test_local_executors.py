"""Tests for constructing managed local replica executors."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
import torch

from phyai.engine import EngineArgs
from phyai.engine_config import (
    DeviceConfig,
    EngineConfig,
    ParallelConfig,
    OuterParallelConfig,
    DenseParallelConfig,
    RuntimeConfig,
)
from phyai.server.executor import MultiprocessExecutor, managed_executors
from phyai.server.lifecycle import EngineUnavailableError, LifecycleState
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.server.local import build_local_executors
from phyai.server.worker_supervisor import WorkerSupervisor


_TEST_SUPPORT_DIR = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def _importable_worker_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(_TEST_SUPPORT_DIR)


def _args(parallel: ParallelConfig | None = None) -> EngineArgs:
    return EngineArgs(
        plugin="pi05",
        plugin_args=PI05Args(),
        config=EngineConfig(
            device=DeviceConfig(target="cpu", params_dtype=torch.float32),
            parallel=parallel or ParallelConfig(),
            runtime=RuntimeConfig(use_cuda_graph=False),
        ),
    )


def test_local_builder_separates_replicas_from_model_axes():
    parallel = ParallelConfig(
        outer=OuterParallelConfig(cfg_size=2), dense=DenseParallelConfig(tp_size=2)
    )
    executors = build_local_executors(
        _args(parallel),
        replica_count=2,
        devices=tuple(range(8)),
    )
    try:
        assert len(executors) == 2
        assert all(isinstance(item, MultiprocessExecutor) for item in executors)
        supervisor = executors[0]._shared.manager
        assert isinstance(supervisor, WorkerSupervisor)
        assert supervisor.plan.replica_count == 2
        assert supervisor.plan.parallel == parallel
        assert supervisor.plan.replica_world_size == 4
        assert [
            worker.replica_rank for worker in supervisor.plan.workers_for_replica(1)
        ] == [0, 1, 2, 3]
    finally:
        for executor in executors:
            executor.close()


def test_local_builder_rejects_insufficient_devices():
    with pytest.raises(ValueError, match="needs 4 worker slots"):
        build_local_executors(
            _args(ParallelConfig(dense=DenseParallelConfig(tp_size=2))),
            replica_count=2,
            devices=(0, 1, 2),
        )


class _FakeSupervisor:
    """Minimal SupervisorProtocol implementation for executor-view tests."""

    def __init__(self, replica_count: int) -> None:
        self.state = LifecycleState.CREATED
        self.healthy: tuple[bool, ...] = (True,) * replica_count
        self.inflight: tuple[int, ...] = (0,) * replica_count

    def start(self) -> None:
        self.state = LifecycleState.RUNNING

    def submit(self, replica_id: int, payload: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result((replica_id, payload))
        return future

    def close(self) -> None:
        self.state = LifecycleState.STOPPED


def test_multiprocess_executor_view_health_follows_supervisor_state():
    fake = _FakeSupervisor(replica_count=2)
    first, second = managed_executors(fake, replica_count=2, rank_count=1)

    # Not started yet: a created / starting supervisor is a valid routing target.
    assert first.healthy and second.healthy
    fake.state = LifecycleState.STARTING
    assert first.healthy and second.healthy

    fake.state = LifecycleState.RUNNING
    fake.healthy = (True, False)
    assert first.healthy
    assert not second.healthy

    for state in (
        LifecycleState.STOPPING,
        LifecycleState.STOPPED,
        LifecycleState.FAILED,
    ):
        fake.state = state
        assert not first.healthy and not second.healthy

    # A supervisor reporting fewer replicas than the views expect degrades to
    # "unhealthy / idle", never IndexError.
    fake.state = LifecycleState.RUNNING
    fake.healthy = (True,)
    fake.inflight = (3,)
    assert first.healthy and first.inflight == 3
    assert not second.healthy and second.inflight == 0

    # A closed view is unavailable on its own, whatever the supervisor says.
    first.close()
    assert not first.healthy and first.inflight == 0
    with pytest.raises(EngineUnavailableError):
        first.submit("request").result(timeout=1)


def test_local_executors_build_the_injected_core_inside_workers():
    from worker_support import FakeCore

    (executor,) = build_local_executors(
        _args(), replica_count=1, devices=(0,), core_factory=FakeCore
    )
    try:
        assert executor.step({"x": 1}) == {"echo": {"x": 1}, "plugin": "pi05"}
        supervisor = executor._shared.manager
        assert supervisor.worker_metadata[0]["plugin"] == "pi05"
        assert supervisor.worker_metadata[0]["device"] == "cpu"
    finally:
        executor.close()
