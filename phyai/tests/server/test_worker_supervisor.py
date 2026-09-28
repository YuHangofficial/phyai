"""Integration tests for spawned worker supervision and fail-fast behavior."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

import pytest
import torch
from phyai.engine_config import (
    ParallelConfig,
    OuterParallelConfig,
    DenseParallelConfig,
    AttentionParallelConfig,
    MoeParallelConfig,
)

from phyai.server.executor import EngineUnavailableError
from phyai.server.deployment import DeploymentPlan, NodeResources
from phyai.server.worker_supervisor import (
    LifecycleState,
    WorkerFactorySpec,
    WorkerSupervisor,
    WorkerSupervisorConfig,
)


_FACTORY = "worker_support:create_fake_runtime"
_GLOO_FACTORY = "worker_support:create_gloo_runtime"
_TEST_SUPPORT_DIR = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def _importable_worker_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(_TEST_SUPPORT_DIR)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _parallel(*, cfg_size: int = 1, tensor_size: int = 1) -> ParallelConfig:
    return ParallelConfig(
        outer=OuterParallelConfig(cfg_size=cfg_size),
        dense=DenseParallelConfig(tp_size=tensor_size),
    )


def _supervisor(
    *,
    replica_count: int = 1,
    tensor_size: int = 1,
    factory_args=None,
    execution_timeout_s: float | None = None,
) -> WorkerSupervisor:
    sizes = _parallel(tensor_size=tensor_size)
    world = sizes.infer_replica_world_size()
    devices = tuple(range(replica_count * world))
    deployment = DeploymentPlan.build(
        sizes, replica_count, (NodeResources(0, devices),)
    )
    return WorkerSupervisor(
        deployment,
        WorkerFactorySpec(_FACTORY, factory_args or {}),
        WorkerSupervisorConfig(
            startup_timeout_s=30,
            shutdown_timeout_s=2,
            execution_timeout_s=execution_timeout_s,
        ),
    )


def test_spawned_workers_inherit_visibility_and_bind_by_index():
    supervisor = _supervisor(replica_count=2)
    try:
        supervisor.start()

        assert supervisor.state == LifecycleState.RUNNING
        first = supervisor.worker_metadata[0]
        second = supervisor.worker_metadata[1]
        # Full visibility: workers see exactly what the parent sees — no
        # per-worker CUDA_VISIBLE_DEVICES mask is injected.
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert first["cuda_visible_devices"] == parent_mask
        assert second["cuda_visible_devices"] == parent_mask
        # Index binding: each worker owns its placement device index, which
        # LOCAL_RANK mirrors as a backstop for bare-"cuda" resolves.
        assert (first["device_index"], first["local_rank"]) == (0, "0")
        assert (second["device_index"], second["local_rank"]) == (1, "1")
        assert first["rank"] == "0"
        assert first["world_size"] == "1"
        assert supervisor.execute(0, "left") == {
            "worker_id": 0,
            "payload": "left",
        }
        assert supervisor.execute(1, "right") == {
            "worker_id": 1,
            "payload": "right",
        }
    finally:
        supervisor.close()
        supervisor.close()

    assert supervisor.state == LifecycleState.STOPPED


def test_extra_env_cannot_override_placement_or_rendezvous_variables():
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        WorkerSupervisorConfig(extra_env=(("CUDA_VISIBLE_DEVICES", "3"),))
    with pytest.raises(ValueError, match="MASTER_PORT"):
        WorkerSupervisorConfig(extra_env=(("MASTER_PORT", "1"),))


def test_model_parallel_environment_is_scoped_to_one_replica():
    sizes = _parallel(cfg_size=2, tensor_size=2)
    deployment = DeploymentPlan.build(
        sizes,
        2,
        (NodeResources(0, tuple(range(8))),),
    )
    supervisor = WorkerSupervisor(deployment, WorkerFactorySpec(_FACTORY, {}))

    first_placement = deployment.workers_for_replica(0)[3]
    second_placement = deployment.workers_for_replica(1)[0]
    first = supervisor._worker_environment(first_placement)
    second = supervisor._worker_environment(second_placement)

    assert "CUDA_VISIBLE_DEVICES" not in first
    assert "GROUP_RANK" not in first
    # A local plan carries no node address: rendezvous at the supervisor's
    # master_addr default.
    assert first["MASTER_ADDR"] == "127.0.0.1"
    assert first["RANK"] == "3"
    assert first["WORLD_SIZE"] == "4"
    assert first["LOCAL_RANK"] == "3"
    assert "LOCAL_WORLD_SIZE" not in first  # cannot pair with a device-index LOCAL_RANK
    assert first_placement.group_rank("cfg") == 1
    assert first_placement.group_rank("dense_tp") == 1
    assert second["RANK"] == "0"
    assert second["LOCAL_RANK"] == "4"
    assert second["MASTER_PORT"] == "29501"


def test_cpu_tensor_payload_round_trips_through_spawned_worker():
    supervisor = _supervisor()
    payload = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    try:
        supervisor.start()
        result = supervisor.execute(0, payload)

        assert result["worker_id"] == 0
        assert torch.equal(result["payload"], payload)
        assert result["payload"] is not payload
        assert payload.is_shared()
    finally:
        supervisor.close()


def test_spawned_model_parallel_group_completes_real_gloo_collective():
    sizes = _parallel(cfg_size=2, tensor_size=2)
    deployment = DeploymentPlan.build(
        sizes,
        1,
        (NodeResources(0, (0, 1, 2, 3)),),
    )
    supervisor = WorkerSupervisor(
        deployment,
        WorkerFactorySpec(
            _GLOO_FACTORY,
            {"init_mesh": True, "parallel": sizes},
        ),
        WorkerSupervisorConfig(
            base_port=_free_port(),
            startup_timeout_s=30,
            shutdown_timeout_s=2,
        ),
    )
    try:
        supervisor.start()
        # cfg and tp have size 2, the implicit world axis spans all 4 ranks;
        # size-1 axes are skipped. Warmed on CPU tensors through gloo even
        # though this host has CUDA (see collective_device).
        assert set(supervisor.worker_metadata[0]["warmed"]) == {
            "world",
            "cfg",
            "dense_tp",
            "attention_tp",
            "moe_tp",
            "moe_tp_ep",
        }
        assert supervisor.execute(0, 2) == {
            "rank": 0,
            "sum": 14.0,
            "cfg_source": 0.0,
        }
    finally:
        supervisor.close()


def test_overlapping_domains_reinitialize_without_crossing_replicas():
    replica_count, world = 2, 2
    parallel = ParallelConfig(
        dense=DenseParallelConfig(tp_size=world),
        attention=AttentionParallelConfig(dp_size=2, decode_cp_size=world // 2),
        moe=MoeParallelConfig(ep_size=2),
    )
    plan = DeploymentPlan.build(
        parallel,
        replica_count,
        (NodeResources(0, tuple(range(replica_count * world))),),
    )
    supervisor = WorkerSupervisor(
        plan,
        WorkerFactorySpec(
            _GLOO_FACTORY,
            {"init_mesh": True, "parallel": parallel, "check_domains": True},
        ),
        WorkerSupervisorConfig(
            base_port=_free_port(),
            startup_timeout_s=60,
            execution_timeout_s=30,
            shutdown_timeout_s=3,
        ),
    )
    try:
        supervisor.start()
        for replica_id in range(replica_count):
            assert supervisor.execute(replica_id, None) == {
                "replica_id": replica_id,
                "world": world,
            }
    finally:
        supervisor.close()


def test_startup_failure_terminates_partial_worker_group():
    supervisor = _supervisor(tensor_size=2, factory_args={"startup_fail_worker": 1})

    # Every backend-death path, including startup, surfaces as the routing
    # layer's "take it out of rotation" type.
    with pytest.raises(EngineUnavailableError, match="requested startup failure"):
        supervisor.start()
    with pytest.raises(EngineUnavailableError, match="state"):
        supervisor.start()

    assert supervisor.state == LifecycleState.FAILED
    assert supervisor.failure is not None
    assert all(not handle.process.is_alive() for handle in supervisor._handles.values())
    supervisor.close()


def test_worker_crash_fails_request_and_entire_supervisor():
    supervisor = _supervisor(tensor_size=2)
    try:
        supervisor.start()
        future = supervisor.submit(0, {"exit_worker": 1})

        with pytest.raises(EngineUnavailableError, match="worker 1"):
            future.result(timeout=10)
        assert supervisor.state == LifecycleState.FAILED
    finally:
        supervisor.close()


def test_result_serialization_failure_fails_entire_supervisor():
    supervisor = _supervisor()
    try:
        supervisor.start()
        future = supervisor.submit(0, {"unpicklable_output": 0})

        with pytest.raises(RuntimeError, match="worker 0 failed during execute"):
            future.result(timeout=10)
        assert supervisor.state == LifecycleState.FAILED
    finally:
        supervisor.close()


def test_execution_timeout_fails_collective_worker_group():
    supervisor = _supervisor(tensor_size=2, execution_timeout_s=0.1)
    try:
        supervisor.start()
        future = supervisor.submit(0, {"sleep_by_worker": {1: 2.0}})

        with pytest.raises(EngineUnavailableError, match="execution timeout"):
            future.result(timeout=10)
        assert supervisor.state == LifecycleState.FAILED
    finally:
        supervisor.close()


def test_close_during_request_is_idempotent_and_not_a_worker_failure():
    supervisor = _supervisor()
    supervisor.start()
    future = supervisor.submit(0, {"sleep_by_worker": {0: 0.1}})

    closers = [threading.Thread(target=supervisor.close) for _ in range(2)]
    for closer in closers:
        closer.start()
    for closer in closers:
        closer.join(timeout=5)

    with pytest.raises(RuntimeError, match="closed before request completion"):
        future.result(timeout=10)
    assert all(not closer.is_alive() for closer in closers)
    assert supervisor.state == LifecycleState.STOPPED
    assert supervisor.failure is None


def test_shutdown_failure_is_reflected_in_supervisor_state():
    supervisor = _supervisor(factory_args={"shutdown_fail_worker": 0})
    supervisor.start()

    supervisor.close()

    assert supervisor.state == LifecycleState.FAILED
    assert supervisor.failure is not None
    assert "exited with code" in str(supervisor.failure)


def test_worker_failure_fails_the_whole_supervisor_group():
    supervisor = _supervisor(replica_count=2)
    try:
        supervisor.start()
        failed = supervisor.submit(0, {"fail_worker": 0})

        with pytest.raises(EngineUnavailableError, match="requested execute failure"):
            failed.result(timeout=10)
        # Fail-fast: the sibling replica is disabled with the whole group.
        assert supervisor.state == LifecycleState.FAILED
        assert supervisor.healthy == (False, False)
        with pytest.raises(EngineUnavailableError, match="state"):
            supervisor.submit(1, "healthy")
    finally:
        supervisor.close()

    assert supervisor.state == LifecycleState.FAILED
    assert supervisor.failure is not None


def test_replica_spanning_two_nodes_gets_one_rendezvous_and_per_node_placement():
    # Multi-node plans are not reachable from DeploymentConfig today; this pins
    # the supervisor-level contract the gateway seam relies on, without
    # spawning anything.
    sizes = _parallel(tensor_size=4)
    plan = DeploymentPlan.build(
        sizes,
        1,
        (NodeResources(0, (0, 1), "node-a"), NodeResources(1, (0, 1), "node-b")),
    )
    node0 = WorkerSupervisor(plan, WorkerFactorySpec(_FACTORY, {}))
    node1 = WorkerSupervisor(
        plan, WorkerFactorySpec(_FACTORY, {}), WorkerSupervisorConfig(node_rank=1)
    )

    assert [w.worker_id for w in node0._placements] == [0, 1]
    assert [w.worker_id for w in node1._placements] == [2, 3]

    remote = node1._worker_environment(node1._placements[1])
    assert remote["MASTER_ADDR"] == "node-a"  # rendezvous at the replica's root
    assert remote["MASTER_PORT"] == "29500"
    assert (remote["RANK"], remote["WORLD_SIZE"]) == ("3", "4")
    assert remote["LOCAL_RANK"] == "1"
