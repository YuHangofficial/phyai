"""CUDA integration tests for replica executors and named rank groups."""

from __future__ import annotations

import os
import socket
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

from phyai.server.dispatcher import RequestDispatcher
from phyai.server.executor import managed_executors
from phyai.server.deployment import DeploymentPlan, NodeResources
from phyai.server.worker_supervisor import (
    LifecycleState,
    WorkerFactorySpec,
    WorkerSupervisor,
    WorkerSupervisorConfig,
)


pytestmark = pytest.mark.gpu

_FACTORY = "gpu_worker_support:create_cuda_runtime"
_TEST_SUPPORT_DIR = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def _importable_worker_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(_TEST_SUPPORT_DIR)


def _require_cuda(device_count: int = 1) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < device_count:
        pytest.skip(f"requires at least {device_count} visible CUDA devices")


def _require_nccl() -> None:
    if not torch.distributed.is_nccl_available():
        pytest.skip("the installed PyTorch build has no NCCL support")


def _free_port(count: int = 1) -> int:
    """Return a base port such that ``base .. base + count - 1`` are all free.

    The supervisor gives replica ``i`` the rendezvous port ``base_port + i``,
    so every replica's port has to be checked, not just the first. The check
    is still a bind-then-release probe (racy by nature); a handful of attempts
    keeps the flake rate negligible on a shared box.
    """
    for _attempt in range(20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                base = int(sock.getsockname()[1])
            for offset in range(1, count):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("127.0.0.1", base + offset))
            return base
        except OSError:
            continue
    pytest.skip("cannot reserve a contiguous range of local rendezvous ports")


def _supervisor(
    sizes: ParallelConfig,
    *,
    replica_count: int = 1,
    mode: str,
    collective: bool = False,
) -> WorkerSupervisor:
    plan_world_size = sizes.infer_replica_world_size()
    devices = tuple(range(replica_count * plan_world_size))
    deployment = DeploymentPlan.build(
        sizes, replica_count, (NodeResources(0, devices),)
    )
    return WorkerSupervisor(
        deployment,
        WorkerFactorySpec(
            _FACTORY,
            {
                "mode": mode,
                "collective": collective,
                "parallel": sizes,
            },
        ),
        WorkerSupervisorConfig(
            base_port=_free_port(replica_count),
            startup_timeout_s=120,
            shutdown_timeout_s=10,
            execution_timeout_s=30,
        ),
    )


def _parallel(*, cfg: int = 1, tp: int = 1) -> ParallelConfig:
    return ParallelConfig(
        outer=OuterParallelConfig(cfg_size=cfg), dense=DenseParallelConfig(tp_size=tp)
    )


@pytest.mark.multi_gpu
def test_replica_dispatcher_routes_whole_requests_to_independent_gpus():
    _require_cuda(2)
    sizes = _parallel(tp=1)
    supervisor = _supervisor(sizes, replica_count=2, mode="replica")
    executors = managed_executors(
        supervisor,
        replica_count=2,
        rank_count=1,
    )
    dispatcher = RequestDispatcher(executors)
    try:
        dispatcher.setup()
        metadata = supervisor.worker_metadata
        assert set(metadata) == {0, 1}
        # Full visibility: every worker sees the parent's whole device set
        # and no per-worker CUDA_VISIBLE_DEVICES mask is injected...
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert {item["cuda_device_count"] for item in metadata.values()} == {
            torch.cuda.device_count()
        }
        assert {item["cuda_visible_devices"] for item in metadata.values()} == {
            parent_mask
        }
        # ...while each worker binds its own placement index.
        assert {item["cuda_current_device"] for item in metadata.values()} == {0, 1}

        first = dispatcher.submit({"value": 2.0, "sleep_s": 0.2})
        second = dispatcher.submit({"value": 3.0})
        first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)

        assert {first_result["replica_id"], second_result["replica_id"]} == {0, 1}
        assert first_result["device"] == f"cuda:{first_result['replica_id']}"
        assert second_result["device"] == f"cuda:{second_result['replica_id']}"
        assert first_result["value"] in (4.0, 5.0)
        assert second_result["value"] in (6.0, 7.0)
        assert supervisor.inflight == (0, 0)
        assert supervisor.state == LifecycleState.RUNNING
    finally:
        dispatcher.close()


def test_worker_supervisor_transports_local_cuda_tensor_with_torch_ipc():
    _require_cuda()
    supervisor = _supervisor(_parallel(tp=1), mode="cuda_tensor")
    result = None
    try:
        supervisor.start()
        result = supervisor.execute(0, {"value": 7.5})

        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"
        assert result.item() == pytest.approx(7.5)
    finally:
        del result
        supervisor.close()


@pytest.mark.multi_gpu
def test_cuda_results_arrive_as_ipc_views_on_each_workers_device():
    """Device numbering agrees across processes, so results stay on cuda:i."""
    _require_cuda(2)
    supervisor = _supervisor(_parallel(tp=1), replica_count=2, mode="cuda_tensor")
    results: dict[int, torch.Tensor] = {}
    try:
        supervisor.start()
        for replica_id, value in ((0, 1.25), (1, -4.5)):
            result = supervisor.execute(replica_id, {"value": value})
            assert isinstance(result, torch.Tensor)
            # The tensor is a CUDA-IPC view of the worker's memory on the
            # worker's own device — the parent sees the same index.
            assert result.device == torch.device("cuda", replica_id)
            assert result.item() == pytest.approx(value)
            # Parent-side arithmetic on the shared storage must work.
            assert (result * 2.0).item() == pytest.approx(value * 2.0)
            results[replica_id] = result
        merged = torch.cat(tuple(item.cpu() for item in results.values()))
        assert merged.tolist() == pytest.approx([1.25, -4.5])
    finally:
        results.clear()
        supervisor.close()


@pytest.mark.multi_gpu
def test_tensor_group_reduce_gather_scatter_and_broadcast():
    _require_cuda(2)
    _require_nccl()
    supervisor = _supervisor(_parallel(tp=2), mode="tp", collective=True)
    try:
        supervisor.start()
        assert all(
            "pynccl" in metadata["collective_backends"]
            for metadata in supervisor.worker_metadata.values()
        )
        # Placement was probed at init: every rank agrees, both ranks share
        # this host, and NVLink matches what NVML says for devices 0 and 1
        # (the workers bind device indices 0 and 1 under the same visibility).
        from phyai.utils import nvml

        topologies = [m["topology"] for m in supervisor.worker_metadata.values()]
        assert all(t == topologies[0] for t in topologies)
        assert topologies[0]["is_single_node"]
        assert (topologies[0]["n_nodes"], topologies[0]["n_gpus_per_node"]) == (1, 2)
        try:
            expected_nvlink = nvml.nvlink_fully_connected(
                [nvml.physical_device_index(0), nvml.physical_device_index(1)]
            )
        except Exception:  # no NVML in this container: the probe degraded too
            expected_nvlink = None
        if expected_nvlink is not None:
            assert topologies[0]["is_full_nvlink"] == expected_nvlink
        result = supervisor.execute(0, {})
        assert result["rank"] == 0
        assert result["tensor_rank"] == 0
        assert result["reduced"] == pytest.approx(3.0)
        assert result["gathered"] == pytest.approx([1.0, 2.0])
        assert result["scattered"] == pytest.approx(3.0)
        assert result["broadcast"] == pytest.approx(42.0)
    finally:
        supervisor.close()


@pytest.mark.multi_gpu
def test_cfg_group_gathers_cond_uncond_in_stable_order():
    _require_cuda(2)
    _require_nccl()
    supervisor = _supervisor(_parallel(cfg=2), mode="cfg", collective=True)
    try:
        supervisor.start()
        result = supervisor.execute(0, {"guidance_scale": 3.0})
        assert result["rank"] == 0
        assert result["cfg_rank"] == 0
        assert result["pair"] == pytest.approx([10.0, 20.0])
        assert result["guided"] == pytest.approx(-10.0)
    finally:
        supervisor.close()


@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "mode", ("domains", "sequence_parallel", "transformer_sequence_parallel")
)
def test_domain_groups_and_sequence_parallel_on_cuda(mode):
    _require_cuda(2)
    _require_nccl()
    config = ParallelConfig(
        dense=DenseParallelConfig(tp_size=2, sequence_parallel=mode != "domains"),
        attention=AttentionParallelConfig(dp_size=2 if mode == "domains" else 1),
        moe=MoeParallelConfig(ep_size=2 if mode == "domains" else 1),
    )
    supervisor = _supervisor(config, mode=mode, collective=True)
    try:
        supervisor.start()
        result = supervisor.execute(0, {})
        expected = (
            {"rank": 0, "world": 2}
            if mode == "domains"
            else {
                "rank": 0,
                "shape": (2, 4, 32)
                if mode == "transformer_sequence_parallel"
                else (2, 16),
            }
        )
        assert result == expected
    finally:
        supervisor.close()


@pytest.mark.multi_gpu
def test_cfg_and_tensor_groups_compose_in_one_four_gpu_replica():
    _require_cuda(4)
    _require_nccl()
    supervisor = _supervisor(
        _parallel(cfg=2, tp=2),
        mode="cfg_tensor",
        collective=True,
    )
    try:
        supervisor.start()
        result = supervisor.execute(0, {})
        assert result["rank"] == 0
        assert result["cfg_rank"] == 0
        assert result["tensor_rank"] == 0
        assert result["tensor_sum"] == pytest.approx(3.0)
        assert result["pair"] == pytest.approx([10.0, 20.0])
    finally:
        supervisor.close()


_GRAPH_REDUCED = [[3.0, 5.0], [7.0, 9.0]]  # rank sums of [[r+1, r+2], [r+3, r+4]]
_GRAPH_REDUCED_VIEW = [[3.0], [7.0]]  # rank sums of the first column only
_GRAPH_GATHERED = [[1.0, 2.0, 2.0, 3.0], [3.0, 4.0, 4.0, 5.0]]  # last-dim concat


@pytest.mark.multi_gpu
def test_graph_captured_pynccl_collectives_match_eager():
    """pynccl drives captured tp collectives; dim=-1 all_gather must be exact."""
    _require_cuda(2)
    _require_nccl()
    supervisor = _supervisor(_parallel(tp=2), mode="graph", collective=True)
    try:
        supervisor.start()
        assert all(
            "pynccl" in metadata["collective_backends"]
            for metadata in supervisor.worker_metadata.values()
        )
        result = supervisor.execute(0, {})
        assert result["captured_backend"] == "pynccl"
        assert result["captured_reduced"] == _GRAPH_REDUCED
        assert result["eager_reduced"] == _GRAPH_REDUCED
        # Non-contiguous input reduced through pynccl's raw-pointer path.
        assert result["captured_reduced_view"] == _GRAPH_REDUCED_VIEW
        assert result["eager_reduced_view"] == _GRAPH_REDUCED_VIEW
        assert result["captured_gathered"] == _GRAPH_GATHERED
        assert result["eager_gathered"] == _GRAPH_GATHERED
    finally:
        supervisor.close()


@pytest.mark.multi_gpu
def test_graph_captured_world_collective_falls_back_to_torch_nccl():
    """This runtime attaches pynccl to ``dense_tp`` only (see
    ``gpu_worker_support``), so a captured ``world`` collective must fall
    through to NcclBackend instead of failing inside pynccl at execute time.
    Engine defaults attach every group, ``world`` included."""
    _require_cuda(2)
    _require_nccl()
    supervisor = _supervisor(_parallel(tp=2), mode="graph_world", collective=True)
    try:
        supervisor.start()
        result = supervisor.execute(0, {})
        assert result["captured_backend"] == "nccl"
        assert result["captured_reduced"] == _GRAPH_REDUCED
        assert result["eager_reduced"] == _GRAPH_REDUCED
        assert result["captured_reduced_view"] == _GRAPH_REDUCED_VIEW
    finally:
        supervisor.close()


@pytest.mark.multi_gpu
def test_two_replicas_of_two_ranks_bind_and_communicate_independently():
    """replica_count=2 x tp=2 under full visibility: every worker binds its own
    placement index and pynccl attaches to that same GPU (not rank % count)."""
    _require_cuda(4)
    _require_nccl()
    supervisor = _supervisor(
        _parallel(tp=2), replica_count=2, mode="graph", collective=True
    )
    executors = managed_executors(supervisor, replica_count=2, rank_count=2)
    dispatcher = RequestDispatcher(executors)
    try:
        dispatcher.setup()
        metadata = supervisor.worker_metadata
        assert set(metadata) == {0, 1, 2, 3}
        for worker_id, item in metadata.items():
            # Plan: worker w -> replica w // 2, device index w.
            assert item["device_index"] == worker_id
            assert item["cuda_current_device"] == item["device_index"]
            assert item["pynccl_attach_devices"], "pynccl attached nothing"
            assert set(item["pynccl_attach_devices"].values()) == {
                f"cuda:{item['device_index']}"
            }

        for replica_id in (0, 1):
            result = supervisor.execute(replica_id, {})
            assert result["captured_backend"] == "pynccl"
            assert result["captured_reduced"] == _GRAPH_REDUCED
            assert result["captured_gathered"] == _GRAPH_GATHERED
        assert supervisor.inflight == (0, 0)
        assert supervisor.state == LifecycleState.RUNNING
    finally:
        dispatcher.close()
