"""Unit tests for phyai.parallel.init's collective-device resolution.

No process group or NCCL library is touched: ``PyNCCLBackend.attach`` is
stubbed so the device that reaches it can be asserted on any machine.
"""

from __future__ import annotations

import pytest
import torch

import phyai.parallel as P
from phyai.parallel.backends.pynccl import PyNCCLBackend
from phyai.parallel.mesh import Mesh
from phyai.parallel.layout import build_rank_layout
from phyai.engine_config import ParallelConfig


@pytest.fixture
def current_device_is_two(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)


def test_collective_device_is_the_explicit_index_or_the_pinned_device(
    current_device_is_two,
):
    # Full visibility: the caller pinned its placement index before init, so
    # None and a bare "cuda" land there, never on rank % device_count.
    assert P.resolve_collective_device(None) == torch.device("cuda", 2)
    assert P.resolve_collective_device("cuda") == torch.device("cuda", 2)
    assert P.resolve_collective_device("cuda:3") == torch.device("cuda", 3)
    with pytest.raises(ValueError, match="CUDA device"):
        P.resolve_collective_device("cpu")


@pytest.mark.parametrize(
    ("world_backend", "requested", "device", "expected"),
    (
        # An externally created group with torch's composite default backend
        # reports "undefined"; auto-selection must derive from the device.
        ("undefined", None, "cuda", "nccl"),
        ("undefined", "auto", "cpu", "gloo"),
        ("undefined", None, torch.device("cuda", 1), "nccl"),
        ("gloo", None, "cuda", "gloo"),  # a plain world backend is reused
        ("nccl", "gloo", "cuda", "gloo"),  # an explicit request always wins
    ),
)
def test_resolve_backend_prefers_request_then_world_then_device(
    monkeypatch, world_backend, requested, device, expected
):
    monkeypatch.setattr(P.torch_dist, "get_backend", lambda: world_backend)
    assert P._resolve_backend(requested, device) == expected


class _FakeComm:
    def __init__(self):
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


def test_pynccl_close_destroys_every_communicator():
    backend = PyNCCLBackend()
    comms = {("model", "dense_tp"): _FakeComm(), ("model", "cfg"): _FakeComm()}
    comms[("model", "attention_tp")] = comms[("model", "dense_tp")]
    backend._comms.update(comms)
    backend.close()

    assert all(comm.destroyed == 1 for comm in comms.values())
    assert backend.attached() == {}


def test_registry_failure_closes_partially_attached_communicators(monkeypatch):
    comm = _FakeComm()

    def fail_attach(self, mesh, groups, *, device):
        self._comms[(mesh.name, "dense_tp")] = comm
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(PyNCCLBackend, "attach", fail_attach)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        P._make_registry(
            Mesh(build_rank_layout(ParallelConfig(), 2)),
            backend="nccl",
            device="cuda:0",
            enable_pynccl=True,
            pynccl_groups=None,
            pynccl_library_path=None,
        )
    assert comm.destroyed == 1


def test_shutdown_clears_process_level_state_and_is_idempotent():
    P.init(ParallelConfig())
    assert P.default_mesh().name == "model"
    P.get_dispatcher()

    P.shutdown()
    P.shutdown()

    with pytest.raises(RuntimeError, match="init"):
        P.get_dispatcher()
    with pytest.raises(KeyError, match="Unknown mesh"):
        P.default_mesh()


def test_all_to_all_requires_both_split_lists_or_neither():
    P.init(ParallelConfig())
    try:
        x = torch.zeros(4, 2)
        with pytest.raises(ValueError, match="given together"):
            P.all_to_all(x, group="dense_tp", in_splits=[2, 2])
        with pytest.raises(ValueError, match="given together"):
            P.all_to_all(x, group="dense_tp", out_splits=[2, 2])
        # Even exchange (neither) and fully specified uneven exchange (both)
        # are accepted; at world size one both are the identity.
        assert torch.equal(P.all_to_all(x, group="dense_tp"), x)
        assert torch.equal(
            P.all_to_all(x, group="dense_tp", in_splits=[4], out_splits=[4]), x
        )
    finally:
        P.shutdown()


def test_warmup_runs_host_tensors_inline_without_touching_cuda(monkeypatch):
    def no_cuda(*_args, **_kwargs):
        raise AssertionError("CPU warmup must not create CUDA streams")

    monkeypatch.setattr(torch.cuda, "Stream", no_cuda)
    monkeypatch.setattr(torch.cuda, "synchronize", no_cuda)
    calls: list[torch.device] = []
    result = P.warmup(
        lambda tensor, *, group: calls.append(tensor.device) or group,
        torch.zeros(1),
        group="dense_tp",
    )
    assert result == "dense_tp"
    assert calls == [torch.device("cpu")]
