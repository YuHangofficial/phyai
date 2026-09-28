"""Unit tests for phyai.parallel.dist.init_dist.

No real process group is created: ``init_process_group`` is stubbed so the
ordering and argument contract can be asserted on CPU, which is where CI
runs.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

import phyai.parallel.dist as D


@pytest.fixture(autouse=True)
def _restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def trace(monkeypatch):
    """Record the ordering of set_device / init_process_group calls."""
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        D.dist,
        "init_process_group",
        lambda *a, **kw: calls.append(("init_process_group", a, kw)),
    )
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a, **kw: calls.append(("set_device", a, kw))
    )
    return calls


def _names(trace):
    return [name for name, _args, _kwargs in trace]


def test_single_rank_pins_the_resolved_device_and_builds_no_group(trace):
    os.environ["LOCAL_RANK"] = "2"
    assert D.init_dist(world_size=1, device_type="cpu") is False
    assert trace == []

    # A bare "cuda" target (torchrun style) folds in LOCAL_RANK, while a
    # managed worker's explicit cuda:i must survive bootstrap untouched.
    assert D.init_dist(world_size=1, device_type="cuda", device="cuda") is False
    assert D.init_dist(world_size=1, device_type="cuda", device="cuda:1") is False
    assert _names(trace) == ["set_device", "set_device"]
    assert [args[0] for _name, args, _kwargs in trace] == [
        torch.device("cuda", 2),
        torch.device("cuda", 1),
    ]


def test_device_is_pinned_before_the_group_is_built(trace):
    """NCCL binds to the current device while building its communicator."""
    os.environ["LOCAL_RANK"] = "3"
    assert D.init_dist(world_size=4, device_type="cuda") is True
    assert _names(trace) == ["set_device", "init_process_group"]
    assert trace[0][1][0] == torch.device("cuda", 3)

    trace.clear()
    assert D.init_dist(world_size=4, device_type="cuda", device="cuda:1") is True
    assert _names(trace) == ["set_device", "init_process_group"]
    assert trace[0][1][0] == torch.device("cuda", 1)


def test_launcher_environment_drives_rank_rendezvous_and_timeout(trace):
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(key, None)
    D.init_dist(world_size=2, device_type="cpu", timeout=timedelta(seconds=42))
    _name, args, kwargs = trace[-1]
    assert args == ("gloo",)
    assert (kwargs["rank"], kwargs["world_size"]) == (0, 2)
    assert kwargs["timeout"] == timedelta(seconds=42)
    # Defaults are written back so later readers see the same rendezvous.
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["WORLD_SIZE"] == "2"

    os.environ.update({"RANK": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "1"})
    D.init_dist(world_size=8, device_type="cpu")
    _name, _args, kwargs = trace[-1]
    assert (kwargs["rank"], kwargs["world_size"]) == (5, 8)
    assert "timeout" not in kwargs  # None keeps torch's own default


def test_existing_group_is_reused_only_when_its_size_matches(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    assert D.init_dist(world_size=4, device_type="cuda") is False
    with pytest.raises(ValueError, match="does not match"):
        D.init_dist(world_size=2, device_type="cuda")
