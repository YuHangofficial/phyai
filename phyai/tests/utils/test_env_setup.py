"""Unit tests for phyai.utils.env_setup, the write side of the environment."""

from __future__ import annotations

import os
import resource

import pytest

from phyai.utils.env_setup import (
    TUNED_ENV_VARS,
    init_env,
    init_process_debug,
    set_ulimit,
)


@pytest.fixture(autouse=True)
def _clean_environ():
    """Snapshot/restore os.environ and start every test from an untuned state."""
    saved = dict(os.environ)
    for var in TUNED_ENV_VARS:
        os.environ.pop(var.name, None)
    os.environ.pop("PHYAI_SKIP_ENV_SETUP", None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _cuda_vars() -> tuple[str, ...]:
    return tuple(v.name for v in TUNED_ENV_VARS if v.applies_when(1, "cuda"))


def test_init_env_writes_applicable_vars_once_and_never_overwrites_a_preset():
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    applied = init_env(world_size=1, device_type="cuda")
    assert set(applied) == set(_cuda_vars()) - {"CUDA_DEVICE_MAX_CONNECTIONS"}
    for name, value in applied.items():
        assert os.environ[name] == value
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert init_env(world_size=1, device_type="cuda") == {}  # idempotent


def test_init_env_stays_out_of_the_way_for_cpu_targets_and_opt_outs():
    assert init_env(world_size=1, device_type="cpu") == {}
    os.environ["PHYAI_SKIP_ENV_SETUP"] = "1"
    assert init_env(world_size=1, device_type="cuda") == {}
    assert not any(name in os.environ for name in _cuda_vars())


def test_recommended_only_vars_are_documented_but_never_written():
    """The NCCL / allocator entries stay documentation until measured."""
    applied = init_env(world_size=8, device_type="cuda")
    for name in (
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NCCL_GRAPH_MIXING_SUPPORT",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        assert name not in applied and name not in os.environ
    for var in TUNED_ENV_VARS:
        assert var.why.strip(), f"{var.name} has no rationale"


def test_set_ulimit_never_lowers_a_limit_and_warns_instead_of_raising_when_capped():
    """A hard limit below the target is the operator's call, not an error."""
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    set_ulimit(target_soft_limit=1)
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == before
    _soft, hard = before
    target = (1 << 40) if hard == resource.RLIM_INFINITY else hard + 1
    set_ulimit(target_soft_limit=target)  # must not raise
    soft_after, hard_after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert hard_after == hard
    if hard != resource.RLIM_INFINITY:
        assert soft_after <= hard


def test_init_process_debug_sets_only_the_requested_title(monkeypatch):
    titles = []
    monkeypatch.setattr("setproctitle.setproctitle", titles.append)
    init_process_debug()
    init_process_debug(title="phyai::test_DP1_TP2")
    assert titles == ["phyai::test_DP1_TP2"]
