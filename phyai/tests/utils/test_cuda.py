"""Unit tests for phyai.utils.cuda device-resolution and probe helpers."""

from __future__ import annotations

import os

import pytest
import torch

from phyai.utils.cuda import (
    available_memory_bytes,
    format_gib,
    init_cublas,
    init_cuda,
    memory_summary,
    resolve_device,
)


@pytest.fixture(autouse=True)
def _restore_globals():
    dtype = torch.get_default_dtype()
    saved_env = dict(os.environ)
    yield
    torch.set_default_dtype(dtype)
    os.environ.clear()
    os.environ.update(saved_env)


def test_resolve_device_folds_local_rank_into_a_bare_cuda_target_only():
    """A torchrun rank must not silently bind GPU 0; an explicit index, a
    non-CUDA device and a torch.device object pass through untouched."""
    os.environ["LOCAL_RANK"] = "3"
    assert resolve_device("cuda") == torch.device("cuda", 3)
    assert resolve_device("cuda:2") == torch.device("cuda", 2)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cuda", 1)) == torch.device("cuda", 1)
    os.environ.pop("LOCAL_RANK")
    assert resolve_device("cuda") == torch.device("cuda", 0)


def test_init_cuda_sets_the_default_dtype_and_cublas_warmup_is_silent():
    torch.set_default_dtype(torch.float32)
    assert init_cuda("cpu", torch.float64) is None
    assert torch.get_default_dtype() is torch.float64
    assert init_cublas() is None  # whether or not a context already exists


def test_memory_probes_degrade_and_agree():
    assert memory_summary("cuda:999") == (0, 0)
    free, _total = memory_summary()
    # ``mem_get_info`` is a live driver query; two adjacent calls can differ
    # by allocator/page bookkeeping while no test-owned allocation changed.
    assert abs(available_memory_bytes() - free) <= 8 * (1 << 20)
    assert (format_gib(0), format_gib(1 << 30), format_gib(3 * (1 << 29))) == (
        "0.00",
        "1.00",
        "1.50",
    )
