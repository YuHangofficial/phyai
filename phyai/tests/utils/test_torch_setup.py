"""Unit tests for phyai.utils.torch_setup, process-wide torch state.

Every test restores what it changed: these are process globals, and a leak
would silently reconfigure the rest of the suite.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from phyai.utils.torch_setup import disable_grad, init_seed, init_threads, local_rank


@pytest.fixture(autouse=True)
def _restore_torch_globals():
    threads = torch.get_num_threads()
    grad = torch.is_grad_enabled()
    saved_env = dict(os.environ)
    yield
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad)
    os.environ.clear()
    os.environ.update(saved_env)


def test_init_threads_pins_accelerators_to_one_thread_and_leaves_cpu_alone():
    """On a CPU target the intra-op pool IS the compute; do not shrink it."""
    before = torch.get_num_threads()
    assert init_threads(device_type="cpu") == before == torch.get_num_threads()
    assert init_threads(device_type="cuda") == 1 == torch.get_num_threads()
    assert init_threads(device_type="cuda", num_threads=3) == 3
    assert init_threads(device_type="cpu", num_threads=2) == 2
    with pytest.raises(ValueError, match="num_threads"):
        init_threads(device_type="cuda", num_threads=0)


def test_init_seed_covers_the_three_global_rngs_and_nothing_else():
    """``None`` is a no-op; a seed makes random/numpy/torch reproducible; model
    code owning a RandomState (the cosmos3 samplers' per-request noise) must be
    immune to process seeding."""
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    init_seed(None)
    assert torch.equal(torch.rand(4), expected)

    init_seed(7)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    init_seed(7)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == first

    local = np.random.RandomState(99).rand(4)
    init_seed(12345)
    assert np.array_equal(np.random.RandomState(99).rand(4), local)


def test_disable_grad_turns_autograd_off_idempotently():
    torch.set_grad_enabled(True)
    disable_grad()
    assert not torch.is_grad_enabled()
    disable_grad()
    assert not torch.is_grad_enabled()


def test_local_rank_reads_the_launcher_env_and_falls_back_to_zero():
    os.environ["LOCAL_RANK"] = "3"
    assert local_rank() == 3
    os.environ["LOCAL_RANK"] = "not-an-int"
    assert local_rank() == 0
    os.environ.pop("LOCAL_RANK")
    assert local_rank() == 0
