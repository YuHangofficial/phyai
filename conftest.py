"""Workspace-wide pytest bootstrap.

The phyai test suites REQUIRE CUDA: the engine's execution target is the
GPU, and numerical tests exercise real CUDA kernels (a CPU torch fallback
once hid a CUDA-only shape bug for months). Abort collection loudly on a
machine without CUDA instead of letting device-less tests pass and call
that green.
"""

from __future__ import annotations

import pytest
import torch


def pytest_sessionstart(session):
    if not torch.cuda.is_available():
        pytest.exit("phyai test suite requires CUDA.", returncode=1)
