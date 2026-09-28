"""Tests for Engine worker device binding helpers."""

from __future__ import annotations

import pytest

from phyai.server.engine_worker import worker_device_target


@pytest.mark.parametrize(
    ("target", "device_index", "expected"),
    (
        ("cuda", 2, "cuda:2"),
        ("cuda:0", 2, "cuda:2"),
        ("cuda:7", 0, "cuda:0"),
        ("cpu", 3, "cpu"),
    ),
)
def test_worker_device_target_rebinds_cuda_targets_only(target, device_index, expected):
    # Placement owns device assignment in managed mode: any CUDA target lands
    # on the placement index; non-CUDA targets pass through untouched.
    assert worker_device_target(target, device_index) == expected
