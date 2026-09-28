"""The torch backend deliberately rejects multi-shard splitting.

ATen exposes no public split / remainder API, and two back-to-back
``GreenContext.create`` calls produce streams that nearly serialise rather
than running disjoint, so the backend raises ``BackendCapabilityError`` instead
of handing callers non-disjoint shards. Single-shard ``create_single`` works.
"""

from __future__ import annotations

import pytest
import torch

import phyai.vgpu as V
from phyai.vgpu.exceptions import BackendCapabilityError


def test_torch_backend_refuses_every_split_but_creates_a_single_vgpu():
    V.init(device="cuda:0", backend="torch")
    with pytest.raises(BackendCapabilityError) as excinfo:
        V.split_device("cuda:0", num_groups=2, min_count=16)
    assert "torch backend" in str(excinfo.value) and "flashinfer" in str(excinfo.value)
    with pytest.raises(BackendCapabilityError):
        V.split_device_by_sm_count("cuda:0", sm_counts=[16, 16])

    a = V.vGPU(name="solo", sm_count=64, backend="torch")
    try:
        assert (a.shard.sm_count, a.shard.backend) == (64, "torch")
        assert isinstance(a.stream, torch.cuda.Stream)
        with a.activate():
            x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
            _ = x @ x
        torch.cuda.synchronize()
    finally:
        a.close()
