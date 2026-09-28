"""NVML helpers: physical indices, NVLink probe, fabric identity."""

from __future__ import annotations

import pytest
import torch

from phyai.utils import nvml


def test_nvml_probe_is_available_and_self_consistent():
    assert nvml.node_identity() and nvml.node_identity() == nvml.node_identity()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not nvml.nvml_available():
        pytest.skip("pynvml is not importable")
    try:
        with nvml.nvml_session():
            pass
    except Exception as error:  # containers without the NVML library
        pytest.skip(f"NVML is unavailable here: {error}")
    physical = [nvml.physical_device_index(i) for i in range(torch.cuda.device_count())]
    assert len(set(physical)) == len(physical)
    assert nvml.nvlink_fully_connected(physical[:1]) is True
    if len(physical) >= 2:
        forward = nvml.nvlink_fully_connected(physical[:2])
        assert isinstance(forward, bool)
        assert forward == nvml.nvlink_fully_connected(physical[1::-1])
    clique = nvml.fabric_clique(physical[0])
    assert clique is None or (isinstance(clique, str) and ":" in clique)
