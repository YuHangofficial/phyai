"""End-to-end smoke tests for the vGPU class on the flashinfer backend."""

from __future__ import annotations

import pytest
import torch

import phyai.vgpu as V


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _flashinfer_available(), reason="flashinfer required for default backend"
)


def test_activate_runs_on_the_vgpu_stream_and_computes_correctly():
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    try:
        dev = torch.device("cuda:0")
        torch.manual_seed(0)
        x = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        y = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        z_ref = (x @ y).clone()
        torch.cuda.synchronize()
        # ``current_stream()`` returns a fresh wrapper each call: compare handles.
        before_handle = torch.cuda.current_stream().cuda_stream
        with a.activate():
            assert torch.cuda.current_stream().cuda_stream == a.stream.cuda_stream
            z = x @ y
        torch.cuda.synchronize()
        assert torch.cuda.current_stream().cuda_stream == before_handle
        rel_err = ((z - z_ref).abs().mean() / z_ref.abs().mean()).item()
        assert rel_err < 1e-2, rel_err
    finally:
        a.close()


def test_close_is_idempotent_and_activate_after_close_raises():
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    a.close()
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        with a.activate():
            pass


def test_own_mem_pool_defaults_on_and_can_be_disabled():
    V.init(device="cuda:0")
    pooled = V.vGPU(name="a", sm_count=64)
    shared = V.vGPU(name="b", sm_count=64, own_mem_pool=False)
    try:
        assert pooled.mem_pool is not None and shared.mem_pool is None
    finally:
        pooled.close()
        shared.close()


def test_create_vgpus_names_shards_and_appends_the_remainder_on_request():
    V.init(device="cuda:0")
    a, b = V.create_vgpus(device="cuda:0", sm_counts=[64, 64], names=["a", "b"])
    try:
        assert (a.name, b.name) == ("a", "b")
        assert a.shard.sm_count == b.shard.sm_count == 64
    finally:
        a.close()
        b.close()
    vgpus = V.create_vgpus(
        device="cuda:0", sm_counts=[16, 16], include_remainder_vgpu=True
    )
    try:
        assert len(vgpus) == 3 and vgpus[-1].shard.is_remainder is True
    finally:
        for v in vgpus:
            v.close()
    with pytest.raises(ValueError, match="names length"):
        V.create_vgpus(device="cuda:0", sm_counts=[16, 16], names=["only-one"])


def test_vgpu_from_shard_adopts_the_shard_and_its_stream():
    V.init(device="cuda:0")
    shards = V.split_device("cuda:0", num_groups=2, min_count=16)
    a = V.vGPU.from_shard(shards[0], name="a")
    try:
        assert a.name == "a" and a.shard is shards[0] and a.stream is shards[0].stream
    finally:
        a.close()
