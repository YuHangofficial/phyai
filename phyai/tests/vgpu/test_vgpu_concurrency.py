"""Two-vGPU concurrency test: GEMM on two 64-SM shards in parallel.

The flashinfer backend guarantees disjoint SM allocation, so two shards
should overlap rather than serialise. We use a generous concurrent/single
ratio threshold — anything close to fully sequential indicates
non-disjoint execution and is a regression.
"""

from __future__ import annotations

import time

import pytest
import torch

import phyai.vgpu as V


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def test_two_vgpus_run_concurrently():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for disjoint multi-shard split")
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if total_sms < 132:
        pytest.skip(f"need >= 132 SMs to allocate two 64-SM shards, have {total_sms}")

    V.init(device="cuda:0")
    a, b = V.create_vgpus(
        device="cuda:0",
        sm_counts=[64, 64],
        names=["a", "b"],
        own_mem_pool=False,  # Reuse the default pool for cleaner measurements.
    )
    try:
        dev = torch.device("cuda:0")
        x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)

        N = 30

        def warmup(stream):
            with torch.cuda.stream(stream):
                for _ in range(5):
                    _ = x @ y

        def bench_single(stream):
            warmup(stream)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(N):
                    _ = x @ y
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / N

        def bench_concurrent(s_a, s_b):
            warmup(s_a)
            warmup(s_b)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(s_a):
                for _ in range(N):
                    _ = x @ y
            with torch.cuda.stream(s_b):
                for _ in range(N):
                    _ = x @ y
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / N

        t_single = sorted(bench_single(a.stream) for _ in range(3))[1]
        t_conc = sorted(bench_concurrent(a.stream, b.stream) for _ in range(3))[1]
        ratio = t_conc / t_single

        # Generous band: memory-bound GEMM on disjoint shards still
        # contends for HBM/L2, so a perfectly flat ratio is not expected.
        # Anything close to fully sequential indicates the shards
        # serialised — a real regression.
        assert ratio < 1.7, (
            f"concurrent/single ratio {ratio:.2f}x is too close to "
            f"sequential (2.0x); shards may not be disjoint. "
            f"single={t_single * 1000:.2f} ms, concurrent={t_conc * 1000:.2f} ms"
        )
    finally:
        a.close()
        b.close()
