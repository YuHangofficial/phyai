"""Default select-time benchmark hook for ``profile: autotune``.

``Selector._tune`` calls the hook once per prepared candidate and keeps the
fastest. This module supplies the hook the engine installs when the policy
asks for autotune and the caller did not provide one programmatically
(``KernelConfig.benchmark``).

Only operations whose :class:`~phyai.kernel.opspec.OpSpec` declares
``bench_args`` can be measured: those are the tensor-level ops (GEMM, the
norms) whose inputs are fully described by the call's facts. The attention
family prepares a backend constructor that needs a live runner and KV state,
which cannot be synthesized here — its calls raise, ``_tune`` records no
measurement, and the catalog's priority order stands. Pin attention variants
with a user-supplied policy rule backed by a real measurement instead (set
``PHYAI_KERNEL_CONFIG`` to that policy's YAML path).
"""

from __future__ import annotations

import time
import statistics
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phyai.kernel.facts import Facts
    from phyai.kernel.opspec import Impl
    from phyai.kernel.registry import Catalog
    from phyai.kernel.selector import Selection, BenchmarkFn

#: Unmeasured invocations before timing starts.
WARMUP_ITERS = 3
#: Measured invocations; the median is reported.
TIMED_ITERS = 10


def bench_device(selection: "Selection") -> torch.device:
    """Return the torch device a selection's query describes."""

    profile = selection.query.device
    if profile.vendor in {"nvidia", "amd"}:
        return torch.device("cuda", profile.index or 0)
    return torch.device("cpu")


def default_benchmark(catalog: "Catalog") -> "BenchmarkFn":
    """Build the engine's default benchmark hook over an op catalog."""

    def measure(impl: "Impl", facts: "Facts", selection: "Selection") -> float:
        spec = catalog.op(impl.op)
        if spec.bench_args is None:
            raise NotImplementedError(
                f"op {impl.op!r} declares no bench_args and cannot be measured "
                f"at selection time"
            )
        device = bench_device(selection)
        args = spec.bench_args(facts, device)
        fn = selection.implementation

        samples: list[float] = []
        if device.type == "cuda":
            for _ in range(WARMUP_ITERS):
                fn(*args)
            torch.cuda.synchronize(device)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(TIMED_ITERS):
                begin.record()
                fn(*args)
                end.record()
                end.synchronize()
                samples.append(float(begin.elapsed_time(end)))
        else:
            for _ in range(WARMUP_ITERS):
                fn(*args)
            for _ in range(TIMED_ITERS):
                started = time.perf_counter()
                fn(*args)
                samples.append((time.perf_counter() - started) * 1000.0)
        return statistics.median(samples)

    return measure


__all__ = ["bench_device", "default_benchmark"]
