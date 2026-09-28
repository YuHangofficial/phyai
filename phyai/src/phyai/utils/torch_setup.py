"""Process-wide torch state: thread pools, RNG seeding, autograd mode.

Three entry points the engine calls once at startup, split out of
:class:`~phyai.engine.Engine` for the same reason as
:func:`~phyai.utils.cuda.init_cuda` — each is independently callable, so a
test or a bespoke harness can opt into one without the full engine
bootstrap.

The distinction from :mod:`phyai.utils.env_setup` matters: that module
writes ``os.environ``, which mostly affects *child* processes; this one
mutates torch globals that take effect immediately in *this* process.
Where the two overlap (thread counts) both are needed, and neither is
sufficient alone — see :func:`init_threads`.

Default dtype is deliberately not here. It is pinned for inference by
:func:`phyai.utils.cuda.init_cuda`.
``torch.set_default_device`` is deliberately never set: phyai layers pass
``device=`` explicitly, and host-side tensors (``cu_seqlens``, sampling
metadata) must stay on CPU.
"""

from __future__ import annotations

import os
import random

import torch

from phyai.utils.logging import get_logger


logger = get_logger(__name__)


#: Device types where the CPU is a launcher, not the compute engine, so a
#: core-count-sized intra-op pool is pure contention.
ACCELERATOR_DEVICE_TYPES: frozenset[str] = frozenset({"cuda", "npu", "mlu", "xpu"})


def init_threads(*, device_type: str, num_threads: int | None = None) -> int:
    """Size torch's intra-op thread pool for the target device.

    Args:
        device_type: device *type* of the run (``"cuda"`` / ``"cpu"`` / ...).
        num_threads: explicit override. ``None`` (the default) auto-picks:
            ``1`` on an accelerator, and *no change at all* on CPU.

    Returns:
        The effective ``torch.get_num_threads()`` after the call.

    On an accelerator the CPU only builds metadata and launches kernels,
    so torch's default pool (one thread per core) buys nothing and costs
    two things. First, weight loading gets slower, not faster — many
    threads contend on the same shard reads and allocator locks (sglang
    sets ``num_threads=1`` right before ``load_model`` for exactly this
    reason). Second, and worse for phyai's latency-critical robot
    workloads: TP=8 or several serving replicas on one box means eight
    processes each opening a core-count-sized pool, oversubscribing the
    machine by an order of magnitude and showing up as launch-side jitter
    in p99.

    A CPU target is the opposite case — the pool *is* the compute — so
    ``None`` leaves it untouched rather than crippling CPU inference and
    the CPU-default test suite.

    ``torch.set_num_interop_threads`` is attempted too, but only ever
    best-effort: torch refuses it once any parallel work has started, and
    a late call must not fail a launch.
    """
    if num_threads is None:
        if device_type not in ACCELERATOR_DEVICE_TYPES:
            current = torch.get_num_threads()
            logger.debug_rank0(
                "device_type=%s is the compute device; leaving torch.num_threads "
                "at %d.",
                device_type,
                current,
            )
            return current
        num_threads = 1

    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}.")

    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError as e:
        # Already-started parallel work. Expected in-process on a second
        # engine build; not worth more than a debug line.
        logger.debug_rank0("set_num_interop_threads(%d): %s", num_threads, e)
    if num_threads != previous:
        logger.info_rank0(
            "torch intra-op threads %d -> %d (device_type=%s)",
            previous,
            num_threads,
            device_type,
        )
    return torch.get_num_threads()


def init_seed(seed: int | None) -> None:
    """Seed every global RNG phyai or its dependencies might draw from.

    Covers ``random``, ``numpy``, ``torch`` and all CUDA devices. ``None``
    is a no-op — an unseeded process is the default so that nothing
    silently changes for callers who never asked for determinism.

    This only touches *global* generators. Model code that wants
    reproducibility independent of process-level seeding should keep using
    its own generator object (the cosmos3 samplers construct a fresh
    ``numpy.random.RandomState(seed)`` per request precisely so their
    noise is unaffected by whatever else the process did).
    """
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info_rank0("seeded global RNGs with %d", seed)


def disable_grad() -> None:
    """Turn autograd off for the current thread.

    Unconditional, and not a config knob: :class:`~phyai.engine.Engine` is
    the inference entry point, so "this process computes gradients" is not
    a state it has. Code that wants gradients builds phyai layers directly
    and never constructs an Engine.

    Today this is redundant — every phyai layer allocates its parameters
    with ``requires_grad=False``, so no forward pass produces a
    grad-requiring tensor to record. It is here because that invariant is
    held by *convention* across 20-odd allocation sites and held
    unevenly: only 2 of 9 schedulers wrap ``setup()``, which is where
    warmup and CUDA-graph capture run. One line at the process level makes
    the invariant enforced instead of conventional, and costs nothing to
    keep.

    :class:`~phyai.engine.Engine` leaves autograd disabled for its inference
    thread, including after shutdown.
    """
    torch.set_grad_enabled(False)


def local_rank() -> int:
    """This process's local rank from the launcher env, or ``0``.

    Reads ``LOCAL_RANK`` (torchrun / torchelastic). Lives here rather than
    in :mod:`phyai.env` because it is a *launcher* variable, not a
    ``PHYAI_*`` knob, and several bootstrap sites need it before any
    config exists.
    """
    raw = os.environ.get("LOCAL_RANK")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning_rank0(
            "LOCAL_RANK=%r is not an integer; treating this process as local rank 0.",
            raw,
        )
        return 0


__all__ = [
    "ACCELERATOR_DEVICE_TYPES",
    "disable_grad",
    "init_seed",
    "init_threads",
    "local_rank",
]
