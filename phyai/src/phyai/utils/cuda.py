"""CUDA device-capability helpers shared across phyai.

:func:`device_capability` returns the raw ``(major, minor)`` tuple that
callers like :func:`phyai.vgpu.topology.round_up_sm_count` expect, and
raises if CUDA is unavailable. :func:`sm_arch` returns the packed integer
form (``major * 10 + minor``) used for kernel dispatch keys, with a
graceful ``0`` fallback so init paths stay safe on developer laptops or
in forked subprocesses. :func:`resolve_device` turns a config's device
string into the concrete device *this* process should own, folding in the
launcher's ``LOCAL_RANK``. :func:`init_cuda` / :func:`init_cublas` are the
discrete bootstrap entry points the engine and tests call to pin device
+ default dtype and tune cuBLAS/cuDNN — each is independently callable
so callers can opt into pieces without committing to the full engine
orchestration. :func:`memory_summary` / :func:`available_memory_bytes`
report free and total device memory. :func:`print_topology` dumps a per-device summary
plus a peer-access matrix for the local node;
:func:`print_distributed_topology` extends that to a multi-node
:mod:`torch.distributed` group with per-host IB HCAs and GPU↔NIC affinity
from ``nvidia-smi topo -m``.
"""

from __future__ import annotations

import sys
from typing import TextIO

import torch

from phyai.utils.torch_setup import local_rank


def device_capability(
    device: "torch.device | str | int | None" = None,
) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def current_device() -> torch.device:
    """Return the current device for tensor allocation.

    Picks the active CUDA device when CUDA is available, otherwise
    ``cpu``. ``torch.cuda.current_device()`` returns an int rank, but
    callers typically want a ``torch.device`` they can pass to
    ``.to(...)`` or ``torch.empty(..., device=...)``; this wraps the
    rank into a ``torch.device("cuda", rank)``.

    Use this in place of hard-coded ``"cuda"`` / ``"cpu"`` strings so
    a process started under ``CUDA_VISIBLE_DEVICES=...`` lands on the
    intended device, and CPU-only dev / CI environments degrade
    gracefully.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def sm_arch(
    device: "torch.device | str | int | None" = None,
) -> int:
    """Compute capability as ``major * 10 + minor``, or ``0`` without CUDA.

    The ``0`` is a *truth value*, not a version. Callers may test it; they must
    not compare it. ``phyai.kernel`` uses the opposite convention -- ``None`` for
    "this device has no compute capability" -- precisely because a 0 that reaches
    a comparison makes a CPU host read as "too old" rather than "unknown", which
    is how the old selector reported "no backend available" on CPU.
    """

    if not torch.cuda.is_available():
        return 0
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (RuntimeError, AssertionError):
        # CUDA may be visible but unusable (e.g. forked subprocess of a
        # parent that already initialized CUDA).
        return 0
    return major * 10 + minor


def resolve_device(device: "torch.device | str") -> torch.device:
    """Resolve a config device string to the device this process should own.

    Pure function, no side effects. Three cases:

    * an explicit index (``"cuda:3"``) is returned unchanged — the caller
      knows what it wants;
    * a bare ``"cuda"`` under a launcher becomes ``cuda:{LOCAL_RANK}``;
    * a bare ``"cuda"`` with no launcher becomes ``cuda:0``.

    The middle case is the one that matters. ``EngineConfig.device.target``
    defaults to plain ``"cuda"``, and pinning that to device 0 means every
    rank of a ``torchrun`` job binds the same GPU for the whole window
    between here and the point where the process group is up — so warmup
    tensors, cuBLAS handles and workspace allocations all land on GPU 0
    while rank 3 believes it is on GPU 3. Folding ``LOCAL_RANK`` in here
    makes the very first ``set_device`` correct.

    Non-CUDA devices pass through untouched.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda" or dev.index is not None:
        return dev
    return torch.device("cuda", local_rank())


def init_cuda(
    device: "torch.device | str",
    params_dtype: torch.dtype,
) -> None:
    """Pin the CUDA current device and the process default dtype.

    A CPU device skips ``torch.cuda.set_device``, but the dtype is still
    pinned because fp32 and fp64 weights need it just as much. The engine
    keeps this dtype for the lifetime of its inference process, including
    after shutdown.

    The device is routed through :func:`resolve_device`, so a bare
    ``"cuda"`` lands on this process's local rank rather than on device 0.
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.set_default_dtype(params_dtype)


def init_cublas(*, allow_tf32: bool = False) -> None:
    """Create the cuBLAS handle up front and set fp32 matmul precision.

    The handle is built by running one tiny matmul. That looks pointless
    and is not: cuBLAS initializes lazily on first use, so without this
    the cost (and any failure — a missing library, a driver mismatch)
    surfaces inside whatever happens to issue the first GEMM. That is
    frequently a warmup step inside CUDA-graph capture, where a lazy
    initialization is both hard to read in a profile and, for some
    library versions, illegal.

    ``allow_tf32`` controls the fp32 paths only (``matmul.allow_tf32`` and
    the equivalent ``set_float32_matmul_precision``). Default off: phyai
    runs bf16/fp8/nvfp4 weights, where the flag is irrelevant, and the
    fp32 modules that do exist (ViT stems, time-embedding MLPs) are
    exactly the places a silent precision drop would be unwelcome.

    cuDNN is deliberately left alone. ``cudnn.benchmark`` re-plans on new
    shapes, which fights CUDA-graph capture, and cuDNN kernels inside a
    captured region have already caused hangs in the multi-replica
    green-context path — so tuning it is an explicit, per-model decision,
    not a process default.
    """
    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    a = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    torch.matmul(a, a)
    return None


def memory_summary(
    device: "torch.device | str | int | None" = None,
) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for ``device``.

    ``(0, 0)`` when CUDA is unavailable or the device cannot be queried,
    so callers can treat "no numbers" and "no device" identically instead
    of branching on ``torch.cuda.is_available()`` at every site.

    Reports the *driver's* view (``torch.cuda.mem_get_info``), not
    torch's allocator view: memory held by another process on the same
    GPU counts against free, which is the whole point on a shared box.
    """
    if not torch.cuda.is_available():
        return (0, 0)
    try:
        free, total = torch.cuda.mem_get_info(device)
    except (RuntimeError, AssertionError, ValueError):
        return (0, 0)
    return (int(free), int(total))


def available_memory_bytes(
    device: "torch.device | str | int | None" = None,
) -> int:
    """Return free device memory in bytes, or ``0`` when unavailable."""
    free, _total = memory_summary(device)
    return free


def format_gib(num_bytes: int) -> str:
    """``12.34`` for a byte count, for log lines that append ``GiB`` themselves."""
    return f"{num_bytes / (1 << 30):.2f}"


def print_topology(*, file: TextIO | None = None) -> None:
    out = file if file is not None else sys.stdout

    if not torch.cuda.is_available():
        print("CUDA: unavailable", file=out)
        return

    n = torch.cuda.device_count()
    cur = torch.cuda.current_device()
    print(f"CUDA: {n} device(s), current=cuda:{cur}", file=out)

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        major, minor = device_capability(i)
        mem_gib = props.total_memory / (1 << 30)
        print(
            f"  cuda:{i}  {props.name}  sm_{major}{minor}  "
            f"SMs={props.multi_processor_count}  mem={mem_gib:.1f} GiB",
            file=out,
        )

    if n >= 2:
        print("peer access (P=can access, .=cannot):", file=out)
        print("       " + " ".join(f"{j:>3}" for j in range(n)), file=out)
        for i in range(n):
            cells = []
            for j in range(n):
                if i == j:
                    cells.append("  -")
                else:
                    ok = torch.cuda.can_device_access_peer(i, j)
                    cells.append("  P" if ok else "  .")
            print(f"  {i:>3}: " + "".join(cells), file=out)


def print_distributed_topology(*, file: TextIO | None = None) -> None:
    import os
    import glob
    import socket
    import subprocess

    import torch.distributed as dist

    out = file if file is not None else sys.stdout

    if not dist.is_available() or not dist.is_initialized():
        print(
            "torch.distributed not initialized; "
            "call dist.init_process_group(...) first",
            file=out,
        )
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
    gpu_uuid = ""
    if dev_idx >= 0:
        gpu_uuid = str(getattr(torch.cuda.get_device_properties(dev_idx), "uuid", ""))

    ib_hcas = sorted(os.path.basename(p) for p in glob.glob("/sys/class/infiniband/*"))

    nvsmi_topo = ""
    try:
        nvsmi_topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.rstrip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    info = {
        "hostname": socket.gethostname(),
        "local_rank_env": os.environ.get("LOCAL_RANK"),
        "dev_idx": dev_idx,
        "gpu_uuid": gpu_uuid,
        "ib_hcas": ib_hcas,
        "nvsmi_topo": nvsmi_topo,
    }

    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, info)

    if rank != 0:
        return

    by_host: dict[str, list[tuple[int, dict]]] = {}
    for r, item in enumerate(gathered):
        assert item is not None
        by_host.setdefault(item["hostname"], []).append((r, item))

    print(
        f"distributed: world_size={world_size}, hosts={len(by_host)}",
        file=out,
    )

    for host, ranks_on_host in by_host.items():
        print(f"\n[{host}]", file=out)
        for r, item in ranks_on_host:
            lr = item["local_rank_env"] or "?"
            uuid = item["gpu_uuid"] or "?"
            print(
                f"  rank {r:>3} (LOCAL_RANK={lr}): cuda:{item['dev_idx']}  {uuid}",
                file=out,
            )
        # NIC info is host-level, so report once per host (first rank).
        rep = ranks_on_host[0][1]
        if rep["ib_hcas"]:
            print(f"  IB HCAs: {', '.join(rep['ib_hcas'])}", file=out)
        if rep["nvsmi_topo"]:
            print("  nvidia-smi topo -m:", file=out)
            for line in rep["nvsmi_topo"].splitlines():
                print(f"    {line}", file=out)
