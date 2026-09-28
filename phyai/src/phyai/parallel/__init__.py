"""Distributed primitives over explicit logical groups."""

from __future__ import annotations

import torch
import torch.distributed as torch_dist

from phyai.parallel.backend import Backend, Op, Topology
from phyai.parallel.config import ParallelConfig
from phyai.parallel.dispatch import (
    Dispatcher,
    get_dispatcher,
    reset_dispatcher,
    set_dispatcher,
)
from phyai.parallel.exceptions import NoBackendError, PhyaiDistError
from phyai.parallel.layout import GROUP_NAMES, WORLD, RankLayout, build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.parallel.process_groups import ProcessGroupPool
from phyai.parallel.topology import PlacementEntry, probe_placement
from phyai.parallel.ops import (
    all_gather,
    all_reduce,
    all_to_all,
    barrier,
    broadcast,
    recv,
    reduce_scatter,
    send,
)
from phyai.parallel.registry import DefaultPolicy, ForcedPolicy, Policy, Registry
from phyai.parallel.state import (
    Mode,
    clear_meshes,
    current_mode,
    default_mesh,
    graph_capture,
    register_mesh,
    registered_meshes,
    use_mesh,
)
from phyai.parallel.backends import GlooBackend, NcclBackend, PyNCCLBackend
from phyai.utils import get_logger

logger = get_logger(__name__)


def _resolve_backend(backend: str | None, device: str | torch.device | None) -> str:
    if backend not in (None, "auto"):
        return backend
    world_backend = str(torch_dist.get_backend())
    if world_backend in ("nccl", "gloo"):
        return world_backend
    if device is None:
        return "nccl" if torch.cuda.is_available() else "gloo"
    return "nccl" if torch.device(device).type == "cuda" else "gloo"


def resolve_collective_device(device: str | torch.device | None) -> torch.device:
    """Resolve the CUDA device this rank's NCCL collectives bind.

    ``None`` and a bare ``"cuda"`` resolve to the device the caller pinned
    before :func:`init` — the engine pins in ``init_cuda`` / ``init_dist``,
    and spawned workers pin their placement index — so under full device
    visibility the communicator lands on this rank's own GPU rather than on
    ``rank % device_count`` arithmetic that only held with per-worker
    ``CUDA_VISIBLE_DEVICES`` masks. An explicit ``"cuda:K"`` is honored
    as-is. Non-CUDA devices are a configuration error here: an NCCL
    registry without a CUDA device cannot work.
    """
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"NCCL collectives require a CUDA device, got {device!r}.")
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _make_registry(
    mesh: Mesh,
    *,
    backend: str,
    device: str | torch.device | None,
    enable_pynccl: bool,
    pynccl_groups: list[str] | None,
    pynccl_library_path: str | None,
) -> Registry:
    registry = Registry()
    if backend == "nccl":
        if enable_pynccl:
            pynccl = PyNCCLBackend(library_path=pynccl_library_path)
            groups = list(mesh.group_names) if pynccl_groups is None else pynccl_groups
            try:
                pynccl.attach(mesh, groups, device=resolve_collective_device(device))
            except BaseException:
                pynccl.close()
                raise
            registry.register(
                pynccl,
                prefer_for={
                    Op.ALL_REDUCE,
                    Op.ALL_GATHER,
                    Op.REDUCE_SCATTER,
                    Op.BROADCAST,
                    Op.SEND,
                    Op.RECV,
                },
            )
        registry.register(NcclBackend())
    elif backend == "gloo":
        registry.register(GlooBackend())
    else:
        raise ValueError(
            f"phyai.parallel.init: unsupported backend {backend!r}. "
            "Use 'nccl', 'gloo', or 'auto'."
        )
    try:
        registry.validate(
            group_sizes={mesh.group_size(name) for name in mesh.group_names}
        )
    except BaseException:
        for registered in registry.all():
            registered.close()
        raise
    return registry


def init(
    parallel: ParallelConfig,
    *,
    replica_world_size: int | None = None,
    device: str | torch.device | None = None,
    backend: str | None = None,
    enable_pynccl: bool = True,
    pynccl_groups: list[str] | None = None,
    pynccl_library_path: str | None = None,
) -> Mesh:
    """Initialize all logical groups of one serving replica.

    The physical replica world is the size of the initialized torch process
    group. Without one only a single-rank replica can be initialized; its
    size is ``replica_world_size`` or is inferred from ``parallel``.

    ``device`` selects the device this rank's NCCL collectives bind (see
    :func:`resolve_collective_device`): ``None`` or a bare ``"cuda"`` mean
    the device the caller already pinned, an explicit ``"cuda:K"`` is
    honored. The value is ignored for gloo, whose collectives are host-side.

    ``pynccl_groups`` names the groups that get a direct NCCL communicator
    for graph capture. ``None`` attaches every group of the mesh, ``world``
    included; groups with the same members share one communicator, and a
    group left out falls back to torch's own NCCL process group.
    """
    if not isinstance(parallel, ParallelConfig):
        raise TypeError(
            "phyai.parallel.init expects a ParallelConfig, "
            f"got {type(parallel).__name__}."
        )
    if torch_dist.is_initialized():
        world = torch_dist.get_world_size()
        if replica_world_size is not None and replica_world_size != world:
            raise ValueError(
                f"replica_world_size={replica_world_size} does not match the "
                f"initialized process group of {world} ranks."
            )
    else:
        world = (
            parallel.infer_replica_world_size()
            if replica_world_size is None
            else replica_world_size
        )
        if world > 1:
            raise RuntimeError(
                "phyai.parallel.init requires an initialized torch.distributed "
                f"process group for a multi-rank replica (world_size={world})."
            )
    layout = build_rank_layout(parallel.resolve(world))
    if not torch_dist.is_initialized():
        mesh = Mesh(layout)
        register_mesh(mesh)
        set_dispatcher(Dispatcher(registry=Registry()))
        return mesh

    resolved_backend = _resolve_backend(backend, device)
    rank = torch_dist.get_rank()
    build_cpu = enable_pynccl and resolved_backend == "nccl"
    pool = ProcessGroupPool(layout, backend=resolved_backend, build_cpu=build_cpu)
    process_groups: dict[str, torch_dist.ProcessGroup] = {}
    cpu_groups: dict[str, torch_dist.ProcessGroup] = {}
    for name in layout.groups:
        ranks = layout.members_for(name, rank)
        if len(ranks) > 1:
            process_groups[name] = pool.get(ranks, resolved_backend)
            if build_cpu:
                cpu_groups[name] = pool.get(ranks, "gloo")
    mesh = Mesh(
        layout,
        rank=rank,
        process_groups=process_groups,
        cpu_groups=cpu_groups,
        pool=pool,
    )
    # Placement is probed, not configured: node identity and NVML device
    # index are gathered over the world group, NVLink is checked with NVML.
    try:
        probe_device = (
            resolve_collective_device(device)
            if resolved_backend == "nccl"
            else torch.device("cpu")
        )
        mesh.set_placement(probe_placement(torch_dist.group.WORLD, device=probe_device))
    except Exception as error:  # noqa: BLE001 - a probe failure must not fail init
        logger.warning_rank0(
            "placement probe failed (%s: %s); using the device-count fallback",
            type(error).__name__,
            error,
        )
    try:
        registry = _make_registry(
            mesh,
            backend=resolved_backend,
            device=device,
            enable_pynccl=enable_pynccl,
            pynccl_groups=pynccl_groups,
            pynccl_library_path=pynccl_library_path,
        )
        dispatcher = Dispatcher(registry=registry)
    except BaseException:
        mesh.close()
        raise
    register_mesh(mesh)
    set_dispatcher(dispatcher)
    return mesh


def warmup(callable, /, *args, **kwargs) -> object:
    """Run one call on a CUDA side stream when its tensors live on CUDA.

    Calls whose tensors are on the host run inline: touching ``torch.cuda``
    for them would create a CUDA context on device 0 in every CPU/gloo worker
    of a GPU host (and fail outright when that device is full).
    """
    tensors = [
        value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)
    ]
    cuda_devices = [tensor.device for tensor in tensors if tensor.device.type == "cuda"]
    if not cuda_devices:
        return callable(*args, **kwargs)
    with torch.cuda.device(cuda_devices[0]):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = callable(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
    return result


def collective_device(mesh: Mesh, group: str) -> torch.device:
    """Device that collectives on ``group`` operate on, from its backend.

    NCCL groups move CUDA tensors, gloo groups host tensors. The decision
    follows the group's backend rather than whether CUDA
    happens to be available on the host, so a CPU/gloo replica on a GPU box
    warms and runs on the CPU path it will use.
    """
    backend = str(torch_dist.get_backend(mesh.group(group)))
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    if backend == "gloo":
        return torch.device("cpu")
    # Composite / unknown backend (an externally created default group):
    # fall back to the host's capability.
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def warmup_collectives(groups: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Build communicators up front with one tiny all-reduce per group.

    The implicit ``world`` group is included by default so its NCCL
    communicator is built during engine warmup rather than on the first
    world-spanning collective (for example a tiled VAE decode). Each group is
    warmed on the device its process group serves (see
    :func:`collective_device`).
    """
    if not torch_dist.is_initialized():
        return ()
    mesh = default_mesh()
    candidates = mesh.group_names if groups is None else groups
    warmed: list[str] = []
    for group in candidates:
        if mesh.group_size(group) <= 1:
            continue
        device = collective_device(mesh, group)
        tensor = torch.zeros(1, dtype=torch.float32, device=device)
        warmup(all_reduce, tensor, group=group)
        warmed.append(group)
    return tuple(warmed)


def shutdown() -> None:
    """Release everything :func:`init` built, leaving torch.distributed alone.

    Closes every registered backend (pynccl destroys its direct NCCL
    communicators), destroys owned device and CPU groups, and drops the
    process-level dispatcher and mesh registry so a later :func:`init` in the
    same process starts clean. Call it before
    ``torch.distributed.destroy_process_group``. Idempotent.
    """
    try:
        dispatcher = get_dispatcher()
    except RuntimeError:
        dispatcher = None
    error: BaseException | None = None
    resources = list(dispatcher.registry.all()) if dispatcher is not None else []
    resources.extend(registered_meshes())
    for resource in resources:
        try:
            resource.close()
        except BaseException as caught:
            if error is None:
                error = caught
    reset_dispatcher()
    clear_meshes()
    if error is not None:
        raise error


__all__ = [
    "init",
    "shutdown",
    "collective_device",
    "GROUP_NAMES",
    "WORLD",
    "RankLayout",
    "PlacementEntry",
    "Mesh",
    "Mode",
    "default_mesh",
    "use_mesh",
    "current_mode",
    "graph_capture",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "send",
    "recv",
    "barrier",
    "warmup",
    "warmup_collectives",
    "NcclBackend",
    "GlooBackend",
    "PyNCCLBackend",
    "Backend",
    "Op",
    "Topology",
    "Registry",
    "Policy",
    "DefaultPolicy",
    "ForcedPolicy",
    "Dispatcher",
    "get_dispatcher",
    "PhyaiDistError",
    "NoBackendError",
]
