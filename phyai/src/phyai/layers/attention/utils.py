"""Shared utilities for ``phyai.layers.attention``.

The flashinfer split-k scratch is process-global and per-device. Every
attention backend that uses flashinfer (the no-cache
:class:`~phyai.layers.attention.nocache.Attention` stack and the paged
:class:`~phyai.layers.attention.paged.PagedAttention` stack)
falls back to this buffer when the caller doesn't pass an explicit
``fi_workspace``. Sharing one scratch across every layer keeps memory
flat regardless of model depth.

Sizing
------
* Default: ``RuntimeConfig.flashinfer_workspace_bytes`` (128 MiB out of
  the box, 1x flashinfer's recommendation).
* Override the engine-level value via ``PHYAI_FLASHINFER_WORKSPACE_BYTES``
  (overlaid onto :class:`~phyai.engine_config.RuntimeConfig` by
  :meth:`EngineConfig.from_env`) or by passing a bespoke
  :class:`~phyai.engine_config.EngineConfig` to the engine. The
  resolver consults the :class:`EngineConfig` singleton — no direct
  env reads here.
* The first caller for a given device may also pass ``workspace_bytes``
  to :func:`get_global_fi_workspace`; once the buffer for that device
  exists, the parameter is ignored.

External pools
--------------
:func:`register_global_fi_workspace` lets a runtime hand in its own
:class:`torch.Tensor` (own allocator, pinned region, deterministic test
bytes, etc.) and the registry will treat it as the canonical buffer for
that device. The registry is keyed by ``(device.type, device.index)`` so
multi-GPU processes get one buffer per device rather than one total.
"""

from __future__ import annotations

import torch

from phyai.engine_config import get_engine_config
from phyai.utils.logging import get_logger


logger = get_logger(__name__)


#: Minimum flashinfer scratch per prefill kernel, in bytes. FA2's split-k
#: scratch for a short-query/long-KV joint attention (pi0.5's action expert,
#: head_dim 256) does not fit the 128 MiB engine default, and a wrapper that
#: runs FA2 is the only one that needs the extra room.
#:
#: A floor rather than a size: the engine-wide
#: :attr:`~phyai.engine_config.RuntimeConfig.flashinfer_workspace_bytes` still
#: wins when it is larger, so raising it globally keeps working.
PREFILL_WORKSPACE_FLOORS: dict[str, int] = {
    "fa2": 256 * 1024 * 1024,
}


def resolve_workspace_bytes(
    override: int | None = None,
    *,
    prefill_backend: str | None = None,
) -> int:
    """Resolve the flashinfer scratch size for one wrapper.

    Order of precedence: explicit ``override`` ->
    :class:`~phyai.engine_config.RuntimeConfig.flashinfer_workspace_bytes`
    on the :class:`EngineConfig` singleton (which has already absorbed
    any ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env override), then raised to
    ``PREFILL_WORKSPACE_FLOORS[prefill_backend]`` if that kernel needs more.
    Raises :class:`ValueError` for non-positive ``override``; the engine config
    is validated at construction time so the singleton value is always
    a positive int.

    ``prefill_backend`` is the kernel this wrapper actually resolved to, which
    the kernel catalog decided. Sizing here rather than on the engine config is
    what lets the kernel choice be per-call-site: the engine cannot know, at
    config time, which sites will end up on FA2.
    """
    if override is not None:
        if override <= 0:
            raise ValueError(f"workspace_bytes={override} must be positive.")
        base = override
    else:
        base = get_engine_config().runtime.flashinfer_workspace_bytes
    if prefill_backend is None:
        return base
    return max(base, PREFILL_WORKSPACE_FLOORS.get(prefill_backend, 0))


# Process-global flashinfer scratch. Keyed on
# ``(device.type, device.index)`` so a multi-GPU process gets one
# buffer per device rather than one buffer total.
_global_fi_workspaces: dict[tuple[str, int | None], torch.Tensor] = {}


def _device_key(device: torch.device | str) -> tuple[str, int | None]:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return (dev.type, dev.index)


def get_global_fi_workspace(
    device: torch.device | str,
    *,
    workspace_bytes: int | None = None,
    prefill_backend: str | None = None,
) -> torch.Tensor:
    """Get-or-create the process-global flashinfer scratch on ``device``.

    Allocated lazily on first call for each device. Size comes from
    ``workspace_bytes`` if given, else
    :attr:`~phyai.engine_config.RuntimeConfig.flashinfer_workspace_bytes`
    on the :class:`EngineConfig` singleton (which absorbs the
    ``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env override), raised to
    ``prefill_backend``'s entry in :data:`PREFILL_WORKSPACE_FLOORS`.

    ``prefill_backend`` is the kernel the *calling* wrapper resolved to. Since
    the buffer is shared, whichever wrapper is built first would otherwise fix
    the size for everyone — and build order does not follow need. pi0.5 builds
    vision, then the LLM prefix, then the FA2 action expert last, so a
    first-allocation-wins buffer would hand the one wrapper that needs 256 MiB
    a 128 MiB scratch.

    So a request for more than the existing buffer **grows** it: later callers
    get the larger tensor while wrappers already holding the smaller one keep a
    buffer that was, by construction, big enough for them. Both stay resident
    until those wrappers are released, which is why the growth is logged rather
    than silent.

    ``workspace_bytes`` only applies while it is the largest request seen for
    ``device``; it is *not* a per-instance override. To swap in your own
    pre-allocated tensor, use :func:`register_global_fi_workspace`.
    """
    key = _device_key(device)
    needed = resolve_workspace_bytes(workspace_bytes, prefill_backend=prefill_backend)
    ws = _global_fi_workspaces.get(key)
    if ws is not None and ws.numel() >= needed:
        return ws

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if ws is not None:
        # INFO, not a warning: for a model whose policy pins FA2 on a site that
        # is built last (pi0.5's action expert), this is the expected path, not
        # a misconfiguration. It is worth one line because it explains a memory
        # figure that would otherwise look unaccounted for.
        logger.info_once(
            "flashinfer scratch on %s grown %d -> %d bytes for prefill_backend=%s. "
            "Wrappers built earlier keep the smaller buffer (which was large "
            "enough for them), so both are resident until they are released. "
            "Set RuntimeConfig.flashinfer_workspace_bytes to the larger value "
            "to allocate once instead.",
            dev,
            ws.numel(),
            needed,
            prefill_backend,
        )
    ws = torch.empty(needed, dtype=torch.uint8, device=dev)
    _global_fi_workspaces[key] = ws
    return ws


def register_global_fi_workspace(
    device: torch.device | str, workspace: torch.Tensor
) -> None:
    """Inject a pre-allocated tensor as the global scratch for ``device``.

    Useful when the runtime owns the GPU memory pool itself (custom
    allocator, pinned scratch shared with another subsystem, or a
    deterministic-bytes test harness). Replaces any previous binding for
    ``device``. The tensor must be 1-D ``uint8`` and live on a device
    matching ``device``.
    """
    if workspace.dtype != torch.uint8 or workspace.ndim != 1:
        raise ValueError(
            f"workspace must be a 1-D uint8 tensor, got "
            f"shape={tuple(workspace.shape)}, dtype={workspace.dtype}."
        )
    key = _device_key(device)
    if (workspace.device.type, workspace.device.index) != key:
        raise ValueError(
            f"workspace.device={workspace.device} does not match device="
            f"{torch.device(device)}."
        )
    _global_fi_workspaces[key] = workspace


def release_global_fi_workspaces() -> None:
    """Drop every process-global FlashInfer workspace reference.

    Called by ``Engine.close``. Wrappers still holding the tensor keep it
    alive until they are collected; this only releases the registry's
    reference so a closed engine stops pinning device memory.
    """

    _global_fi_workspaces.clear()


def _reset_global_fi_workspaces() -> None:
    """Drop the global workspace registry. Tests only."""
    _global_fi_workspaces.clear()


__all__ = [
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "PREFILL_WORKSPACE_FLOORS",
    "release_global_fi_workspaces",
    "resolve_workspace_bytes",
]
