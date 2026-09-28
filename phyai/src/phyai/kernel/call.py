"""Build kernel queries and select implementations at call sites."""

from __future__ import annotations

import weakref
from typing import Any, Mapping, Sequence

import torch

from phyai.env import envs
from phyai.kernel.types import KernelMode, KernelQuery
from phyai.utils.logging import get_logger
from phyai.kernel.selector import Selection
from phyai.kernel.bootstrap import get_kernel_selector

logger = get_logger(__name__)


def select(
    op: str,
    *,
    role: str = "",
    device: torch.device | str | int | None = None,
    dtype: Mapping[str, Any] | None = None,
    quant: Any = None,
    shape: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    mode: KernelMode | str | None = None,
    prefer: Sequence[str] = (),
) -> Selection:
    """Select an implementation for one call."""

    query = KernelQuery.build(
        op,
        role=role,
        device=device,
        dtype=dict(dtype or {}),
        quant=quant,
        shape=dict(shape or {}),
        attrs=dict(attrs or {}),
        mode=current_mode() if mode is None else mode,
    )
    return get_kernel_selector().select(query, prefer=tuple(prefer))


class CallSite:
    """Memoize selections for one operation and optionally freeze warmed choices."""

    __slots__ = (
        "op",
        "role",
        "prefer",
        "static_dims",
        "static_attrs",
        "_memo",
        "_key_paths",
        "_key_plan",
        "_fingerprint",
        "_frozen",
        # Support weak references from the process-wide registry.
        "__weakref__",
    )

    def __init__(
        self,
        op: str,
        *,
        role: str = "",
        prefer: Sequence[str] = (),
        dims: Mapping[str, int] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.op = str(op)
        self.role = str(role)
        self.prefer = tuple(prefer)
        # Static facts are omitted from per-call memo keys.
        self.static_dims: dict[str, int] = dict(dims or {})
        self.static_attrs: dict[str, Any] = dict(attrs or {})
        self._memo: dict[tuple, Selection] = {}
        self._key_paths: tuple[str, ...] = ()
        # Group key paths by namespace for direct mapping lookups.
        self._key_plan: tuple[tuple[str, ...], ...] = ((), (), (), ())
        self._fingerprint: tuple[object, ...] | None = None
        # Frozen selections are stored separately for each execution mode.
        self._frozen: dict[str, Selection] | None = None
        _call_sites.add(self)

    def key_paths(self) -> tuple[str, ...]:
        """Return fact paths required by the memo key."""

        selector = get_kernel_selector()
        # The inherited selector device and model context are part of cache
        # invalidation, as is the selector epoch (bumped by clear_cache).
        fingerprint = (
            selector.catalog.version,
            selector.policy.version,
            () if selector.device is None else selector.device.fingerprint,
            selector.model.fingerprint,
            selector.epoch,
        )
        if fingerprint != self._fingerprint:
            paths: set[str] = set()
            for impl in selector.catalog.impls(self.op):
                paths |= impl.when.facts_used()
            paths |= selector.policy.facts_used()
            # Library facts do not vary within a process.
            self._key_paths = tuple(
                sorted(path for path in paths if not path.startswith("lib."))
            )
            grouped: dict[str, list[str]] = {
                "dtype": [],
                "shape": [],
                "attrs": [],
                "quant": [],
            }
            for path in self._key_paths:
                namespace, _, name = path.partition(".")
                # Keep only namespaces that vary between calls at this site.
                if namespace in grouped:
                    grouped[namespace].append(name)
            self._key_plan = (
                tuple(grouped["dtype"]),
                tuple(grouped["shape"]),
                tuple(grouped["attrs"]),
                tuple(grouped["quant"]),
            )
            self._fingerprint = fingerprint
            self._memo.clear()
        return self._key_paths

    def select(
        self,
        *,
        device: torch.device | str | int | None = None,
        dtype: Mapping[str, Any] | None = None,
        quant: Any = None,
        dims: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        mode: KernelMode | str | None = None,
    ) -> Selection:
        """Select for one call, reusing the previous answer when it applies."""

        resolved_mode = current_mode() if mode is None else mode
        frozen = self._frozen
        if frozen is not None and not verify_frozen():
            hit = frozen.get(getattr(resolved_mode, "value", resolved_mode))
            if hit is not None:
                return hit
            # Unseen modes continue through normal memoized selection.

        shape = {**self.static_dims, **(dims or {})}
        merged_attrs = {**self.static_attrs, **(attrs or {})}

        self.key_paths()
        key = _memo_key(
            self._key_plan, device, dtype, quant, shape, merged_attrs, resolved_mode
        )
        hit = self._memo.get(key)
        if hit is not None:
            self._check_frozen(frozen, resolved_mode, hit)
            return hit

        selection = select(
            self.op,
            role=self.role,
            device=device,
            dtype=dtype,
            quant=quant,
            shape=shape,
            attrs=merged_attrs,
            mode=resolved_mode,
            prefer=self.prefer,
        )
        self._memo[key] = selection
        self._check_frozen(frozen, resolved_mode, selection)
        return selection

    def _check_frozen(self, frozen, mode: object, selection: Selection) -> None:
        """Validate a selection against the frozen choice for its mode."""

        if frozen is None:
            return
        expected = frozen.get(getattr(mode, "value", mode))
        if expected is None or expected is selection:
            return
        raise FrozenChoiceError(
            f"{self.op}: call site was frozen to {expected.kernel_id!r} in "
            f"{getattr(mode, 'value', mode)!r} mode but this call selects "
            f"{selection.kernel_id!r}; the site is not monomorphic in that mode "
            f"and must not be frozen"
        )

    def __call__(self, **kwargs: Any) -> Selection:
        return self.select(**kwargs)

    @property
    def frozen_choices(self) -> dict[str, Selection] | None:
        """Return frozen selections by execution mode."""

        return None if self._frozen is None else dict(self._frozen)

    def freeze(self) -> dict[str, Selection]:
        """Freeze each warmed execution mode to its single observed selection."""

        if self._frozen is not None:
            return dict(self._frozen)
        if not self._memo:
            raise FrozenChoiceError(
                f"{self.op}: nothing to freeze -- this call site has not run yet, so "
                f"warm it up (or capture a graph) before freezing"
            )
        by_mode: dict[str, dict[int, Selection]] = {}
        for key, selection in self._memo.items():
            # The memo key stores the execution mode first.
            by_mode.setdefault(str(key[0]), {})[id(selection)] = selection
        for mode_name, distinct in by_mode.items():
            if len(distinct) > 1:
                ids = sorted(sel.kernel_id for sel in distinct.values())
                raise FrozenChoiceError(
                    f"{self.op}: cannot freeze a polymorphic call site -- in "
                    f"{mode_name!r} mode it has selected {ids}"
                )
        self._frozen = {
            mode_name: next(iter(distinct.values()))
            for mode_name, distinct in by_mode.items()
        }
        return dict(self._frozen)

    def unfreeze(self) -> None:
        """Resume memo-key selection for this call site."""

        self._frozen = None


def _memo_key(
    plan: tuple[tuple[str, ...], ...],
    device: object,
    dtype: Mapping[str, Any] | None,
    quant: Any,
    shape: Mapping[str, Any],
    attrs: Mapping[str, Any],
    mode: object,
) -> tuple:
    """Build a memo key from the pre-grouped dynamic fact paths."""

    dtype_names, shape_names, attr_names, quant_names = plan
    parts: list[object] = [
        getattr(mode, "value", mode),
        getattr(device, "type", device),
        getattr(device, "index", None),
    ]
    if dtype_names:
        source = dtype or {}
        for name in dtype_names:
            parts.append(source.get(name))
    for name in shape_names:
        parts.append(shape.get(name))
    for name in attr_names:
        parts.append(attrs.get(name))
    for name in quant_names:
        parts.append(_quant_part(quant, name))
    return tuple(parts)


def _quant_part(quant: Any, name: str) -> object:
    if quant is None:
        return None
    if isinstance(quant, Mapping):
        return quant.get(name)
    value = getattr(quant, name, None)
    return tuple(value) if isinstance(value, list) else value


class FrozenChoiceError(RuntimeError):
    """Raised when a call site cannot use its frozen selection."""


_call_sites: "weakref.WeakSet[CallSite]" = weakref.WeakSet()


def get_call_sites() -> tuple[CallSite, ...]:
    """Return all live registered call sites."""

    return tuple(_call_sites)


def freeze_kernel_choices() -> dict[str, int]:
    """Freeze warmed monomorphic call sites and return outcome counts."""

    tally = {"frozen": 0, "polymorphic": 0, "cold": 0}
    for site in get_call_sites():
        if site.frozen_choices is not None:
            tally["frozen"] += 1
            continue
        try:
            site.freeze()
        except FrozenChoiceError:
            key = "cold" if not site._memo else "polymorphic"
            tally[key] += 1
        else:
            tally["frozen"] += 1
    logger.info_rank0(
        "kernel: froze %d call sites (%d polymorphic, %d never ran)",
        tally["frozen"],
        tally["polymorphic"],
        tally["cold"],
    )
    return tally


def unfreeze_kernel_choices() -> None:
    """Unfreeze every registered call site."""

    for site in get_call_sites():
        site.unfreeze()


_verify_frozen: bool | None = None


def verify_frozen() -> bool:
    """Return the cached frozen-choice verification setting."""

    global _verify_frozen
    if _verify_frozen is None:
        _verify_frozen = bool(envs.PHYAI_KERNEL_VERIFY_FROZEN.get())
    return _verify_frozen


def reset_verify_frozen() -> None:
    """Re-read ``PHYAI_KERNEL_VERIFY_FROZEN`` on the next call."""

    global _verify_frozen
    _verify_frozen = None


def backend_preference(op: str, backend: str | None) -> tuple[str, ...]:
    """Translate a backend hint into preferred kernel IDs."""

    if backend is None:
        return ()
    catalog = get_kernel_selector().catalog
    ids = catalog.ids_for_backend(op, backend)
    if not ids:
        available = catalog.backends(op)
        raise ValueError(
            f"unknown backend {backend!r} for {op!r}; available: {list(available)}"
        )
    return ids


def explain(op: str, **kwargs: Any):
    """Return the selection trace for one kernel call."""

    query = KernelQuery.build(
        op,
        role=kwargs.pop("role", ""),
        device=kwargs.pop("device", None),
        dtype=dict(kwargs.pop("dtype", None) or {}),
        quant=kwargs.pop("quant", None),
        shape=dict(kwargs.pop("shape", None) or {}),
        attrs=dict(kwargs.pop("attrs", None) or {}),
        mode=kwargs.pop("mode", None) or current_mode(),
    )
    return get_kernel_selector().explain(
        query,
        prefer=tuple(kwargs.pop("prefer", ()) or ()),
    )


_parallel_mode = None


def current_mode() -> KernelMode:
    """Return the ambient execution mode."""

    global _parallel_mode
    if _parallel_mode is None:
        from phyai.parallel.state import current_mode as parallel_mode

        _parallel_mode = parallel_mode
    return KernelMode.normalize(_parallel_mode())


def token_shape(tensor: torch.Tensor, **dims: int) -> dict[str, Any]:
    """Return standard shape facts for a token-major tensor."""

    if tensor.ndim == 0:
        tokens = 1
    else:
        last = int(tensor.shape[-1])
        tokens = int(tensor.numel() // max(last, 1))
    facts: dict[str, Any] = {"tokens": tokens, "M": tokens}
    facts.update(dims)
    return facts


def param_dtypes(
    op: str,
    *,
    activation: torch.dtype | str,
    known: Mapping[str, Any] | None = None,
    preferred: Mapping[str, str] | None = None,
    prefer: Sequence[str] = (),
) -> dict[str, str]:
    """Choose parameter dtypes from construction-time facts."""

    from phyai.kernel.types import dtype_name

    return get_kernel_selector().param_dtypes(
        op,
        activation=dtype_name(activation),
        known=known,
        preferred=preferred,
        prefer=tuple(prefer),
    )


def torch_dtype(name: str) -> torch.dtype:
    """Map a canonical dtype name back to a torch dtype for allocation."""

    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "uint8": torch.uint8,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"no torch dtype for {name!r}") from exc


__all__ = [
    "CallSite",
    "FrozenChoiceError",
    "backend_preference",
    "current_mode",
    "explain",
    "get_call_sites",
    "param_dtypes",
    "freeze_kernel_choices",
    "reset_verify_frozen",
    "select",
    "token_shape",
    "torch_dtype",
    "unfreeze_kernel_choices",
    "verify_frozen",
]
