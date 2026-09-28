"""Registry of backends + selection Policy.

Each backend is registered once with optional ``prefer_for`` hints
(per-op override). The default policy uses ``prefer_for`` first, then
registration order, then a capture-safety filter. There are no numeric
scores — backends are ordered by registration intent rather than tuned
priority numbers.
"""

from __future__ import annotations

from typing import Final, Iterable, Protocol, runtime_checkable

import torch

from phyai.parallel.backend import Backend, Op, Topology
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import Mode


# The fallback guarantee is probed under the least favourable placement a
# group can have: spread over hosts, no NVLink. A backend that only serves
# NVLink islands or a size whitelist declines this by design, so it can never
# be mistaken for the universal fallback, whatever machine runs the check.
WORST_CASE_TOPOLOGY: Final[Topology] = Topology(
    is_full_nvlink=False, is_single_node=False, n_nodes=2, n_gpus_per_node=1
)


class Registry:
    """Process-level registry. Built once at init time."""

    def __init__(self) -> None:
        self._backends: list[Backend] = []
        self._prefer: dict[Op, list[str]] = {}

    def register(
        self,
        backend: Backend,
        *,
        prefer_for: set[Op] | None = None,
    ) -> None:
        self._backends.append(backend)
        if prefer_for:
            for op in prefer_for:
                self._prefer.setdefault(op, []).append(backend.name)

    def candidates(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> list[Backend]:
        out: list[Backend] = []
        prefer_names = self._prefer.get(op, [])

        # 1. Preferred backends in declaration order
        for name in prefer_names:
            for b in self._backends:
                if b.name != name:
                    continue
                if mode == Mode.GRAPH_CAPTURING and not b.supports_capture():
                    continue
                if b.can_handle(
                    op=op,
                    mode=mode,
                    nbytes=nbytes,
                    dtype=dtype,
                    world_size=world_size,
                    topology=topology,
                    **extra,
                ):
                    out.append(b)
                break

        # 2. Remaining backends in registration order
        for b in self._backends:
            if b.name in prefer_names:
                continue
            if mode == Mode.GRAPH_CAPTURING and not b.supports_capture():
                continue
            if b.can_handle(
                op=op,
                mode=mode,
                nbytes=nbytes,
                dtype=dtype,
                world_size=world_size,
                topology=topology,
                **extra,
            ):
                out.append(b)
        return out

    def has(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> bool:
        return bool(
            self.candidates(
                op=op,
                mode=mode,
                nbytes=nbytes,
                dtype=dtype,
                world_size=world_size,
                topology=topology,
                **extra,
            )
        )

    def all(self) -> list[Backend]:
        return list(self._backends)

    def validate(self, *, group_sizes: Iterable[int]) -> None:
        """Assert that a universal fallback exists for every common op.

        Two checks: every ``prefer_for`` name is registered, and for each
        common op and mode at least one backend accepts
        :data:`WORST_CASE_TOPOLOGY` at every multi-rank group size in
        ``group_sizes`` (the sizes actually present in the mesh). Sizes are
        probed one by one because size whitelists are not monotonic: a
        backend may take 8 ranks and refuse 16.

        Probes with ``pg=None`` so backends fall through their permissive
        probe-time path. ``Mode.GRAPH_CAPTURING`` is only checked when at
        least one registered backend supports capture (gloo-only setups
        legitimately have no capture coverage).
        """
        names = {b.name for b in self._backends}
        for op, prefs in self._prefer.items():
            for n in prefs:
                if n not in names:
                    raise NoBackendError(
                        f"prefer_for[{op.value}]={n!r} but no backend with "
                        f"that name is registered (have: {sorted(names)})"
                    )

        has_capture_backend = any(b.supports_capture() for b in self._backends)
        modes: list[Mode] = [Mode.EAGER]
        if has_capture_backend:
            modes.append(Mode.GRAPH_CAPTURING)

        sizes = sorted({int(size) for size in group_sizes if size > 1})
        for op in (
            Op.ALL_REDUCE,
            Op.ALL_GATHER,
            Op.REDUCE_SCATTER,
            Op.BROADCAST,
            Op.ALL_TO_ALL,
            Op.SEND,
            Op.RECV,
        ):
            for mode in modes:
                for size in sizes:
                    if not self.has(
                        op=op,
                        mode=mode,
                        nbytes=1024,
                        dtype=torch.bfloat16,
                        world_size=size,
                        topology=WORST_CASE_TOPOLOGY,
                    ):
                        raise NoBackendError(
                            f"no backend handles op={op.value} mode={mode.value} "
                            f"for a {size}-rank group under the worst-case "
                            "topology (a universal fallback is required)"
                        )


@runtime_checkable
class Policy(Protocol):
    def select(self, candidates: list[Backend]) -> Backend: ...


class DefaultPolicy:
    """Pick the first candidate. Registry has already ordered them by
    (preferred-for-this-op, registration-order)."""

    def select(self, candidates: list[Backend]) -> Backend:
        if not candidates:
            raise NoBackendError("no candidate backend")
        return candidates[0]


class ForcedPolicy:
    """Honor PHYAI_FORCE_COLLECTIVE_BACKEND=<name> if set; otherwise fall back."""

    def __init__(self, name: str, fallback: Policy = DefaultPolicy()) -> None:
        self.name = name
        self.fallback = fallback

    def select(self, candidates: list[Backend]) -> Backend:
        for b in candidates:
            if b.name == self.name:
                return b
        return self.fallback.select(candidates)
