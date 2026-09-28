"""Register kernel operations and implementations."""

from __future__ import annotations

import json
import hashlib
from typing import Mapping, Iterable

from phyai.kernel.opspec import Impl, OpSpec


class UnknownOperationError(KeyError):
    """Raised when an operation is not registered."""


class UnknownKernelError(KeyError):
    """Raised when a kernel ID is not registered."""


class Catalog:
    """Store operations and implementations in deterministic order."""

    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}
        self._impls: dict[str, Impl] = {}
        # Cache the catalog fingerprint until registration changes it.
        self._version: str | None = None

    def register_op(self, spec: OpSpec) -> OpSpec:
        existing = self._ops.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(
                f"operation {spec.name!r} is already registered with a different schema"
            )
        self._ops[spec.name] = spec
        self._version = None
        return spec

    def register(self, impl: Impl) -> Impl:
        """Add an implementation, validating it against its operation."""

        spec = self._ops.get(impl.op)
        if spec is None:
            raise UnknownOperationError(
                f"implementation {impl.kernel_id!r} targets unregistered "
                f"operation {impl.op!r}; register the OpSpec first "
                f"(known: {sorted(self._ops)})"
            )
        existing = self._impls.get(impl.kernel_id)
        if existing is not None and existing is not impl:
            raise ValueError(f"duplicate kernel id {impl.kernel_id!r}")
        impl.check_against(spec)
        self._impls[impl.kernel_id] = impl
        self._version = None
        return impl

    def register_many(self, impls: Iterable[Impl]) -> None:
        for impl in impls:
            self.register(impl)

    def op(self, name: str) -> OpSpec:
        try:
            return self._ops[str(name).lower()]
        except KeyError as exc:
            raise UnknownOperationError(
                f"unknown operation {name!r} (known: {sorted(self._ops)})"
            ) from exc

    def maybe_op(self, name: str) -> OpSpec | None:
        return self._ops.get(str(name).lower())

    def ops(self) -> tuple[OpSpec, ...]:
        return tuple(self._ops[name] for name in sorted(self._ops))

    def get(self, kernel_id: str) -> Impl:
        try:
            return self._impls[str(kernel_id).lower()]
        except KeyError as exc:
            raise UnknownKernelError(f"unknown kernel {kernel_id!r}") from exc

    def maybe_get(self, kernel_id: str) -> Impl | None:
        return self._impls.get(str(kernel_id).lower())

    def impls(self, op: str | None = None) -> tuple[Impl, ...]:
        """Return implementations ordered by priority and kernel ID."""

        values = tuple(self._impls.values())
        if op is not None:
            name = str(op).lower()
            values = tuple(item for item in values if item.op == name)
        return tuple(sorted(values, key=lambda item: (-item.priority, item.kernel_id)))

    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._impls))

    def match_ids(self, pattern: str) -> tuple[str, ...]:
        """Expand a ``restrict_to`` glob over kernel ids."""

        from fnmatch import fnmatchcase

        text = str(pattern).lower()
        return tuple(
            kernel_id
            for kernel_id in sorted(self._impls)
            if fnmatchcase(kernel_id, text)
        )

    def backends(self, op: str | None = None) -> tuple[str, ...]:
        """Return backend names registered for an operation."""

        return tuple(
            sorted({item.kernel_id.partition(".")[0] for item in self.impls(op)})
        )

    def ids_for_backend(self, op: str, backend: str) -> tuple[str, ...]:
        """Return a backend's kernel IDs in priority order."""

        wanted = str(backend).strip().lower().replace("-", "_")
        return tuple(
            item.kernel_id
            for item in self.impls(op)
            if item.kernel_id.partition(".")[0] == wanted
        )

    def libraries(self, op: str | None = None) -> frozenset[str]:
        """Return import names used by implementation capabilities."""

        return frozenset().union(
            *(item.libraries for item in self.impls(op)), frozenset()
        )

    def coverage_gaps(self) -> dict[str, str]:
        """Return operations missing required reference implementations."""

        gaps: dict[str, str] = {}
        for spec in self.ops():
            rows = self.impls(spec.name)
            if not rows:
                gaps[spec.name] = "no implementations registered"
            elif spec.requires_reference and not any(item.reference for item in rows):
                gaps[spec.name] = (
                    "no reference implementation; candidates are "
                    + ", ".join(item.kernel_id for item in rows)
                )
        return gaps

    @property
    def version(self) -> str:
        """Return the stable fingerprint used by selection caches."""

        if self._version is not None:
            return self._version

        payload = [
            {
                "id": item.kernel_id,
                "op": item.op,
                "priority": item.priority,
                "reference": item.reference,
                "capture_safe": item.capture_safe,
                "when": item.when.render(),
                "params": {
                    name: (contract.rule.value, contract.dtype)
                    for name, contract in sorted(item.params.items())
                },
            }
            for item in self.impls()
        ]
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        self._version = hashlib.sha256(encoded).hexdigest()[:16]
        return self._version

    def manifest(self) -> dict[str, object]:
        """Return a manifest of registered operations and implementations."""

        return {
            "version": self.version,
            "operations": [
                {
                    "name": spec.name,
                    "signature": spec.signature,
                    "dims": list(spec.dims),
                    "dtypes": list(spec.dtypes),
                    "optional_dtypes": list(spec.optional_dtypes),
                    "attributes": list(spec.attributes),
                    "params": list(spec.params),
                    "requires_reference": spec.requires_reference,
                    "returns": spec.returns.kind,
                    "implementations": [
                        {
                            "id": item.kernel_id,
                            "priority": item.priority,
                            "reference": item.reference,
                            "capture_safe": item.capture_safe,
                            "when": item.when.render(),
                            "libraries": sorted(item.libraries),
                            "metadata": dict(item.metadata),
                        }
                        for item in self.impls(spec.name)
                    ],
                }
                for spec in self.ops()
            ],
        }

    def describe(self) -> str:
        """Return one summary line per implementation."""

        width = max((len(i.kernel_id) for i in self.impls()), default=0)
        return "\n".join(
            f"{item.kernel_id:<{width}}  {item.op:<22}  {item.when.render()}"
            for item in self.impls()
        )


def build_catalog() -> Catalog:
    """Populate and return a catalog from the built-in operation modules."""

    from phyai.kernel.ops import populate

    return populate(Catalog())


__all__ = [
    "Catalog",
    "UnknownKernelError",
    "UnknownOperationError",
    "build_catalog",
]
