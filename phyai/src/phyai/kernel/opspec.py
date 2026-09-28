"""Define operation schemas and kernel implementation contracts."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Mapping, Callable, Iterable, Sequence
from dataclasses import field, dataclass

from phyai.kernel.facts import OP_SCOPED_PREFIXES, FactKind, suggest_name
from phyai.kernel.predicate import Predicate


class ParamRule(Enum):
    """Describe an implementation's parameter dtype constraint."""

    #: Accept any supported floating dtype.
    ANY_FLOAT = "any_float"
    #: Require the activation dtype.
    MATCHES_ACTIVATION = "matches_activation"
    #: Require one fixed dtype.
    FIXED = "fixed"


@dataclass(frozen=True)
class ParamContract:
    """Store one implementation's parameter dtype requirement."""

    rule: ParamRule
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.rule is ParamRule.FIXED and not self.dtype:
            raise ValueError("ParamRule.FIXED requires a dtype")
        if self.rule is not ParamRule.FIXED and self.dtype is not None:
            raise ValueError(f"{self.rule} must not name a dtype")

    def allows(self, candidate: str, *, activation: str) -> bool:
        if self.rule is ParamRule.ANY_FLOAT:
            return candidate in {"bf16", "fp16", "fp32", "fp64"}
        if self.rule is ParamRule.MATCHES_ACTIVATION:
            return candidate == activation
        return candidate == self.dtype


def fixed(dtype: str) -> ParamContract:
    return ParamContract(ParamRule.FIXED, dtype)


def matches_activation() -> ParamContract:
    return ParamContract(ParamRule.MATCHES_ACTIVATION)


def any_float() -> ParamContract:
    return ParamContract(ParamRule.ANY_FLOAT)


@dataclass(frozen=True)
class Returns:
    """Describe the value returned by an implementation's ``prepare`` function."""

    kind: str = "callable"
    protocol: type | None = None
    constructed_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"callable", "instance"}:
            raise ValueError(f"unknown Returns kind {self.kind!r}")
        object.__setattr__(self, "constructed_with", tuple(self.constructed_with))

    @property
    def is_instance(self) -> bool:
        return self.kind == "instance"


def returns_callable() -> Returns:
    return Returns("callable")


def returns_instance(
    protocol: type, *, constructed_with: Iterable[str] = ()
) -> Returns:
    return Returns("instance", protocol, tuple(constructed_with))


@dataclass(frozen=True)
class OpSpec:
    """Store the fact schema and calling convention for one operation."""

    name: str

    #: Integer dimension names exposed under ``shape``.
    dims: tuple[str, ...] = ()

    #: Required dtype roles exposed under ``dtype``.
    dtypes: tuple[str, ...] = ()

    #: Dtype roles that may be absent from a query.
    optional_dtypes: tuple[str, ...] = ()

    #: Required attribute names exposed under ``attrs``.
    attributes: tuple[str, ...] = ()

    optional_attributes: tuple[str, ...] = ()

    #: Call signature shown in errors and manifests.
    signature: str = ""

    returns: Returns = field(default_factory=returns_callable)

    #: Parameters with implementation-specific allocation constraints.
    params: tuple[str, ...] = ()

    #: Whether the operation requires a reference implementation.
    requires_reference: bool = True

    #: Synthesize positional arguments for a prepared callable from a call's
    #: facts, so ``profile: autotune`` can measure the candidates at selection
    #: time: ``bench_args(facts, torch_device) -> tuple``. ``None`` (or a
    #: raise from the builder, e.g. for a quantized layout it cannot fake)
    #: means the call is not measurable and keeps the priority order.
    bench_args: Callable[..., tuple] | None = None

    doc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip().lower())
        if not self.name:
            raise ValueError("OpSpec.name must be non-empty")
        for attribute in (
            "dims",
            "dtypes",
            "optional_dtypes",
            "attributes",
            "optional_attributes",
            "params",
        ):
            object.__setattr__(self, attribute, tuple(getattr(self, attribute)))
        overlap = set(self.dtypes) & set(self.optional_dtypes)
        if overlap:
            raise ValueError(
                f"op {self.name!r}: dtype roles cannot be both required and "
                f"optional: {sorted(overlap)}"
            )
        overlap = set(self.attributes) & set(self.optional_attributes)
        if overlap:
            raise ValueError(
                f"op {self.name!r}: attributes cannot be both required and "
                f"optional: {sorted(overlap)}"
            )

    @property
    def optional_paths(self) -> frozenset[str]:
        """Fact paths an implementation may leave unconstrained."""

        return frozenset(
            [f"dtype.{name}" for name in self.optional_dtypes]
            + [f"attrs.{name}" for name in self.optional_attributes]
        )

    def known_paths(self) -> frozenset[str]:
        """Every op-scoped fact path this operation declares."""

        return frozenset(
            [f"shape.{name}" for name in self.dims]
            + [f"dtype.{name}" for name in (*self.dtypes, *self.optional_dtypes)]
            + [
                f"attrs.{name}"
                for name in (*self.attributes, *self.optional_attributes)
            ]
        )

    def kind_of(self, path: str) -> FactKind:
        prefix, _, name = path.partition(".")
        if prefix == "shape":
            return FactKind.INT
        if prefix == "dtype":
            return FactKind.DTYPE
        return FactKind.ANY

    def validate_paths(self, paths: Iterable[str], *, context: str) -> None:
        """Reject operation-scoped fact paths not declared by this schema."""

        known = self.known_paths()
        for path in sorted(paths):
            prefix = path.partition(".")[0]
            if prefix not in OP_SCOPED_PREFIXES:
                continue  # global facts are validated against facts.py
            if path in known:
                continue
            raise ValueError(
                f"{context}: op {self.name!r} has no fact {path!r}"
                f"{suggest_name(path, known)} (declared: {sorted(known)})"
            )


class Priority(IntEnum):
    """Define priority band starts for kernel implementations."""

    REFERENCE = 0
    GENERAL = 4
    OPTIMIZED = 8
    SPECIALIZED = 12
    PLUGIN = 16


#: Exclusive upper bound for implementation priorities.
PRIORITY_LIMIT = 20


def band_for(value: int) -> Priority:
    """Return the priority band containing a value."""

    return max((band for band in Priority if int(band) <= value), key=int)


def validate_priority(value: int | Priority) -> int:
    """Return a priority as an integer after validating its range."""

    number = int(value)
    if not 0 <= number < PRIORITY_LIMIT:
        names = ", ".join(f"{band.name}={int(band)}" for band in Priority)
        raise ValueError(
            f"priority must be in [0, {PRIORITY_LIMIT}), got {number}. Use a "
            f"Priority band ({names}), optionally with a small +offset."
        )
    return number


@dataclass(frozen=True)
class Impl:
    """Describe one executable implementation of an operation."""

    kernel_id: str
    op: str
    when: Predicate
    prepare: Callable[..., Any]

    #: Ordering among eligible candidates without a policy preference.
    priority: int = int(Priority.REFERENCE)

    #: Whether this implementation is a reference fallback.
    reference: bool = False

    #: Whether the implementation supports CUDA graph capture.
    capture_safe: bool = True

    #: Allocation constraints for the operation's declared parameters.
    params: Mapping[str, ParamContract] = field(default_factory=dict)

    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_id", str(self.kernel_id).strip().lower())
        object.__setattr__(self, "op", str(self.op).strip().lower())
        if not self.kernel_id:
            raise ValueError("Impl.kernel_id must be non-empty")
        object.__setattr__(self, "priority", validate_priority(self.priority))
        object.__setattr__(self, "reference", bool(self.reference))
        object.__setattr__(self, "capture_safe", bool(self.capture_safe))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def libraries(self) -> frozenset[str]:
        """Import names whose availability this implementation depends on."""

        return frozenset(
            path.partition(".")[2]
            for path in self.when.facts_used()
            if path.startswith("lib.")
        )

    def check_against(self, spec: OpSpec) -> None:
        """Validate this row's capability against the operation's schema."""

        if self.op != spec.name:
            raise ValueError(
                f"impl {self.kernel_id!r} declares op {self.op!r} but was "
                f"registered under {spec.name!r}"
            )
        spec.validate_paths(
            self.when.facts_used(), context=f"capability of {self.kernel_id!r}"
        )
        unknown_params = set(self.params) - set(spec.params)
        if unknown_params:
            raise ValueError(
                f"impl {self.kernel_id!r}: op {spec.name!r} has no parameter(s) "
                f"{sorted(unknown_params)} (declared: {list(spec.params)})"
            )
        self._reject_vacuous_capability(spec)

    def _reject_vacuous_capability(self, spec: OpSpec) -> None:
        """Reject capabilities that constrain only optional facts."""

        used = {
            path
            for path in self.when.facts_used()
            if path.partition(".")[0] in OP_SCOPED_PREFIXES
        }
        if not used:
            return
        if used <= spec.optional_paths:
            raise ValueError(
                f"impl {self.kernel_id!r} constrains only optional facts "
                f"{sorted(used)}, so omitting them all makes it unconditionally "
                f"eligible. Constrain a required fact, or declare one of these "
                f"required on op {spec.name!r}."
            )


def resolve_param_dtypes(
    spec: OpSpec,
    impls: Sequence[Impl],
    *,
    activation: str,
    preferred: Mapping[str, str] | None = None,
    prefer: Sequence[str] = (),
) -> dict[str, str]:
    """Choose parameter dtypes that keep the highest-ranked implementation viable."""

    boost = {str(item).lower() for item in prefer}
    chosen: dict[str, str] = {}
    for name in spec.params:
        want = (preferred or {}).get(name)
        # Prefer an explicit request, then the activation dtype, then the
        # supported floats from widest adoption to widest range.
        order = [want, activation, "bf16", "fp16", "fp32"]

        best: tuple[int, int] | None = None
        best_dtype: str | None = None
        for index, candidate in enumerate(order):
            if candidate is None:
                continue
            viable = [
                impl
                for impl in impls
                if name not in impl.params
                or impl.params[name].allows(candidate, activation=activation)
            ]
            if not viable:
                continue
            # Explicit kernel preferences outrank all built-in priority bands.
            rank = max(
                impl.priority + (PRIORITY_LIMIT if impl.kernel_id in boost else 0)
                for impl in viable
            )
            # Break equal implementation ranks by dtype preference order.
            score = (rank, -index)
            if best is None or score > best:
                best = score
                best_dtype = candidate

        if best_dtype is None:
            constraints = {
                impl.kernel_id: impl.params[name]
                for impl in impls
                if name in impl.params
            }
            raise ValueError(
                f"op {spec.name!r}: no dtype satisfies parameter {name!r} for "
                f"activation {activation!r}; constraints={constraints}"
            )
        chosen[name] = best_dtype
    return chosen


__all__ = [
    "PRIORITY_LIMIT",
    "Impl",
    "OpSpec",
    "ParamContract",
    "Priority",
    "ParamRule",
    "Returns",
    "any_float",
    "band_for",
    "fixed",
    "matches_activation",
    "resolve_param_dtypes",
    "returns_callable",
    "returns_instance",
    "validate_priority",
]
