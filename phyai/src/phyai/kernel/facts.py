"""Define the fact vocabulary used by kernel predicates and policies."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Callable, Iterable
from difflib import get_close_matches
from dataclasses import field, dataclass

from phyai.kernel.types import Arch, dtype_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phyai.kernel.types import KernelQuery, DeviceProfile
    from phyai.kernel.opspec import OpSpec
    from phyai.kernel.predicate import Predicate


#: Sentinel for a fact that was not supplied.
MISSING = object()


class ParensError(TypeError):
    """Raised when a fact or predicate is coerced to a boolean."""


def no_truth_value(what: str) -> Callable[[Any], bool]:
    def __bool__(self: Any) -> bool:
        raise ParensError(
            f"{what} has no truth value. Operator precedence bites here: "
            f"write `(shape.K >= 16) & (quant.format == 'bf16')`, not "
            f"`shape.K >= 16 & quant.format == 'bf16'`. To test a boolean "
            f"fact, use it directly: `implies(attrs.bias, ...)`."
        )

    return __bool__


class FactKind(Enum):
    """Describe how a fact value is normalized and compared."""

    INT = "int"
    STR = "str"
    ARCH = "arch"
    DTYPE = "dtype"
    BOOL = "bool"
    STRSET = "strset"
    ANY = "any"

    @property
    def ordered(self) -> bool:
        return self is FactKind.INT

    def normalize(self, value: object) -> object:
        """Canonicalize a value or literal so both sides compare equal."""

        if value is None:
            return None
        if self is FactKind.INT:
            return int(value)
        if self is FactKind.BOOL:
            if isinstance(value, str):
                return value.strip().lower() not in {"0", "false", "no", "off", ""}
            return bool(value)
        if self is FactKind.DTYPE:
            return dtype_name(value)
        if self is FactKind.ARCH:
            # Store architecture facts as canonical names for serialization.
            return Arch.parse(value).name
        if self is FactKind.STR:
            return str(getattr(value, "value", value)).lower()
        if self is FactKind.STRSET:
            if isinstance(value, str):
                return frozenset({value.lower()})
            return frozenset(str(item).lower() for item in value)
        return value


_predicate_module = None


def predicate_nodes():
    """Return the lazily imported predicate module."""

    global _predicate_module
    if _predicate_module is None:
        from phyai.kernel import predicate

        _predicate_module = predicate
    return _predicate_module


class Fact:
    """Represent a named fact whose operators build predicates."""

    __slots__ = ("path", "kind")

    def __init__(self, path: str, kind: FactKind = FactKind.ANY) -> None:
        self.path = path
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Fact({self.path!r}, {self.kind.value})"

    def __hash__(self) -> int:
        return hash((self.path, self.kind))

    __bool__ = no_truth_value("a fact")

    def __eq__(self, other: object) -> "Predicate":  # type: ignore[override]
        return predicate_nodes().Cmp(self, "==", other)

    def __ne__(self, other: object) -> "Predicate":  # type: ignore[override]
        return predicate_nodes().Cmp(self, "!=", other)

    def __lt__(self, other: object) -> "Predicate":
        return predicate_nodes().Cmp(self, "<", other)

    def __le__(self, other: object) -> "Predicate":
        return predicate_nodes().Cmp(self, "<=", other)

    def __gt__(self, other: object) -> "Predicate":
        return predicate_nodes().Cmp(self, ">", other)

    def __ge__(self, other: object) -> "Predicate":
        return predicate_nodes().Cmp(self, ">=", other)

    def __mod__(self, modulus: int) -> "ModExpr":
        return ModExpr(self, int(modulus))

    def in_(self, values: Iterable[object]) -> "Predicate":
        return predicate_nodes().In(self, values)

    def at_least(self, bound: object) -> "Predicate":
        """Build a same-series architecture lower-bound predicate."""

        return predicate_nodes().ArchAtLeast(self, bound)

    def family_in(self, families: Iterable[object]) -> "Predicate":
        """Build an architecture-family membership predicate."""

        return predicate_nodes().ArchFamilyIn(self, families)

    def between(self, low: int, high: int) -> "Predicate":
        n = predicate_nodes()
        return n.And((n.Cmp(self, ">=", low), n.Cmp(self, "<=", high)))

    def has(self, value: object) -> "Predicate":
        """Set membership for ``STRSET`` facts (tags)."""

        return predicate_nodes().SetHas(self, value)

    def intersects(self, values: Iterable[object]) -> "Predicate":
        return predicate_nodes().SetIntersects(self, values)

    def is_none(self) -> "Predicate":
        return predicate_nodes().IsNone(self)

    def is_set(self) -> "Predicate":
        return predicate_nodes().IsSet(self)

    def as_predicate(self) -> "Predicate":
        """Use a boolean fact directly, e.g. ``implies(attrs.bias, ...)``."""

        return predicate_nodes().IsTruthy(self)


class ModExpr:
    """Represent an intermediate modulo expression."""

    __slots__ = ("fact", "modulus")

    def __init__(self, fact: Fact, modulus: int) -> None:
        if modulus <= 0:
            raise ValueError(f"modulus must be positive, got {modulus}")
        self.fact = fact
        self.modulus = modulus

    __bool__ = no_truth_value("a modulo expression")

    def __eq__(self, remainder: object) -> "Predicate":  # type: ignore[override]
        return predicate_nodes().Mod(  # type: ignore[arg-type]
            self.fact, self.modulus, int(remainder)
        )

    def __hash__(self) -> int:
        return hash((self.fact, self.modulus))


@dataclass(frozen=True)
class Facts:
    """Store normalized values and optional paths for one kernel call."""

    values: Mapping[str, object] = field(default_factory=dict)
    optional: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "optional", frozenset(self.optional or ()))

    def lookup(self, path: str) -> object:
        """Return the raw value, or ``MISSING`` when never provided."""

        return self.values.get(path, MISSING)

    def is_optional(self, path: str) -> bool:
        return path in self.optional


class DeviceNamespace:
    """Expose accelerator facts derived from a device profile."""

    #: Canonical device vendor name (nvidia / amd / ascend / cpu / ...).
    vendor = Fact("device.vendor", FactKind.STR)

    #: Canonical architecture name.
    arch = Fact("device.arch", FactKind.ARCH)


class QuantNamespace:
    """Expose physical quantization facts."""

    #: Exact physical storage format.
    format = Fact("quant.format", FactKind.STR)

    layout = Fact("quant.layout", FactKind.STR)
    granularity = Fact("quant.granularity", FactKind.STR)

    #: Quantization block K dimension.
    block_k = Fact("quant.block_k", FactKind.INT)

    @staticmethod
    def field(name: str) -> Fact:
        """Escape hatch for a scheme-specific field; equality/membership only."""

        return Fact(f"quant.fields.{name}", FactKind.ANY)


class ModelNamespace:
    """Expose model metadata facts supplied by the engine."""

    family = Fact("model.family", FactKind.STR)
    tags = Fact("model.tags", FactKind.STRSET)


class LibNamespace:
    """Build facts for third-party library availability."""

    @staticmethod
    def has(name: str) -> "Predicate":
        return Fact(f"lib.{name}", FactKind.BOOL).as_predicate()


class OpScopedNamespace:
    """Create operation-scoped facts on attribute access."""

    __slots__ = ("prefix", "kind")

    def __init__(self, prefix: str, kind: FactKind) -> None:
        self.prefix = prefix
        self.kind = kind

    def __getattr__(self, name: str) -> Fact:
        if name.startswith("_"):
            # Preserve normal behavior for Python protocol attributes.
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, name: str) -> Fact:
        return Fact(f"{self.prefix}.{name}", self.kind)


# Shape dimensions are integers.
shape = OpScopedNamespace("shape", FactKind.INT)

# Dtype roles use canonical dtype names.
dtype = OpScopedNamespace("dtype", FactKind.DTYPE)

# Attributes accept heterogeneous values.
attrs = OpScopedNamespace("attrs", FactKind.ANY)


device = DeviceNamespace()
quant = QuantNamespace()
model = ModelNamespace()
lib = LibNamespace()

#: Top-level facts that are not part of any namespace. ``role`` is the
#: policy-visible half of a call site's identity (see the A/B recipes in
#: examples/configs/kernel_policy.example.yaml); ``mode`` is matched by capability
#: gating in the selector, not by policies, so it has no Fact object.
op = Fact("op", FactKind.STR)
role = Fact("role", FactKind.STR)


#: Namespace prefixes whose members are declared per operation.
OP_SCOPED_PREFIXES = frozenset({"shape", "dtype", "attrs"})


def suggest_name(path: str, known: Iterable[str]) -> str:
    """Return a did-you-mean hint for an unknown name, or an empty string."""

    options = sorted(known)
    folded = {item.lower(): item for item in options}
    exact = folded.get(path.lower())
    if exact is not None:
        return f"; did you mean {exact!r}? (names are case-sensitive)"
    hint = get_close_matches(path, options, n=1)
    return f"; did you mean {hint[0]!r}?" if hint else ""


#: Every fact path this module declares globally, for policy-load validation.
GLOBAL_FACT_PATHS = frozenset(
    {"op", "role"}
    | {
        value.path
        for namespace in (DeviceNamespace, QuantNamespace, ModelNamespace)
        for value in vars(namespace).values()
        if isinstance(value, Fact)
    }
)


def device_facts(profile: "DeviceProfile") -> dict[str, object]:
    """Return facts derived from a device profile."""

    return {
        "device.vendor": profile.vendor,
        "device.arch": profile.arch,
    }


def facts_from_query(
    query: "KernelQuery",
    spec: "OpSpec",
    *,
    libraries: Mapping[str, bool] | None = None,
) -> Facts:
    """Flatten a kernel query into normalized fact values."""

    values: dict[str, object] = {
        "op": query.op,
        "role": query.role,
        "mode": query.mode.value,
        "model.family": query.model.family,
        "model.tags": frozenset(query.model.tags),
    }
    values.update(device_facts(query.device))

    signature = query.quant
    values.update(
        {
            "quant.format": signature.format if signature else None,
            "quant.layout": signature.layout if signature else None,
            "quant.granularity": signature.granularity if signature else None,
            "quant.block_k": None,
        }
    )
    if signature is not None:
        if signature.block_shape is not None:
            block = tuple(signature.block_shape)
            values["quant.block_k"] = block[1] if len(block) > 1 else None
        fields = dict(signature.fields)
        values.update({f"quant.fields.{key}": item for key, item in fields.items()})

    values.update({f"shape.{name}": item for name, item in query.shape.dims})
    values.update({f"dtype.{name}": item for name, item in query.dtype})

    # Query attributes override attributes attached to shape facts.
    merged_attrs: dict[str, object] = dict(query.shape.attrs)
    merged_attrs.update(dict(query.attrs))
    values.update({f"attrs.{name}": item for name, item in merged_attrs.items()})

    values.update(libraries or {})

    return Facts(values=values, optional=spec.optional_paths)


__all__ = [
    "DeviceNamespace",
    "Fact",
    "FactKind",
    "Facts",
    "MISSING",
    "ModExpr",
    "ParensError",
    "GLOBAL_FACT_PATHS",
    "LibNamespace",
    "ModelNamespace",
    "OP_SCOPED_PREFIXES",
    "OpScopedNamespace",
    "QuantNamespace",
    "attrs",
    "device",
    "device_facts",
    "dtype",
    "facts_from_query",
    "lib",
    "model",
    "no_truth_value",
    "op",
    "predicate_nodes",
    "quant",
    "role",
    "shape",
    "suggest_name",
]
