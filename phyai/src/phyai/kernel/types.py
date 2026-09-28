"""Define immutable value types for kernel selection."""

from __future__ import annotations

import re
from enum import Enum
from typing import Mapping
from dataclasses import field, dataclass

import torch

from phyai.utils.vendors import VENDORS

_UNSET_DEVICE = object()


_ARCH_RE = re.compile(r"^([a-z]+)(\d+)([a-z0-9]*)$")

#: Parsers for architecture series with numeric version structure.
ARCH_SERIES_GRAMMAR = {
    vendor.series: vendor.split_version
    for vendor in VENDORS.values()
    if vendor.split_version is not None
}

#: Architecture series that support version ordering.
ORDERED_ARCH_SERIES = frozenset(
    vendor.series for vendor in VENDORS.values() if vendor.ordered
)

#: Major versions that plausibly exist per series, for typo detection.
SERIES_MAJOR_RANGES = {
    vendor.series: vendor.plausible_majors
    for vendor in VENDORS.values()
    if vendor.plausible_majors is not None
}


@dataclass(frozen=True)
class Arch:
    """Store the parsed components of an architecture name."""

    name: str
    series: str = ""
    major: int | None = None
    minor: int | None = None

    def __str__(self) -> str:
        return self.name

    @property
    def family(self) -> str | None:
        """Return the architecture series and major version."""

        return None if self.major is None else f"{self.series}{self.major}"

    @property
    def ordered(self) -> bool:
        """Return whether this architecture supports version ordering."""

        return self.series in ORDERED_ARCH_SERIES and self.major is not None

    def at_least(self, other: "Arch") -> bool:
        """Return whether this architecture meets a same-series lower bound."""

        if not self.ordered or self.series != other.series:
            return False
        return (self.major, self.minor or 0) >= (other.major, other.minor or 0)

    @classmethod
    def parse(cls, value: object) -> "Arch":
        """Parse an architecture name while accepting unknown formats."""

        name = str(value).lower()
        match = _ARCH_RE.match(name)
        if match is None:
            return cls(name=name)
        series, digits, suffix = match.groups()
        split = ARCH_SERIES_GRAMMAR.get(series)
        if split is None:
            return cls(name=name, series=series)
        parts = split(digits, suffix)
        if parts is None:
            return cls(name=name, series=series)
        return cls(name=name, series=series, major=parts[0], minor=parts[1])

    @classmethod
    def parse_bound(cls, value: object) -> "Arch":
        """Parse and validate an ordered architecture bound."""

        text = str(value).lower().strip()
        if not text:
            raise ValueError("architecture bound must not be empty")
        if text.isdigit() or text.lstrip("-").isdigit():
            raise ValueError(
                f"architecture bound {text!r} has no series prefix, so it "
                f"cannot be ordered -- write 'sm{text}'"
            )
        arch = cls.parse(text)
        if arch.series not in ORDERED_ARCH_SERIES:
            known = ", ".join(sorted(ORDERED_ARCH_SERIES))
            raise ValueError(
                f"architecture bound {text!r} is not in an ordered series "
                f"(ordered: {known}); equality is the only meaningful test "
                f"for it"
            )
        if arch.major is None:
            raise ValueError(
                f"architecture bound {text!r} names a generation, not a "
                f"version, so it cannot be ordered -- write '{text}0' for the "
                f"generation's first version, or test the generation itself "
                f"with family_in({text!r}) in Python / an enumeration of its "
                f"members in YAML"
            )
        majors = SERIES_MAJOR_RANGES.get(arch.series)
        if majors is not None and arch.major not in majors:
            raise ValueError(
                f"architecture bound {text!r} parses as major {arch.major}, which "
                f"is not a plausible compute capability -- 'sm10' means major 1 "
                f"and would hold everywhere; did you mean 'sm100'?"
            )
        return arch

    @classmethod
    def parse_family(cls, value: object) -> str:
        """Parse and validate an architecture-family literal."""

        text = str(value).lower().strip()
        match = _ARCH_RE.match(text)
        if match is None or match.group(3):
            raise ValueError(
                f"architecture generation {text!r} must be a series followed by a "
                f"major version, e.g. 'sm9' or 'sm10'"
            )
        series, digits, _ = match.groups()
        if series not in ARCH_SERIES_GRAMMAR:
            known = ", ".join(sorted(ARCH_SERIES_GRAMMAR))
            raise ValueError(
                f"architecture generation {text!r} names an unknown series "
                f"{series!r} (known: {known})"
            )
        major = int(digits)
        majors = SERIES_MAJOR_RANGES.get(series)
        if majors is not None and major not in majors:
            raise ValueError(
                f"architecture generation {text!r} parses as major {major}, which "
                f"is not a plausible compute capability -- did you mean "
                f"'{series}{digits[:2]}' or the full name '{series}{digits}'?"
            )
        return f"{series}{major}"


def arch_at_least(profile: "DeviceProfile", bound: str) -> bool:
    """Return whether a device profile meets an architecture lower bound."""

    parts = profile.arch_parts
    return parts is not None and parts.at_least(Arch.parse_bound(bound))


class KernelMode(str, Enum):
    """Represent an execution mode used by capability matching."""

    EAGER = "eager"
    CAPTURE = "capture"
    REPLAY = "replay"

    @classmethod
    def normalize(cls, value: object) -> "KernelMode":
        if isinstance(value, cls):
            return value
        # ``getattr`` unwraps ``phyai.parallel.state.Mode``, the one enum
        # that actually reaches this boundary.
        text = str(getattr(value, "value", value)).lower().replace("_", "-")
        if text in {"graph-capturing", "capture"}:
            return cls.CAPTURE
        if text == "replay":
            return cls.REPLAY
        return cls.EAGER


#: Canonical dtype-name aliases shared by torch and string spellings.
_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "float8_e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e4m3fnuz": "fp8_e4m3fnuz",
    "float8_e5m2": "fp8_e5m2",
    "float8_e5m2fnuz": "fp8_e5m2fnuz",
    # Bare FP8 names normalize to E4M3.
    "fp8": "fp8_e4m3",
    "float8": "fp8_e4m3",
}


def dtype_name(value: object) -> str:
    """Return a stable lowercase dtype name for torch or string values."""

    if isinstance(value, torch.dtype):
        text = str(value).removeprefix("torch.")
    else:
        text = str(value).lower().replace("torch.", "")
    return _DTYPE_ALIASES.get(text, text)


def _freeze(value: object) -> object:
    """Convert common YAML/runtime values into hashable deterministic values."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass(frozen=True)
class DeviceProfile:
    """Store a canonical snapshot of the device kernel selection runs on."""

    vendor: str = "cpu"
    arch: str | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        vendor = "cpu" if self.vendor is None else str(self.vendor).lower()
        object.__setattr__(self, "vendor", vendor)
        if self.index is not None:
            object.__setattr__(self, "index", int(self.index))
        if self.arch is not None:
            arch = str(self.arch).lower()
            if arch.isdigit():
                raise ValueError(
                    f"architecture {arch!r} has no series prefix; write the "
                    f"full name, e.g. 'sm{arch}' or 'gfx{arch}'"
                )
            object.__setattr__(self, "arch", arch)

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (self.vendor, self.arch, self.index)

    @property
    def arch_parts(self) -> Arch | None:
        """Return parsed architecture components when an architecture is known."""

        return None if self.arch is None else Arch.parse(self.arch)


@dataclass(frozen=True)
class ModelContext:
    """Store model metadata supplied by the engine or model plugin."""

    family: str = "unknown"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            str(self.family if self.family is not None else "unknown").lower(),
        )
        tags = () if self.tags is None else self.tags
        object.__setattr__(
            self,
            "tags",
            frozenset(str(item).lower() for item in tags),
        )

    @property
    def fingerprint(self) -> tuple[object, ...]:
        """Return stable model facts for kernel cache keys."""

        return (self.family, tuple(sorted(self.tags)))


@dataclass(frozen=True)
class PhysicalSignature:
    """Store the physical layout consumed by kernel capability predicates."""

    format: str
    layout: str | None = None
    block_shape: tuple[int, int] | None = None
    scale_dtype: str | None = None
    granularity: str | None = None
    storage_dtype: str | None = None
    fields: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", dtype_name(self.format))
        if self.layout is not None:
            object.__setattr__(self, "layout", str(self.layout).lower())
        if self.scale_dtype is not None:
            object.__setattr__(self, "scale_dtype", dtype_name(self.scale_dtype))
        if self.storage_dtype is not None:
            object.__setattr__(self, "storage_dtype", dtype_name(self.storage_dtype))
        if self.granularity is not None:
            object.__setattr__(self, "granularity", str(self.granularity).lower())
        if self.block_shape is not None:
            object.__setattr__(
                self, "block_shape", tuple(int(value) for value in self.block_shape)
            )
        fields = (
            self.fields.items()
            if isinstance(self.fields, Mapping)
            else (self.fields or ())
        )
        object.__setattr__(
            self, "fields", tuple(sorted((str(k), _freeze(v)) for k, v in fields))
        )

    @property
    def fingerprint(self) -> tuple[object, ...]:
        """Return stable physical-layout facts for cache keys."""

        return (
            self.format,
            self.layout,
            self.block_shape,
            self.scale_dtype,
            self.granularity,
            self.storage_dtype,
            self.fields,
        )


@dataclass(frozen=True)
class ShapeFacts:
    """Store named dimensions and attributes for one operation."""

    dims: tuple[tuple[str, int], ...] = ()
    attrs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        dims = (
            self.dims.items() if isinstance(self.dims, Mapping) else (self.dims or ())
        )
        attrs = (
            self.attrs.items()
            if isinstance(self.attrs, Mapping)
            else (self.attrs or ())
        )
        object.__setattr__(
            self, "dims", tuple(sorted((str(k), int(v)) for k, v in dims))
        )
        object.__setattr__(
            self, "attrs", tuple(sorted((str(k), _freeze(v)) for k, v in attrs))
        )

    @classmethod
    def from_mapping(
        cls,
        dims: Mapping[str, object] | None = None,
        *,
        attrs: Mapping[str, object] | None = None,
    ) -> "ShapeFacts":
        # Split flat mappings into numeric dimensions and other attributes.
        normalized_dims: dict[str, object] = {}
        normalized_attrs: dict[str, object] = dict(attrs or {})
        for key, value in (dims or {}).items():
            # Treat booleans as attributes despite their integer base class.
            if isinstance(value, bool):
                normalized_attrs.setdefault(str(key), value)
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                normalized_attrs.setdefault(str(key), value)
            else:
                normalized_dims[str(key)] = value
        return cls(
            dims=tuple(sorted((str(k), int(v)) for k, v in normalized_dims.items())),
            attrs=tuple(
                sorted((str(k), _freeze(v)) for k, v in normalized_attrs.items())
            ),
        )

    @property
    def fingerprint(self) -> tuple[object, ...]:
        """Return stable shape facts for cache keys."""

        return (self.dims, self.attrs)


@dataclass(frozen=True)
class KernelQuery:
    """Store a normalized kernel selection query."""

    op: str
    role: str = ""
    # Preserve whether the caller supplied a device.
    device: DeviceProfile | object = field(default=_UNSET_DEVICE)
    model: ModelContext = field(default_factory=ModelContext)
    dtype: tuple[tuple[str, str], ...] = ()
    quant: PhysicalSignature | None = None
    shape: ShapeFacts = field(default_factory=ShapeFacts)
    mode: KernelMode = KernelMode.EAGER
    attrs: tuple[tuple[str, object], ...] = ()
    device_explicit: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_device = self.device
        if raw_device is _UNSET_DEVICE or raw_device is None:
            object.__setattr__(self, "device", DeviceProfile())
            object.__setattr__(self, "device_explicit", False)
        else:
            object.__setattr__(self, "device_explicit", True)
        if not isinstance(self.device, DeviceProfile):
            # probe_device handles live devices and canonical ``vendor:arch``
            # strings, and rejects anything else with guidance.
            from phyai.kernel.device import probe_device

            object.__setattr__(self, "device", probe_device(self.device))
        if not isinstance(self.model, ModelContext):
            # ``build()`` passes a ModelContext; a mapping arrives only from
            # callers spelling the context inline.
            object.__setattr__(self, "model", ModelContext(**self.model))
        if self.quant is not None and not isinstance(self.quant, PhysicalSignature):
            object.__setattr__(self, "quant", PhysicalSignature(**self.quant))
        if not isinstance(self.shape, ShapeFacts):
            # Flat mappings split into numeric dims and other attributes.
            object.__setattr__(self, "shape", ShapeFacts.from_mapping(self.shape))
        object.__setattr__(self, "op", str(self.op).lower())
        object.__setattr__(self, "role", str(self.role).lower())
        # ``build()`` wraps scalar dtypes into ``{"input": ...}``, so a
        # non-mapping here is already the canonical tuple-of-pairs form.
        dtype = (
            self.dtype.items()
            if isinstance(self.dtype, Mapping)
            else (self.dtype or ())
        )
        object.__setattr__(
            self,
            "dtype",
            tuple(sorted((str(k).lower(), dtype_name(v)) for k, v in dtype)),
        )
        object.__setattr__(self, "mode", KernelMode.normalize(self.mode))
        attrs = (
            self.attrs.items()
            if isinstance(self.attrs, Mapping)
            else (self.attrs or ())
        )
        object.__setattr__(
            self, "attrs", tuple(sorted((str(k), _freeze(v)) for k, v in attrs))
        )

    @classmethod
    def build(
        cls,
        op: str,
        *,
        role: str = "",
        device: DeviceProfile | torch.device | str | None = None,
        model: ModelContext | Mapping[str, object] | str | None = None,
        dtype: Mapping[str, object] | object | None = None,
        quant: PhysicalSignature | Mapping[str, object] | str | None = None,
        shape: ShapeFacts | Mapping[str, object] | None = None,
        mode: KernelMode | str = KernelMode.EAGER,
        attrs: Mapping[str, object] | None = None,
    ) -> "KernelQuery":
        return cls(
            op=op,
            role=role,
            device=_UNSET_DEVICE if device is None else device,
            model=ModelContext() if model is None else model,
            dtype=(
                dtype
                if isinstance(dtype, Mapping)
                else {"input": dtype}
                if dtype is not None
                else {}
            ),
            quant=quant,
            shape=ShapeFacts() if shape is None else shape,
            mode=KernelMode.normalize(mode),
            attrs=tuple((str(k), v) for k, v in (attrs or {}).items()),
        )

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.op,
            self.role,
            self.device.fingerprint,
            self.model.fingerprint,
            self.dtype,
            self.quant.fingerprint if self.quant is not None else None,
            self.shape.fingerprint,
            self.mode.value,
            self.attrs,
        )


__all__ = [
    "DeviceProfile",
    "KernelMode",
    "KernelQuery",
    "ModelContext",
    "PhysicalSignature",
    "ShapeFacts",
    "arch_at_least",
    "dtype_name",
]
