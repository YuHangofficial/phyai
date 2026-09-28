"""Static accelerator-vendor knowledge: names, architecture grammar, products.

One record per vendor. The kernel selection system derives its device
grammar from these records -- series prefixes, version ordering, product-name
rejection -- but nothing here depends on the kernel package or on torch, so
any tool that needs to answer "which vendor owns the ``sm`` series", "what
architecture is an H100", or "how does a gfx name split into major/minor"
can import this directly.

Adding a vendor is one :class:`Vendor` record; everything else derives.
"""

from __future__ import annotations

from typing import Callable, Mapping
from dataclasses import field, dataclass


def split_sm_version(digits: str, suffix: str) -> tuple[int, int] | None:
    """Parse an NVIDIA SM version into major and minor components."""

    if len(digits) < 2:
        # A single digit names a generation ("sm9"), not a version; treating
        # it as a version would invent a minor and make "sm9" mean sm99.
        return None
    return int(digits[:-1]), int(digits[-1])


def split_gfx_version(digits: str, suffix: str) -> tuple[int, int] | None:
    """Parse an AMD GFX version into major and minor components."""

    combined = f"{digits}{suffix}"
    if len(combined) < 3:
        return None
    return int(combined[:-2]), int(combined[-2])


@dataclass(frozen=True)
class Vendor:
    """Everything phyai knows statically about one accelerator vendor."""

    #: Canonical lowercase vendor name; the only accepted spelling.
    name: str

    #: Series prefix its architecture names start with ("sm90" -> "sm").
    series: str

    #: Example architecture for error messages.
    example_arch: str

    #: Whether versions within the series order meaningfully (``at_least``).
    ordered: bool = False

    #: Split a version's digits and suffix into ``(major, minor)``, or
    #: ``None`` when the digits name a generation rather than a version.
    split_version: Callable[[str, str], tuple[int, int] | None] | None = None

    #: Major versions that plausibly exist, for typo detection.
    plausible_majors: range | None = None

    #: Product names people type by mistake, and the architecture to write.
    products: Mapping[str, str] = field(default_factory=dict)


NVIDIA = Vendor(
    name="nvidia",
    series="sm",
    example_arch="sm90",
    ordered=True,
    split_version=split_sm_version,
    plausible_majors=range(5, 20),
    products={
        # Datacenter.
        "a100": "sm80",
        "a800": "sm80",
        "l40": "sm89",
        "l40s": "sm89",
        "h100": "sm90",
        "h200": "sm90",
        "h800": "sm90",
        "h20": "sm90",
        "gh200": "sm90",
        "b100": "sm100",
        "b200": "sm100",
        "gb200": "sm100",
        "b300": "sm103",
        "gb300": "sm103",
        # Jetson edge modules (physical-AI deployment targets). Thor's
        # compute capability is 11.0 under CUDA 13 (briefly sm101 in 12.x).
        "orin": "sm87",
        "orin-nx": "sm87",
        "orin-nano": "sm87",
        "agx-orin": "sm87",
        "thor": "sm110",
        "agx-thor": "sm110",
        # Consumer / workstation.
        "rtx4090": "sm89",
        "rtx5090": "sm120",
        "rtxpro6000": "sm120",
        # DGX Spark desktop (GB10 Grace-Blackwell superchip).
        "gb10": "sm121",
        "dgx-spark": "sm121",
    },
)

AMD = Vendor(
    name="amd",
    series="gfx",
    example_arch="gfx942",
    split_version=split_gfx_version,
    products={
        "mi250": "gfx90a",
        "mi300": "gfx942",
        "mi300x": "gfx942",
        "mi325x": "gfx942",
        "mi355x": "gfx950",
    },
)

ASCEND = Vendor(
    name="ascend",
    series="ascend",
    example_arch="ascend910b",
)

#: Canonical vendors by name.
VENDORS: dict[str, Vendor] = {vendor.name: vendor for vendor in (NVIDIA, AMD, ASCEND)}

#: Rejected vendor spellings and the canonical name to suggest. Used only to
#: build error messages; never accepted silently.
VENDOR_SUGGESTIONS = {
    "nv": "nvidia",
    "cuda": "nvidia",
    "rocm": "amd",
    "hip": "amd",
    "npu": "ascend",
}


__all__ = [
    "AMD",
    "ASCEND",
    "NVIDIA",
    "VENDORS",
    "VENDOR_SUGGESTIONS",
    "Vendor",
]
