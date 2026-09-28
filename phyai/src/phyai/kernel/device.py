"""Probe device facts used by kernel selection."""

from __future__ import annotations

from functools import lru_cache

import torch

from phyai.kernel.types import DeviceProfile
from phyai.utils.vendors import VENDORS, VENDOR_SUGGESTIONS


def _synthetic_profile(value: object) -> DeviceProfile:
    """Parse a canonical ``vendor:arch`` device string without touching CUDA."""

    text = str(value).strip().lower()
    if ":" not in text:
        hint = ""
        for candidate in VENDORS.values():
            if text.startswith(candidate.series):
                hint = f"; did you mean '{candidate.name}:{text}'?"
                break
        raise ValueError(
            f"cannot parse device {value!r}: a synthetic device is written "
            f"'vendor:arch', e.g. 'nvidia:sm90'{hint}"
        )

    vendor_name, arch = text.split(":", 1)
    vendor = VENDORS.get(vendor_name)
    if vendor is None:
        suggestion = VENDOR_SUGGESTIONS.get(vendor_name)
        hint = f"; write '{suggestion}:{arch}'" if suggestion else ""
        raise ValueError(
            f"unknown device vendor {vendor_name!r} "
            f"(known: {', '.join(sorted(VENDORS))}){hint}"
        )
    known = vendor.products.get(arch)
    if known is not None:
        raise ValueError(
            f"{arch!r} is a product name, not an architecture; write "
            f"{known!r} instead (as in {vendor.name}:{known})"
        )
    if not arch.startswith(vendor.series):
        raise ValueError(
            f"{arch!r} does not look like a {vendor.name} architecture; "
            f"{vendor.name} architecture names start with {vendor.series!r} "
            f"(e.g. '{vendor.name}:{vendor.example_arch}')"
        )
    return DeviceProfile(vendor=vendor.name, arch=arch)


@lru_cache(maxsize=16)
def _probe_device_cached(device_key: str) -> DeviceProfile:
    """Return a canonical profile for ``device`` without compiling kernels."""

    target = torch.device(device_key)
    vendor = "cpu"
    arch: str | None = None
    index: int | None = target.index

    if target.type == "cuda":
        # PyTorch exposes ROCm devices through the CUDA device API.
        if getattr(torch.version, "hip", None):
            vendor = "amd"
            try:
                if index is None:
                    index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                arch = (
                    str(getattr(props, "gcnArchName", "") or "")
                    .lower()
                    .split(":", 1)[0]
                    or None
                )
            except (RuntimeError, AssertionError):
                pass
        else:
            vendor = "nvidia"
            try:
                if index is None:
                    index = torch.cuda.current_device()
                major, minor = torch.cuda.get_device_capability(index)
                arch = f"sm{major}{minor}"
            except (RuntimeError, AssertionError):
                pass
    elif target.type in {"npu", "mlu", "xpu"}:
        vendor = {"npu": "ascend", "mlu": "cambricon", "xpu": "intel"}.get(
            target.type, target.type
        )

    return DeviceProfile(vendor=vendor, arch=arch, index=index)


def probe_device(
    device: DeviceProfile | torch.device | str | None = None,
) -> DeviceProfile:
    """Return a canonical profile for a device or ``vendor:arch`` string."""

    if isinstance(device, DeviceProfile):
        return device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return _probe_device_cached(str(torch.device(device)))
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return _synthetic_profile(device)


__all__ = ["probe_device"]
