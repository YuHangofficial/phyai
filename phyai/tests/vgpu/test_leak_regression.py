"""Document the flashinfer per-call driver leak — loudly, until it is fixed.

flashinfer's ``split_device_green_ctx`` leaks ~16 MiB of driver-level
memory on every call — unrecoverable by ``cuStreamDestroy +
cuGreenCtxDestroy``, ``empty_cache``, or ``gc.collect``. Measured linear
at 16-18 MiB/iteration on 0.6.12 AND 0.6.17, so the "vGPUs must be
long-lived" posture in :mod:`phyai.vgpu` stands.

This test asserts the leak is STILL THERE. When an upstream release
finally fixes it, the test fails on purpose: that is the signal to relax
the long-lived-only warnings in ``vgpu.py`` and flip this assertion into
a fixed-behaviour regression guard.

Measurement note: nvidia-smi's ``--id=`` takes a PHYSICAL index and
ignores ``CUDA_VISIBLE_DEVICES``, while ``cuda:0`` is a VISIBLE index.
An earlier version of this test hardcoded ``--id=0`` and was validated
under ``CUDA_VISIBLE_DEVICES=7`` — it watched an idle card while the
leak landed on another, and "passed". Hence ``_physical_index`` below.
"""

from __future__ import annotations

import gc
import os
import subprocess

import pytest
import torch


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def _physical_index(visible_idx: int) -> int:
    """Map a torch (visible) device index to nvidia-smi's physical index.

    ``CUDA_VISIBLE_DEVICES`` reorders what torch sees; nvidia-smi does not
    honour it. UUID-based mapping would be cleaner, but torch's reported
    device UUID does not match NVML's under the CUDA compat driver this
    box runs, so parse the mask instead.
    """
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return visible_idx
    entries = [item.strip() for item in mask.split(",") if item.strip()]
    if visible_idx >= len(entries) or not entries[visible_idx].isdigit():
        pytest.skip(
            f"cannot map cuda:{visible_idx} through CUDA_VISIBLE_DEVICES={mask!r}"
        )
    return int(entries[visible_idx])


def _smi_mem_used_mib(physical_idx: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_idx}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(out)


def test_flashinfer_split_still_leaks_driver_memory():
    if not _flashinfer_available():
        pytest.skip("flashinfer required")

    # nvidia-smi is the only window into driver-level memory; bail if it's
    # missing (e.g. inside a container without the binary).
    try:
        subprocess.check_output(["nvidia-smi", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("nvidia-smi not available")

    physical = _physical_index(0)

    # Prime the primary context.
    _ = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()

    base_smi = _smi_mem_used_mib(physical)

    iters = 5
    from flashinfer.green_ctx import split_device_green_ctx

    for _ in range(iters):
        streams, resources = split_device_green_ctx(
            torch.device("cuda:0"),
            2,
            16,
        )
        del streams, resources
        gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    end_smi = _smi_mem_used_mib(physical)
    delta = end_smi - base_smi

    # ~16 MiB/call on 0.6.12 and 0.6.17 alike; 5 calls land at ~80 MiB.
    # Half that is far above nvidia-smi jitter and far below a real leak,
    # so crossing the bound means upstream behaviour genuinely changed.
    assert delta >= 40, (
        f"split_device_green_ctx grew driver memory by only {delta} MiB over "
        f"{iters} iterations — the upstream leak appears to be FIXED. "
        f"Celebrate, then: relax the long-lived-only warnings in "
        f"phyai/src/phyai/vgpu/vgpu.py and flip this test into a "
        f"fixed-behaviour regression guard (assert delta < 32)."
    )
