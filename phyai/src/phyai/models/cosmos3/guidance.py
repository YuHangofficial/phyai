"""Classifier-free guidance helpers shared by Cosmos3 schedulers."""

from __future__ import annotations

from typing import Any


def combine_cfg(cond: Any, uncond: Any, guidance_scale: float) -> Any:
    """Match the single-axis scheduler's definition of disabled CFG."""
    if guidance_scale <= 1.0:
        return cond
    return uncond + guidance_scale * (cond - uncond)


__all__ = ["combine_cfg"]
