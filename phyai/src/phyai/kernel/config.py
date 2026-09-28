"""Define engine-facing kernel settings."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelConfig:
    """Configure kernel policy selection and autotuning."""

    profile: str | None = None
    config_path: str | None = None
    autotune_cache: str | None = None
    # Benchmarks a prepared candidate during autotuning. ``None`` with
    # ``profile: autotune`` gets the engine's default select-time hook
    # (phyai.kernel.benchmark); set a callable to override it.
    benchmark: Callable[..., float] | None = None

    def __post_init__(self) -> None:
        if self.profile is not None and self.profile.lower() not in {
            "static",
            "autotune",
        }:
            raise ValueError("KernelConfig.profile must be 'static' or 'autotune'")
        if self.profile is not None:
            object.__setattr__(self, "profile", self.profile.lower())
        if self.benchmark is not None and not callable(self.benchmark):
            raise TypeError("KernelConfig.benchmark must be callable or None")

    def policy(self):
        """Build the policy selected by this configuration."""

        from phyai.kernel.registry import build_catalog
        from phyai.kernel.bootstrap import resolve_policy

        return resolve_policy(self, build_catalog())


__all__ = ["KernelConfig"]
