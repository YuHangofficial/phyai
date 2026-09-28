"""Env-var overlay for the kernel policy config.

``PHYAI_KERNEL_CONFIG`` / ``PHYAI_KERNEL_PROFILE`` / ``PHYAI_KERNEL_AUTOTUNE_CACHE``
are the only way to point a deployment at a policy file without editing code,
so a silent break here is expensive and invisible.
"""

from __future__ import annotations

import pytest

from phyai.engine_config import EngineConfig


POLICY_YAML = """\
schema: phyai.kernel/v1
profile: autotune
"""


@pytest.fixture(autouse=True)
def _clear_kernel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PHYAI_KERNEL_CONFIG",
        "PHYAI_KERNEL_PROFILE",
        "PHYAI_KERNEL_AUTOTUNE_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_no_kernel_env_leaves_config_at_defaults():
    kernel = EngineConfig.from_env().kernel
    assert kernel.config_path is None
    # ``auto()`` leaves the profile unset so an external YAML can select
    # ``autotune``; the effective default stays ``static``.
    assert kernel.profile is None and kernel.policy().profile == "static"
    assert kernel.autotune_cache is None


def test_config_path_reaches_the_loader_and_profile_env_is_the_outermost_override(
    monkeypatch, tmp_path
):
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_YAML, encoding="utf-8")
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", str(path))
    kernel = EngineConfig.from_env().kernel
    assert kernel.config_path == str(path)
    assert kernel.policy().profile == "autotune" and kernel.policy().source == str(path)

    # The YAML says autotune; the env var must still win, or there is no way
    # to force deterministic selection on a machine whose policy asks to measure.
    monkeypatch.setenv("PHYAI_KERNEL_PROFILE", "static")
    kernel = EngineConfig.from_env().kernel
    assert kernel.profile == "static" and kernel.policy().profile == "static"

    monkeypatch.setenv("PHYAI_KERNEL_PROFILE", "turbo")
    with pytest.raises(ValueError, match="profile"):
        EngineConfig.from_env()


def test_autotune_cache_env_overlays_rather_than_discards_the_base(
    monkeypatch, tmp_path
):
    cache = tmp_path / "autotune.json"
    monkeypatch.setenv("PHYAI_KERNEL_AUTOTUNE_CACHE", str(cache))
    assert EngineConfig.from_env().kernel.autotune_cache == str(cache)

    base = EngineConfig.auto()
    merged = EngineConfig.from_env(base)
    assert merged.kernel.autotune_cache == str(cache)
    assert merged.kernel.config_path == base.kernel.config_path
    assert merged.kernel.profile == base.kernel.profile
    assert merged.device.target == base.device.target
