"""Backend resolution priority: explicit > env > auto."""

from __future__ import annotations

import warnings

import pytest

from phyai.vgpu import backend as backend_mod
from phyai.vgpu.exceptions import VGPUNotApplicableError


@pytest.fixture(autouse=True)
def _clean_backend_state(monkeypatch):
    """Install fake backends and restore registry state and current pointer."""
    saved_backends = dict(backend_mod._BACKENDS)
    saved_current = backend_mod._CURRENT
    saved_probe = backend_mod._flashinfer_available
    monkeypatch.delenv("PHYAI_VGPU_BACKEND", raising=False)
    backend_mod._BACKENDS.clear()
    backend_mod._BACKENDS["flashinfer"] = type(
        "FakeFlashInfer", (), {"name": "flashinfer"}
    )
    backend_mod._BACKENDS["torch"] = type("FakeTorch", (), {"name": "torch"})
    yield
    backend_mod._BACKENDS.clear()
    backend_mod._BACKENDS.update(saved_backends)
    backend_mod._CURRENT = saved_current
    backend_mod._flashinfer_available = saved_probe


def _fallback_warnings(w) -> list:
    return [x for x in w if "falling back" in str(x.message)]


def test_explicit_beats_env_beats_auto(monkeypatch):
    backend_mod._flashinfer_available = lambda: True
    monkeypatch.setenv("PHYAI_VGPU_BACKEND", "torch")
    assert backend_mod.resolve("flashinfer").name == "flashinfer"
    assert backend_mod.resolve(None).name == "torch"
    monkeypatch.delenv("PHYAI_VGPU_BACKEND")
    assert backend_mod.resolve(None).name == "flashinfer"


def test_auto_falls_back_to_torch_with_one_warning_and_an_explicit_choice_never_warns():
    backend_mod._flashinfer_available = lambda: False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert backend_mod.resolve(None).name == "torch"
    assert len(_fallback_warnings(w)) == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert backend_mod.resolve("torch").name == "torch"
    assert not _fallback_warnings(w)


def test_unknown_names_and_uninitialized_access_are_errors():
    with pytest.raises(VGPUNotApplicableError):
        backend_mod.resolve("nonexistent")
    backend_mod._CURRENT = None
    with pytest.raises(RuntimeError, match="no active backend"):
        backend_mod.get_backend()
