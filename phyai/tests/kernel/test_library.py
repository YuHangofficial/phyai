"""Library availability probing: "installed" is not "importable".

A package whose native extension fails to load (flashinfer on a host with a
mismatched CUDA) is *installed*, so a module-spec check reports it present.
Every kernel gated on it then claims eligibility, preparation throws, the
selector silently falls through, and the trace records a viable kernel that
never was.
"""

from __future__ import annotations

import pytest

from phyai.kernel import library


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    library.reset_library_probes()
    yield
    library.reset_library_probes()


@pytest.fixture
def fake_import(monkeypatch: pytest.MonkeyPatch):
    """Replace ``import_module`` with a scripted one, recording each call."""
    calls: list[str] = []
    real = library.importlib.import_module

    def install(behaviour):
        def patched(target: str):
            calls.append(target)
            outcome = behaviour(target)
            if isinstance(outcome, BaseException):
                raise outcome
            return real(target) if outcome is None else outcome

        monkeypatch.setattr(library.importlib, "import_module", patched)
        return calls

    return install


def test_installed_but_unimportable_modules_are_unavailable(fake_import):
    """A broken native extension raises from the dynamic loader (an ``OSError``,
    not an ``ImportError``) and may even exit, which is why the probe catches
    ``BaseException``."""
    fake_import(
        lambda target: OSError("libcuda.so.1: cannot open shared object file")
        if target == "flashinfer"
        else (SystemExit(1) if target == "brittle" else None)
    )
    assert library.library_available("json")
    assert not library.library_available("flashinfer")
    assert not library.library_available("brittle")
    assert not library.library_available("phyai_no_such_module_xyz")


def test_both_outcomes_are_memoized(fake_import):
    """At most one import attempt per library per process; otherwise every
    selector cache miss pays another failed import."""
    calls = fake_import(
        lambda target: ImportError("nope") if target == "absent" else None
    )
    assert library.library_available("json") and library.library_available("json")
    assert not library.library_available("absent") and not library.library_available(
        "absent"
    )
    assert calls == ["json", "absent"]


def test_library_facts_builds_lib_prefixed_paths():
    values = library.library_facts(frozenset({"json", "phyai_no_such_module_xyz"}))
    assert values == {"lib.json": True, "lib.phyai_no_such_module_xyz": False}


def test_flashinfer_probe_matches_reality_on_this_host():
    try:
        import flashinfer  # noqa: F401
    except BaseException:
        expected = False
    else:
        expected = True
    assert library.library_available("flashinfer") is expected
