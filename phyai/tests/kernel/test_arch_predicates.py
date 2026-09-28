"""``at_least`` / ``family_in``, and the YAML spellings that reach them.

Architecture ordering has two plausible-but-wrong readings, lexicographic and
cross-vendor, and a bare ``>=`` picks one of them silently. These tests pin the
reading, the failure *messages* (they are the reason a tree is used instead of
a callback), and the two literal grammars refusing each other's shape.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import Facts, device, quant, shape
from phyai.kernel.predicate import (
    FALSE,
    TRUE,
    ArchAtLeast,
    ArchFamilyIn,
    is_false,
    predicate_from_literal,
)


def verdict(predicate, arch: object) -> str:
    failure = predicate.eval(Facts(values={"device.arch": arch}))
    return "ok" if failure is None else failure.detail


# --------------------------------------------------------------------------- #
# at_least
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, ok", [("sm86", False), ("sm89", True), ("sm90", True), ("sm120", True)]
)
def test_at_least_matches_the_numeric_gate_it_replaces(arch, ok):
    """``device.sm >= 89`` gated five fp8/nvfp4 rows; this must admit the same
    set, or those kernels quietly stop being eligible."""
    assert (verdict(device.arch.at_least("sm89"), arch) == "ok") is ok


def test_at_least_explains_each_way_it_can_fail():
    """Too old, wrong vendor, unknown and never-provided are different problems,
    and the trace is the only place a user finds out which."""
    predicate = device.arch.at_least("sm89")
    assert predicate.render() == "device.arch >= sm89"
    assert verdict(predicate, "sm80") == "got 'sm80'"
    assert "not a 'sm' architecture" in verdict(predicate, "gfx942")
    # Not the string 'none' and not "too old": that conflation is what made
    # the old numeric fact report "no backend available" on CPU.
    assert verdict(predicate, None) == "device.arch is unknown"
    failure = predicate.eval(Facts(values={}))
    assert failure is not None and failure.detail == "device.arch was not provided"


def test_at_least_is_built_only_from_an_arch_fact_and_a_valid_bound():
    """Import time, not selection time: a bound that cannot mean anything must
    never get as far as a query, and ``ARCH`` stays unordered so the named
    operator (which carries the series) is the only way to compare."""
    with pytest.raises(TypeError, match="no ordering"):
        device.arch >= "sm90"
    for fact in (shape.K, quant.format, device.vendor):
        with pytest.raises(TypeError, match="needs an arch fact"):
            ArchAtLeast(fact, "sm90")
    with pytest.raises(ValueError, match="did you mean 'sm100'"):
        device.arch.at_least("sm10")
    with pytest.raises(ValueError, match="not in an ordered series"):
        device.arch.at_least("gfx942")


# --------------------------------------------------------------------------- #
# family_in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, ok", [("sm90a", True), ("sm100", True), ("sm120", False), ("sm89", False)]
)
def test_family_in_is_discrete_not_a_floor(arch, ok):
    """FlashInfer's GDN kernels ship a Hopper path and a Blackwell path and
    nothing else, so ">= sm90" would wrongly claim sm120."""
    assert (verdict(device.arch.family_in({"sm9", "sm10"}), arch) == "ok") is ok


def test_family_in_reports_and_renders_generations():
    predicate = device.arch.family_in({"sm10", "sm9"})
    # Sorted by (series, major), not by rendered text: {sm10, sm9} reads like a mistake.
    assert predicate.render() == "device.arch family in {sm9, sm10}"
    assert verdict(predicate, "sm120") == "got 'sm120' (generation sm12)"
    assert verdict(device.arch.family_in({"sm9"}), "ascend910b") == (
        "got 'ascend910b', which has no generation"
    )


def test_family_in_is_built_only_from_an_arch_fact_and_generations():
    """``at_least`` takes ``sm90`` and ``family_in`` takes ``sm9``; each must
    reject the other's shape rather than silently reinterpret it."""
    with pytest.raises(ValueError, match="did you mean"):
        device.arch.family_in({"sm90"})
    with pytest.raises(ValueError, match="at least one generation"):
        ArchFamilyIn(device.arch, [])
    with pytest.raises(TypeError, match="an arch fact"):
        ArchFamilyIn(quant.format, {"sm9"})


# --------------------------------------------------------------------------- #
# facts_used and restrict, inherited from Leaf
# --------------------------------------------------------------------------- #


def test_both_nodes_fold_under_restrict_and_report_the_fact_they_read():
    """``CallSite``'s memo key derives from ``facts_used``; ``restrict`` is what
    lets ``param_dtypes`` discard a row this host cannot run before asking it
    about parameter dtypes."""
    assert device.arch.at_least("sm90").facts_used() == frozenset({"device.arch"})
    assert device.arch.family_in({"sm9"}).facts_used() == frozenset({"device.arch"})

    predicate = device.arch.at_least("sm100")
    # A folded-true predicate ignores facts entirely.
    assert predicate.restrict({"device.arch": "sm120"}).eval(Facts()) is None
    assert is_false(predicate.restrict({"device.arch": "sm90"}))
    assert is_false(predicate.restrict({"device.arch": "gfx942"}))
    assert predicate.restrict({"device.vendor": "nvidia"}) is predicate
    # ``Leaf.restrict`` builds a fresh ``Const``, so an identity test against
    # FALSE silently kept single-leaf capabilities alive.
    folded = predicate.restrict({"device.arch": "sm90"})
    assert folded is not FALSE and is_false(folded)
    assert is_false(FALSE) and not is_false(TRUE)


# --------------------------------------------------------------------------- #
# The YAML surface
# --------------------------------------------------------------------------- #


def test_yaml_literals_compile_to_at_least_equality_or_enumeration():
    """``family_in`` gets no YAML spelling on purpose: a bare ``sm9`` would be
    ambiguous between a name and a generation, so a rule enumerates the archs
    it was validated on."""
    predicate = predicate_from_literal(device.arch, ">=sm100")
    assert predicate.render() == "device.arch >= sm100"
    assert verdict(predicate, "sm120") == "ok"

    exact = predicate_from_literal(device.arch, "sm90")
    assert exact.render() == "device.arch == sm90"
    assert verdict(exact, "sm90") == "ok" and verdict(exact, "sm100") != "ok"

    enumerated = predicate_from_literal(device.arch, ["sm90", "sm100"])
    assert verdict(enumerated, "sm100") == "ok"
    assert (
        verdict(enumerated, "sm120") != "ok" and verdict(enumerated, "gfx942") != "ok"
    )
    assert (
        predicate_from_literal(quant.format, ["bf16", "fp8_e4m3"]).render()
        == "quant.format in {bf16, fp8_e4m3}"
    )


@pytest.mark.parametrize(
    "literal, complaint",
    [
        # A pattern compiled as equality never fires and never explains itself;
        # compiled as a glob it hides which devices the rule was validated on.
        (["sm9*"], "enumerate"),
        ("gfx9*", "enumerate"),
        (["sm90", "gfx9*"], "gfx9"),
        # A ``<`` rule ("for old GPUs") nearly always excludes future hardware.
        (">sm90", "only '>=' is"),
        ("<=sm90", "only '>=' is"),
        # Used to compile to equality against the literal string ">=90".
        (">=90", "no series prefix"),
        (100, "no series prefix"),
    ],
)
def test_yaml_literals_that_cannot_mean_anything_are_errors(literal, complaint):
    with pytest.raises(ValueError, match=complaint):
        predicate_from_literal(device.arch, literal)


# --------------------------------------------------------------------------- #
# device facts
# --------------------------------------------------------------------------- #


def test_device_facts_covers_every_device_path_and_reports_absence_as_none():
    """A ``device.*`` path missing from ``device_facts`` stays symbolic through
    ``restrict``, so a row this host cannot run survives the filter and votes
    on what dtype a parameter is allocated in, silently."""
    from phyai.kernel.facts import GLOBAL_FACT_PATHS, device_facts
    from phyai.kernel.types import DeviceProfile

    declared = {p for p in GLOBAL_FACT_PATHS if p.startswith("device.")}
    assert declared == set(device_facts(DeviceProfile(vendor="nvidia", arch="sm90")))
    assert "device.arch" in declared
    values = device_facts(DeviceProfile(vendor="cpu"))
    assert values["device.arch"] is None and values["device.vendor"] == "cpu"
