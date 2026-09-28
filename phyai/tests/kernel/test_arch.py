"""The ``Arch`` value type: parsing, ordering, and what it refuses.

Architecture used to be three facts (a name, a compute-capability number, and
its major) because a string is not comparable and the number was the only way
to write ``sm >= 89``. That put the vendor's naming scheme into the fact schema.
These tests pin the single-fact replacement, and in particular the two things
it must *not* do: order across vendors, and accept a bound that cannot mean
anything.
"""

from __future__ import annotations

import pytest

from phyai.kernel.device import _synthetic_profile
from phyai.kernel.types import (
    ARCH_SERIES_GRAMMAR,
    ORDERED_ARCH_SERIES,
    Arch,
    DeviceProfile,
    arch_at_least,
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, series, major, minor",
    [
        ("sm89", "sm", 8, 9),  # sm_XY is major*10 + minor, NVIDIA's own encoding
        ("sm100", "sm", 10, 0),
        ("sm120", "sm", 12, 0),
        ("sm90a", "sm", 9, 0),  # arch-specific instructions, same capability
        ("gfx90a", "gfx", 9, 0),  # AMD: major.minor.stepping, stepping may be a letter
        ("gfx942", "gfx", 9, 4),
        ("gfx1100", "gfx", 11, 0),
    ],
)
def test_parse_splits_each_series_by_its_own_grammar(name, series, major, minor):
    arch = Arch.parse(name)
    assert (arch.name, arch.series, arch.major, arch.minor) == (
        name,
        series,
        major,
        minor,
    )
    assert Arch.parse(name.upper()) == arch  # case-insensitive


@pytest.mark.parametrize("name", ["ascend910b", "h100", "cpu", ""])
def test_parse_never_raises_and_leaves_unstructured_names_unstructured(name):
    """Parsing runs during evaluation, so it must not turn an unknown
    accelerator into a crash."""
    arch = Arch.parse(name)
    assert arch.name == name
    assert arch.major is None and arch.family is None and not arch.ordered


# --------------------------------------------------------------------------- #
# Ordering, inside a series only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, expected",
    [("sm80", False), ("sm86", False), ("sm89", True), ("sm90", True), ("sm100", True)],
)
def test_at_least_reproduces_the_numeric_comparison_it_replaces(arch, expected):
    """``device.sm >= 89`` admitted sm89 through sm120; so must this. A
    family-scoped comparison (sm89 is family sm8, sm90 is family sm9) would
    have silently disqualified every Hopper and Blackwell host."""
    assert Arch.parse(arch).at_least(Arch.parse_bound("sm89")) is expected


def test_ordering_compares_major_minor_pairs_within_the_nvidia_series_only():
    # sm100 > sm90 even though "100" < "90" as text.
    assert Arch.parse("sm100").at_least(Arch.parse_bound("sm90"))
    assert not Arch.parse("sm90").at_least(Arch.parse_bound("sm100"))
    # Across series the question has no answer; False is the right
    # eligibility verdict and comes from the series mismatch itself.
    assert not Arch.parse("gfx942").at_least(Arch.parse_bound("sm90"))
    assert not Arch.parse("ascend910b").at_least(Arch.parse_bound("sm70"))
    # gfx1100 (RDNA3) is numerically above gfx942 (CDNA3) with a disjoint
    # matrix-instruction set, so ordering gfx would be confidently wrong.
    assert ORDERED_ARCH_SERIES == frozenset({"sm"})
    assert Arch.parse("gfx1100").ordered is False
    assert Arch.parse("gfx1100").at_least(Arch.parse("gfx942")) is False
    # Otherwise ``at_least`` would accept a bound it can never parse.
    assert ORDERED_ARCH_SERIES <= set(ARCH_SERIES_GRAMMAR)


@pytest.mark.parametrize(
    "name, family",
    [
        ("sm90", "sm9"),
        ("sm100", "sm10"),
        ("sm89", "sm8"),
        ("gfx942", "gfx9"),
        ("gfx1100", "gfx11"),
    ],
)
def test_family_is_series_plus_major(name, family):
    assert Arch.parse(name).family == family


# --------------------------------------------------------------------------- #
# Bounds and generations are validated where they are written
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bound, complaint",
    [
        ("100", "no series prefix"),  # a bare number carries no series
        ("sm10", "did you mean 'sm100'"),  # parses as major 1: admits everything
        ("sm9", "names a generation"),  # would silently mean ">= sm99"
        ("gfx942", "not in an ordered series"),
        ("cpu", "not in an ordered series"),
        ("", "must not be empty"),
    ],
)
def test_parse_bound_rejects_what_cannot_be_an_ordered_bound(bound, complaint):
    with pytest.raises(ValueError, match=complaint):
        Arch.parse_bound(bound)


def test_parse_family_takes_generations_and_refuses_architecture_names():
    """The sharpest edge in the design: ``at_least`` takes ``sm90`` and
    ``family_in`` takes ``sm9``. Each grammar must refuse the other's shape."""
    for family in ("sm9", "sm10", "sm12", "gfx9", "gfx11"):
        assert Arch.parse_family(family) == family
    with pytest.raises(ValueError, match="did you mean"):
        Arch.parse_family("sm90")
    with pytest.raises(ValueError, match="unknown series"):
        Arch.parse_family("zz9")
    with pytest.raises(ValueError, match="series followed by a major"):
        Arch.parse_family("sm90a")


# --------------------------------------------------------------------------- #
# The profile helper
# --------------------------------------------------------------------------- #


def test_profile_derives_arch_parts_and_rejects_bare_numbers():
    """Storing major/minor next to the name would be a second source of truth
    and would make ``DeviceProfile(arch="sm90", major=12)`` constructible."""
    profile = DeviceProfile(vendor="nvidia", arch="sm90")
    assert profile.arch_parts == Arch.parse("sm90")
    assert not hasattr(profile, "arch_major")
    assert DeviceProfile(vendor="cpu").arch_parts is None
    # ``DeviceProfile(vendor="amd", arch="100")`` used to fabricate ``gfx100``.
    with pytest.raises(ValueError, match="no series prefix"):
        DeviceProfile(vendor="nvidia", arch="100")
    with pytest.raises(ValueError, match="write the full name"):
        DeviceProfile(vendor="amd", arch="100")


@pytest.mark.parametrize(
    "vendor, arch, bound, expected",
    [
        ("nvidia", "sm90", "sm100", False),
        ("nvidia", "sm120", "sm100", True),
        # The bug this helper exists to kill: "h100" scraped to 100 and read as
        # ">= sm100", so a Hopper card got a Blackwell layout.
        ("nvidia", "h100", "sm100", False),
        (
            "amd",
            "gfx942",
            "sm100",
            False,
        ),  # scraped to 942, held off only by a vendor check
        ("cpu", None, "sm100", False),
    ],
)
def test_arch_at_least_on_a_profile(vendor, arch, bound, expected):
    assert arch_at_least(DeviceProfile(vendor=vendor, arch=arch), bound) is expected


# --------------------------------------------------------------------------- #
# The synthetic-profile boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, arch",
    [
        ("nvidia:sm90", "sm90"),
        ("nvidia:SM100", "sm100"),
        ("amd:gfx942", "gfx942"),
        ("ascend:ascend910b", "ascend910b"),
    ],
)
def test_a_vendor_arch_string_is_accepted(text, arch):
    assert _synthetic_profile(text).arch == arch


@pytest.mark.parametrize(
    "text, complaint",
    [
        # One spelling per device: everything short of 'vendor:arch' names the
        # grammar to write, pointing at the canonical form when intent is obvious.
        ("sm90", "did you mean 'nvidia:sm90'"),
        ("nvidia", "'vendor:arch'"),
        ("cuda:sm90", "write 'nvidia:sm90'"),
        ("intel:xe2", "unknown device vendor"),
        ("nvidia:gfx942", "does not look like a nvidia architecture"),
        (
            "nvidia:hopper",
            "does not look like",
        ),  # not a product name either: no guessing
    ],
)
def test_everything_else_is_rejected_with_guidance(text, complaint):
    with pytest.raises(ValueError) as excinfo:
        _synthetic_profile(text)
    assert complaint in str(excinfo.value)


@pytest.mark.parametrize(
    "text, replacement",
    [
        ("nvidia:H100", "sm90"),
        ("nvidia:B200", "sm100"),
        # Edge and workstation parts matter as much as datacenter ones for
        # physical AI: Jetson Orin/Thor and consumer Blackwell.
        ("nvidia:AGX-Orin", "sm87"),
        ("nvidia:Thor", "sm110"),
        ("nvidia:RTX5090", "sm120"),
        ("amd:MI300", "gfx942"),
    ],
)
def test_a_product_name_is_refused_and_names_its_architecture(text, replacement):
    """A product name matched *nothing*: no capability reads one, so every
    architecture-gated kernel silently became ineligible, while materialization
    scraped the digits so ``h100`` read as ">= sm100"."""
    with pytest.raises(ValueError, match="product name") as excinfo:
        _synthetic_profile(text)
    assert replacement in str(excinfo.value)


def test_the_product_table_maps_to_real_architectures():
    """Every replacement it suggests must itself parse, or the error message
    sends people somewhere equally broken."""
    from phyai.utils.vendors import VENDORS

    for vendor in VENDORS.values():
        for product, arch in vendor.products.items():
            assert (
                Arch.parse(arch).major is not None
            ), f"{product} -> {arch} does not parse"
            assert arch.startswith(
                vendor.series
            ), f"{product} -> {arch} is not a {vendor.name} architecture"
