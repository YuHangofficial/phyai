"""Priority bands: what they promise an out-of-tree kernel author.

A bare integer cannot answer "will 60 beat the built-in FlashInfer row?"; you
have to read every registration, and the answer changes whenever someone bumps
a number. The replacement is named bands, a reserved range above every in-tree
row, and a rejection for anything outside the scheme.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import device, dtype
from phyai.kernel.opspec import (
    PRIORITY_LIMIT,
    Impl,
    OpSpec,
    Priority,
    band_for,
    validate_priority,
)
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def impl(priority=None, kernel_id="toy.kernel", when=None) -> Impl:
    kwargs = {} if priority is None else {"priority": priority}
    return Impl(
        kernel_id=kernel_id,
        op="toy",
        when=dtype.input.is_set() if when is None else when,
        prepare=lambda facts, params: None,
        **kwargs,
    )


def test_the_bands_are_ordered_four_wide_and_plugin_is_on_top():
    """Four wide so an offset expresses relative preference inside one contract
    without crossing into the next band; PLUGIN on top so an out-of-tree row
    can outrank every in-tree row without auditing the tree."""
    values = [int(band) for band in Priority]
    assert values == sorted(values) == [0, 4, 8, 12, 16]
    assert PRIORITY_LIMIT == 20
    assert (
        int(Priority.PLUGIN) == max(values)
        and int(Priority.PLUGIN) + 4 == PRIORITY_LIMIT
    )
    for value, band in (
        (0, Priority.REFERENCE),
        (4, Priority.GENERAL),
        (10, Priority.OPTIMIZED),
        (19, Priority.PLUGIN),
    ):
        assert band_for(value) is band
    assert (
        Priority.OPTIMIZED + 2 == 10
        and band_for(Priority.OPTIMIZED + 2) is Priority.OPTIMIZED
    )


def test_priorities_are_validated_not_clamped():
    """A row written ``priority=100`` under the old free-for-all would silently
    land above the plugin band and outrank everything, the exact coordination
    failure bands exist to prevent. The error names the bands so the fix is in
    the message, not in the source."""
    for value in (-1, 20, 1000):
        with pytest.raises(ValueError, match=r"priority must be in \[0, 20\)"):
            validate_priority(value)
    with pytest.raises(ValueError) as excinfo:
        impl(100)
    assert all(band.name in str(excinfo.value) for band in Priority)
    row = impl(Priority.OPTIMIZED + 2)
    assert row.priority == 10 and type(row.priority) is int
    # A row that does not say otherwise must not outrank one that does.
    assert impl().priority == int(Priority.REFERENCE)


def test_priority_orders_eligible_rows_but_cannot_make_a_row_eligible():
    """Capability is filtered first; priority is only the tiebreaker among
    survivors, otherwise a plugin could force itself onto hardware it cannot
    run on."""
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    catalog.register(impl(Priority.REFERENCE, "low.toy"))
    catalog.register(impl(Priority.PLUGIN, "high.toy"))
    catalog.register(
        impl(Priority.PLUGIN + 3, "blackwell.toy", when=device.arch.at_least("sm100"))
    )
    assert [row.kernel_id for row in catalog.impls("toy")] == [
        "blackwell.toy",
        "high.toy",
        "low.toy",
    ]
    trace = Selector(catalog, device="nvidia:SM90").explain(
        KernelQuery.build("toy", dtype={"input": "bf16"})
    )
    assert trace.selected == "high.toy"


def test_every_in_tree_row_sits_in_a_declared_band_below_plugin():
    """The reserved-headroom promise checked against the real catalog, plus:
    a reference row that outranked a real one would be a mistake in every case."""
    catalog = build_catalog()
    for spec in catalog.ops():
        for row in catalog.impls(spec.name):
            assert 0 <= row.priority < int(Priority.PLUGIN), row.kernel_id
            assert band_for(row.priority) in set(Priority), row.kernel_id
            if row.reference:
                assert band_for(row.priority) is Priority.REFERENCE, row.kernel_id
