"""Unit tests for the predicate algebra and the fact vocabulary.

Grouped by the property being pinned rather than by class, because the
properties are what the rest of the system relies on: three-state fact
semantics, optional-fact vacuity, ``render`` producing the text that lands in
traces, ``restrict`` doing sound partial evaluation (what lets a layer pick a
parameter dtype before it has a tensor), and the parenthesization guard.

The final group writes out the real capability expressions this algebra has
to replace. If those stop fitting, the design is wrong and these fail first.
"""

from __future__ import annotations

import pytest
import torch

from phyai.kernel.facts import (
    Fact,
    FactKind,
    Facts,
    ParensError,
    attrs,
    device,
    dtype,
    lib,
    model,
    quant,
    shape,
)
from phyai.kernel.predicate import (
    FALSE,
    TRUE,
    Const,
    all_of,
    any_of,
    implies,
    none_of,
    predicate_from_literal,
    same,
)


def facts(values: dict[str, object], optional: set[str] | None = None) -> Facts:
    return Facts(values=values, optional=frozenset(optional or ()))


# --------------------------------------------------------------------------- #
# Three-state fact semantics
# --------------------------------------------------------------------------- #


def test_present_none_and_absent_facts_are_three_different_answers():
    """A CPU device has no SM number; that must not read as "sm 0" (the old
    numeric probe returned 0 and turned "unknown device" into the weaker "no
    backend is fast enough"), and "the caller forgot" is a third case."""
    assert (shape.K >= 4096).eval(facts({"shape.K": 4096})) is None
    assert (shape.K >= 8192).eval(facts({"shape.K": 4096})) is not None

    unknown = (shape.K >= 4096).eval(facts({"shape.K": None}))
    assert unknown is not None and unknown.detail == "shape.K is unknown"
    assert "shape.K >= 4096" in str(unknown)

    absent = (shape.K >= 4096).eval(facts({}))
    assert absent is not None and "not provided" in absent.detail
    assert "unknown" not in absent.detail


def test_optional_facts_are_vacuous_when_absent_and_enforced_when_present():
    predicate = dtype.residual == "bf16"
    query = facts({"dtype.input": "bf16"}, optional={"dtype.residual"})
    assert predicate.eval(query) is None
    assert [item.predicate for item in predicate.skipped(query)] == [
        "dtype.residual == bf16"
    ]
    provided = facts({"dtype.residual": "fp32"}, optional={"dtype.residual"})
    assert predicate.eval(provided) is not None


# --------------------------------------------------------------------------- #
# Normalization: one vocabulary, many spellings
# --------------------------------------------------------------------------- #


def test_dtype_spellings_normalize_on_both_sides():
    for spelling in ("bf16", "bfloat16", "torch.bfloat16", torch.bfloat16):
        assert (dtype.input == spelling).eval(facts({"dtype.input": "bf16"})) is None
    assert (dtype.input == "bf16").eval(facts({"dtype.input": torch.bfloat16})) is None
    assert (dtype.input == "fp8").eval(facts({"dtype.input": "fp8_e4m3"})) is None
    assert (dtype.input == "fp8_e4m3").eval(
        facts({"dtype.input": "float8_e4m3fn"})
    ) is None


def test_vendor_names_are_case_insensitive_and_have_no_aliases():
    """One canonical spelling per vendor: ``cuda`` and friends used to
    normalize to ``nvidia`` through an alias table; now an alias simply fails
    to match, visibly in the trace instead of silently rewritten."""
    assert (device.vendor == "NVIDIA").eval(facts({"device.vendor": "nvidia"})) is None
    for alias in ("nv", "cuda", "rocm", "npu"):
        assert (device.vendor == alias).eval(
            facts({"device.vendor": "nvidia"})
        ) is not None


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def test_numeric_operators_and_their_build_time_guards():
    assert (shape.K % 16 == 0).eval(facts({"shape.K": 4096})) is None
    failure = (shape.K % 16 == 0).eval(facts({"shape.K": 4095}))
    assert failure is not None and failure.predicate == "shape.K % 16 == 0"

    between = shape.kernel.between(1, 8)  # inclusive
    assert between.eval(facts({"shape.kernel": 1})) is None
    assert between.eval(facts({"shape.kernel": 8})) is None
    assert between.eval(facts({"shape.kernel": 9})) is not None

    # Catch the mistake where it is written, not where it is evaluated.
    with pytest.raises(TypeError, match="no ordering"):
        _ = device.arch >= "sm90"
    with pytest.raises(TypeError, match="int fact"):
        _ = quant.format % 16 == 0


def test_membership_set_boolean_and_none_operators():
    assert (
        dtype.input.in_({"bf16", "fp16"}).eval(facts({"dtype.input": "fp16"})) is None
    )
    tags = facts({"model.tags": frozenset({"debug", "reference"})})
    assert model.tags.has("debug").eval(tags) is None
    assert model.tags.intersects({"debug", "other"}).eval(tags) is None

    assert quant.format.is_none().eval(facts({"quant.format": None})) is None
    assert quant.format.is_none().eval(facts({"quant.format": "nvfp4"})) is not None

    both = all_of(attrs.causal, device.vendor == "nvidia")  # boolean fact used bare
    assert both.eval(facts({"attrs.causal": True, "device.vendor": "nvidia"})) is None
    assert (
        both.eval(facts({"attrs.causal": False, "device.vendor": "nvidia"})) is not None
    )

    neither = none_of(device.vendor == "amd", device.vendor == "ascend")
    assert neither.eval(facts({"device.vendor": "nvidia"})) is None
    assert neither.eval(facts({"device.vendor": "amd"})) is not None


def test_same_and_implies_relate_facts_to_each_other():
    agree = same(dtype.input, dtype.key, dtype.value)
    assert (
        agree.eval(
            facts({"dtype.input": "bf16", "dtype.key": "bf16", "dtype.value": "bf16"})
        )
        is None
    )
    failure = agree.eval(
        facts({"dtype.input": "bf16", "dtype.key": "fp16", "dtype.value": "bf16"})
    )
    assert failure is not None and "'bf16' vs 'fp16'" in failure.detail

    conditional = implies(attrs.bias, dtype.bias == "fp32")
    assert conditional.eval(facts({"attrs.bias": False})) is None  # vacuous
    assert conditional.eval(facts({"attrs.bias": True, "dtype.bias": "fp32"})) is None
    assert (
        conditional.eval(facts({"attrs.bias": True, "dtype.bias": "bf16"})) is not None
    )


# --------------------------------------------------------------------------- #
# Failure reporting
# --------------------------------------------------------------------------- #


def test_failures_name_the_leaf_and_are_actionable():
    conjunction = all_of(
        device.vendor == "nvidia", device.arch.at_least("sm100"), shape.K % 16 == 0
    )
    failure = conjunction.eval(
        facts({"device.vendor": "nvidia", "device.arch": "sm90", "shape.K": 4095})
    )
    # The first failure in written order, naming the specific leaf.
    assert failure is not None
    assert (failure.predicate, failure.detail) == ("device.arch >= sm100", "got 'sm90'")

    disjunction = any_of(quant.format == "bf16", quant.format == "fp16")
    failure = disjunction.eval(facts({"quant.format": "nvfp4"}))
    assert failure is not None and failure.detail.count("got") == 2  # every alternative

    assert str((shape.K >= 8192).eval(facts({"shape.K": 4096}))) == (
        "shape.K >= 8192 failed: got 4096"
    )


# --------------------------------------------------------------------------- #
# render()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("predicate", "text"),
    [
        (device.arch.at_least("sm100"), "device.arch >= sm100"),
        (quant.format == "nvfp4", "quant.format == nvfp4"),
        (shape.K % 16 == 0, "shape.K % 16 == 0"),
        (dtype.input.in_({"bf16"}), "dtype.input in {bf16}"),
        (model.tags.has("debug"), "'debug' in model.tags"),
        # Numbers sort numerically, not by rendered text: {128, 256, 64} reads
        # as a mistake in a catalog listing that exists to be read.
        (shape.head_dim.in_({128, 256, 64}), "shape.head_dim in {64, 128, 256}"),
        (quant.format.is_none(), "quant.format is none"),
        (attrs.causal.as_predicate(), "attrs.causal"),
        (lib.has("flashinfer"), "lib.flashinfer"),
        (implies(attrs.bias, dtype.bias == "fp32"), "attrs.bias -> dtype.bias == fp32"),
    ],
)
def test_render_is_readable_and_stable(predicate, text: str):
    assert predicate.render() == text


def test_render_parenthesizes_disjunctions_and_flattens_nested_conjunctions():
    mixed = all_of(
        device.vendor == "nvidia",
        any_of(quant.format == "bf16", quant.format == "fp16"),
    )
    assert mixed.render() == (
        "device.vendor == nvidia & (quant.format == bf16 | quant.format == fp16)"
    )
    nested = all_of(
        all_of(device.arch.at_least("sm90"), shape.K % 16 == 0), attrs.causal
    )
    assert nested.render() == "device.arch >= sm90 & shape.K % 16 == 0 & attrs.causal"


def test_facts_used_collects_every_path_and_is_empty_for_constants():
    predicate = all_of(
        lib.has("flashinfer"),
        device.vendor == "nvidia",
        device.arch.at_least("sm100"),
        same(dtype.input, dtype.key),
        implies(attrs.bias, dtype.bias == "fp32"),
    )
    assert predicate.facts_used() == {
        "lib.flashinfer",
        "device.vendor",
        "device.arch",
        "dtype.input",
        "dtype.key",
        "attrs.bias",
        "dtype.bias",
    }
    assert TRUE.facts_used() == frozenset() and all_of().facts_used() == frozenset()


# --------------------------------------------------------------------------- #
# restrict(): partial evaluation
# --------------------------------------------------------------------------- #


def test_restrict_folds_decided_terms_and_leaves_undecided_ones_symbolic():
    """Absent from ``known`` means "not yet decided", not "not provided"."""
    predicate = all_of(
        device.vendor == "nvidia", device.arch.at_least("sm100"), shape.K % 16 == 0
    )
    reduced = predicate.restrict({"device.vendor": "nvidia", "device.arch": "sm100"})
    assert reduced.render() == "shape.K % 16 == 0"
    assert predicate.restrict({"device.vendor": "cpu"}) is FALSE
    assert (
        all_of(device.vendor == "nvidia", device.arch.at_least("sm90")).restrict(
            {"device.vendor": "nvidia", "device.arch": "sm90"}
        )
        is TRUE
    )
    untouched = dtype.weight == "fp32"
    assert untouched.restrict({"device.vendor": "nvidia"}) is untouched

    either = any_of(quant.format == "bf16", quant.format == "nvfp4")
    assert either.restrict({"quant.format": "bf16"}) is TRUE
    assert either.restrict({"quant.format": "fp8_e4m3"}) is FALSE

    conditional = implies(attrs.bias, dtype.bias == "fp32")
    assert conditional.restrict({"attrs.bias": False}) is TRUE
    assert conditional.restrict({"attrs.bias": True}).render() == "dtype.bias == fp32"


def test_restrict_answers_the_parameter_dtype_question():
    """The motivating use: pick a dtype before any tensor exists. A layer
    allocating gamma cannot ask "what did the selector choose?", but it can
    ask which candidate dtypes keep an implementation eligible."""
    flashinfer_layernorm = all_of(
        device.vendor == "nvidia", dtype.input == "bf16", dtype.weight == "fp32"
    )
    construction = {"device.vendor": "nvidia", "dtype.input": "bf16"}
    feasible = {
        candidate
        for candidate in ("bf16", "fp16", "fp32")
        if flashinfer_layernorm.restrict({**construction, "dtype.weight": candidate})
        is not FALSE
    }
    assert feasible == {"fp32"}


# --------------------------------------------------------------------------- #
# The parenthesization guard
# --------------------------------------------------------------------------- #


def test_missing_parens_raise_instead_of_misbehaving_and_parenthesized_expressions_compose():
    """``a >= 90 & b`` parses as ``a >= (90 & b)``; refuse rather than lie."""
    with pytest.raises(ParensError, match="precedence"):
        _ = bool(shape.K >= 4096)
    with pytest.raises(ParensError):
        _ = bool(device.vendor)
    with pytest.raises(ParensError):
        _ = bool(shape.K % 16)
    predicate = (shape.K >= 4096) & (quant.format == "bf16")
    assert predicate.eval(facts({"shape.K": 4096, "quant.format": "bf16"})) is None


# --------------------------------------------------------------------------- #
# YAML literal parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fact", "literal", "text"),
    [
        (shape.K, ">=4096", "shape.K >= 4096"),
        (shape.K, "4096", "shape.K == 4096"),
        (shape.K, 4096, "shape.K == 4096"),
        (shape.K, "<4096", "shape.K < 4096"),
        (device.arch, ">=sm100", "device.arch >= sm100"),
        (device.arch, "sm90", "device.arch == sm90"),
        (shape.K, "%16", "shape.K % 16 == 0"),
        (shape.K, "%16==0", "shape.K % 16 == 0"),
        (dtype.input, ["bf16", "fp16"], "dtype.input in {bf16, fp16}"),
        (model.tags, ["debug"], "model.tags intersects {'debug'}"),
        (model.tags, "debug", "'debug' in model.tags"),
        (quant.format, None, "quant.format is none"),
    ],
)
def test_literals_parse_by_declared_kind(fact: Fact, literal, text: str):
    assert predicate_from_literal(fact, literal).render() == text


def test_a_list_on_a_scalar_fact_means_membership():
    predicate = predicate_from_literal(device.vendor, ["nvidia", "amd"])
    assert predicate.eval(facts({"device.vendor": "amd"})) is None
    assert predicate.eval(facts({"device.vendor": "cpu"})) is not None


# --------------------------------------------------------------------------- #
# Fact namespaces
# --------------------------------------------------------------------------- #


def test_namespaces_are_closed_case_sensitive_and_ignore_dunders():
    """``shape.K`` and ``shape.k`` are different paths, not aliases: four
    spellings of every dimension used to be accepted, which is how the alias
    table grew to forty entries."""
    with pytest.raises(AttributeError):
        _ = device.archh
    with pytest.raises(AttributeError):
        _ = quant.formatt
    assert shape.K.path != shape.k.path
    with pytest.raises(AttributeError):
        _ = shape.__deepcopy__


def test_namespaces_mint_facts_with_the_right_kind_and_expose_escape_hatches():
    assert shape.M.kind is FactKind.INT and shape.M.path == "shape.M"
    assert dtype.a_log.kind is FactKind.DTYPE
    assert attrs.layout.kind is FactKind.ANY
    field = quant.field("scale_mode") == "ue8m0"
    assert field.render() == "quant.fields.scale_mode == ue8m0"
    assert field.facts_used() == {"quant.fields.scale_mode"}
    # Library availability is an ordinary fact, so tests can fake it and traces explain it.
    available = lib.has("flashinfer")
    assert available.facts_used() == {"lib.flashinfer"}
    assert available.eval(facts({"lib.flashinfer": True})) is None
    assert available.eval(facts({"lib.flashinfer": False})) is not None


# --------------------------------------------------------------------------- #
# The real capability contracts
# --------------------------------------------------------------------------- #
#
# These are the expressions that replaced the hand-written capability closures
# and ``can_handle`` methods. Writing them out here is the evidence that the
# algebra is expressive enough, and that it needs no escape hatch to do it.

NVIDIA_FLASHINFER = lib.has("flashinfer") & (device.vendor == "nvidia")
HALF_FLOATS = frozenset({"bf16", "fp16"})


def test_torch_gemm_fp8_scaled_contract():
    """The original tested ``spec_id.startswith("fp8_")``, so an e5m2 weight
    passed and was fed to ``torch._scaled_mm`` under e4m3 assumptions."""
    contract = all_of(
        quant.format == "fp8_e4m3",
        quant.granularity.in_({"per_tensor", "per_channel"}),
        device.vendor == "nvidia",
        device.arch.at_least("sm89"),
        shape.K % 16 == 0,
        shape.N % 16 == 0,
    )
    base = {
        "quant.format": "fp8_e4m3",
        "quant.granularity": "per_channel",
        "device.vendor": "nvidia",
        "device.arch": "sm90",
        "shape.K": 4096,
        "shape.N": 4096,
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm86"})) is not None
    assert contract.eval(facts({**base, "shape.K": 4095})) is not None
    rejected = contract.eval(facts({**base, "quant.format": "fp8_e5m2"}))
    assert rejected is not None and rejected.predicate == "quant.format == fp8_e4m3"
    # A CPU host reports "unknown", not "too old".
    assert (
        contract.eval(facts({**base, "device.vendor": "cpu", "device.arch": None}))
        is not None
    )


def test_flashinfer_gemm_nvfp4_contract():
    """This box is sm90, so the execution path cannot run locally, but the
    *capability* can be checked against a synthetic device: the concrete
    advantage of eligibility as data rather than a closure."""
    contract = all_of(
        NVIDIA_FLASHINFER,
        quant.format == "nvfp4",
        quant.layout == "128x4",
        quant.block_k == 16,
        device.arch.at_least("sm100"),
        shape.K % 16 == 0,
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "device.arch": "sm100",
        "quant.format": "nvfp4",
        "quant.layout": "128x4",
        "quant.block_k": 16,
        "shape.K": 4096,
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm90"})) is not None
    assert (
        contract.eval(facts({**base, "quant.layout": "linear"})) is not None
    )  # torch's row
    assert contract.eval(facts({**base, "lib.flashinfer": False})) is not None


def test_flashinfer_norm_contracts_use_optional_and_conditional_dtypes():
    """rmsnorm: residual optional for the plain form, required for the fused
    one. layernorm: the beta dtype is only constrained when there is a beta."""
    rmsnorm = all_of(
        NVIDIA_FLASHINFER,
        dtype.input == "bf16",
        dtype.weight == "bf16",
        dtype.residual == "bf16",
    )
    plain = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "dtype.input": "bf16",
        "dtype.weight": "bf16",
    }
    assert rmsnorm.eval(facts(plain, optional={"dtype.residual"})) is None
    assert rmsnorm.eval(facts({**plain, "dtype.residual": "fp32"})) is not None
    assert rmsnorm.eval(facts({**plain, "dtype.residual": "bf16"})) is None
    assert rmsnorm.eval(facts({**plain, "dtype.input": "fp32"})) is not None

    layernorm = all_of(
        NVIDIA_FLASHINFER,
        dtype.input == "bf16",
        dtype.weight == "fp32",
        implies(attrs.bias, dtype.bias == "fp32"),
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "dtype.input": "bf16",
        "dtype.weight": "fp32",
    }
    assert layernorm.eval(facts({**base, "attrs.bias": False})) is None
    assert (
        layernorm.eval(facts({**base, "attrs.bias": True, "dtype.bias": "fp32"}))
        is None
    )
    assert (
        layernorm.eval(facts({**base, "attrs.bias": True, "dtype.bias": "bf16"}))
        is not None
    )


def test_flashinfer_gdn_seven_role_contract():
    """Replaces a 24-line runtime-``raise`` dtype check plus a wrong ``min_sm``:
    the backend gates on ``major in (9, 10)`` but the descriptor only required
    ``sm >= 90``, so an sm120 device passed selection and raised inside the
    backend, where a raise can never drive a fallback."""
    contract = all_of(
        NVIDIA_FLASHINFER,
        device.arch.family_in({"sm9", "sm10"}),
        dtype.input.in_(HALF_FLOATS),
        dtype.key.in_(HALF_FLOATS),
        dtype.value.in_(HALF_FLOATS),
        dtype.a.in_(HALF_FLOATS),
        dtype.b.in_(HALF_FLOATS),
        dtype.a_log == "fp32",
        dtype.dt_bias.in_({"bf16", "fp32"}),
        same(dtype.input, dtype.key, dtype.value),
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "device.arch": "sm90",
        "dtype.input": "bf16",
        "dtype.key": "bf16",
        "dtype.value": "bf16",
        "dtype.a": "bf16",
        "dtype.b": "bf16",
        "dtype.a_log": "fp32",
        "dtype.dt_bias": "bf16",
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm100"})) is None
    sm120 = contract.eval(facts({**base, "device.arch": "sm120"}))
    assert sm120 is not None and sm120.predicate == "device.arch family in {sm9, sm10}"
    assert contract.eval(facts({**base, "dtype.value": "fp16"})) is not None
    assert contract.eval(facts({**base, "dtype.a_log": "bf16"})) is not None


def test_sdpa_layout_contract_is_positive_not_negative():
    """Stated as "padded only", which also excludes ``paged`` and an omitted
    layout; the old form rejected ``ragged`` by name and let both through."""
    contract = attrs.layout == "padded"
    assert contract.eval(facts({"attrs.layout": "padded"})) is None
    assert contract.eval(facts({"attrs.layout": "ragged"})) is not None
    assert contract.eval(facts({"attrs.layout": "paged"})) is not None
    assert contract.eval(facts({})) is not None


def test_remaining_real_contracts_need_no_escape_hatch():
    contracts = {
        "flashinfer prefill head_dim": shape.head_dim.in_({64, 128, 256}),
        "rmsnorm_silu_mul hidden": shape.hidden <= 8192,
        "causal_conv kernel": shape.kernel.between(1, 8),
        "rope full rotary": shape.rotary_dim == shape.head_dim,
        "debug tag": model.tags.has("debug"),
    }
    for predicate in contracts.values():
        assert not isinstance(predicate, Const)
        assert predicate.render() and predicate.facts_used()
