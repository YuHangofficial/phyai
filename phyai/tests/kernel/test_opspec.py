"""Operation specifications and implementation rows.

Three properties matter: an unknown fact path is a loud error with a suggestion
(the old matcher returned ``False`` for an unrecognised key, turning a typo into
a rule that could never match while every test stayed green); a capability that
only constrains optional facts is rejected, because omitting those facts would
make the row unconditionally eligible; and parameter dtypes are derived from
declared contracts, which is what lets a layer allocate gamma before any input
tensor exists.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import attrs, device, dtype, lib, quant, shape
from phyai.kernel.opspec import (
    Impl,
    OpSpec,
    ParamContract,
    ParamRule,
    Priority,
    Returns,
    any_float,
    fixed,
    matches_activation,
    resolve_param_dtypes,
    returns_callable,
    returns_instance,
)
from phyai.kernel.predicate import all_of, implies


GEMM = OpSpec(
    name="gemm",
    dims=("M", "N", "K"),
    dtypes=("input", "output"),
    optional_dtypes=("weight",),
    attributes=(),
    signature="(layer, x, bias) -> Tensor",
)

LAYERNORM = OpSpec(
    name="layernorm",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    optional_dtypes=("bias",),
    attributes=("bias",),
    params=("weight", "bias"),
    signature="(x, weight, bias, eps) -> Tensor",
)


def impl(when, **kwargs) -> Impl:
    defaults = {
        "kernel_id": "test.gemm",
        "op": "gemm",
        "when": when,
        "prepare": lambda facts: None,
    }
    return Impl(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# Schema declaration and path validation
# --------------------------------------------------------------------------- #


def test_schema_declares_its_paths_and_validates_itself():
    assert GEMM.known_paths() == {
        "shape.M",
        "shape.N",
        "shape.K",
        "dtype.input",
        "dtype.output",
        "dtype.weight",
    }
    assert LAYERNORM.optional_paths == {"dtype.bias"}
    assert OpSpec(name="  GEMM ").name == "gemm"
    with pytest.raises(ValueError, match="non-empty"):
        OpSpec(name="   ")
    with pytest.raises(ValueError, match="both required and optional"):
        OpSpec(name="bad", dtypes=("input",), optional_dtypes=("input",))
    # Reference rows are required unless the op opts out; capture_safe is per row.
    assert OpSpec(name="gemm").requires_reference
    assert not OpSpec(
        name="attention_paged", requires_reference=False
    ).requires_reference
    assert impl(device.arch.at_least("sm90")).capture_safe
    assert not impl(device.arch.at_least("sm90"), capture_safe=False).capture_safe


def test_unknown_paths_are_loud_errors_with_suggestions():
    """``shape.k`` is a typo, not an alias, answered by an exact
    case-insensitive lookup rather than ``difflib`` (which scores N/M/K
    identically for this input and would break the tie arbitrarily)."""
    with pytest.raises(ValueError, match=r"did you mean 'shape\.K'"):
        GEMM.validate_paths({"shape.KK"}, context="test")
    with pytest.raises(ValueError, match="case-sensitive") as excinfo:
        GEMM.validate_paths({"shape.k"}, context="test")
    assert "did you mean 'shape.K'" in str(excinfo.value)
    with pytest.raises(ValueError, match="has no fact 'dtype.residual'"):
        GEMM.validate_paths({"dtype.residual"}, context="test")
    # device.* / quant.* / lib.* are declared centrally, not per op.
    GEMM.validate_paths(
        {"device.arch", "quant.format", "lib.flashinfer", "op", "mode"}, context="test"
    )


def test_check_against_validates_op_paths_params_and_vacuity():
    impl(all_of(device.arch.at_least("sm90"), shape.K % 16 == 0)).check_against(GEMM)
    impl(all_of(quant.format == "nvfp4", quant.field("mode") == "x")).check_against(
        GEMM
    )
    with pytest.raises(ValueError, match=r"capability of 'test\.gemm'"):
        impl(all_of(device.arch.at_least("sm90"), shape.KK % 16 == 0)).check_against(
            GEMM
        )
    with pytest.raises(ValueError, match="registered under 'gemm'"):
        impl(shape.K % 16 == 0, op="layernorm").check_against(GEMM)
    with pytest.raises(ValueError, match="has no parameter"):
        Impl(
            kernel_id="test.layernorm",
            op="layernorm",
            when=dtype.input == "bf16",
            prepare=lambda facts: None,
            params={"gamma": fixed("fp32")},
        ).check_against(LAYERNORM)
    # A capability constraining only optional facts would be always eligible
    # once those facts are omitted; a device-only gate is legitimate.
    with pytest.raises(ValueError, match="only optional facts"):
        impl(dtype.weight == "bf16").check_against(GEMM)
    impl(all_of(dtype.input == "bf16", dtype.weight == "bf16")).check_against(GEMM)
    impl(all_of(device.vendor == "nvidia", lib.has("flashinfer"))).check_against(GEMM)


def test_the_real_flashinfer_layernorm_row_passes_and_derives_its_library():
    row = Impl(
        kernel_id="  FlashInfer.LayerNorm  ",
        op="layernorm",
        when=all_of(
            lib.has("flashinfer"),
            device.vendor == "nvidia",
            dtype.input == "bf16",
            dtype.weight == "fp32",
            implies(attrs.bias, dtype.bias == "fp32"),
        ),
        prepare=lambda facts: None,
        params={"weight": fixed("fp32"), "bias": fixed("fp32")},
    )
    row.check_against(LAYERNORM)
    assert row.kernel_id == "flashinfer.layernorm"  # normalized
    assert row.libraries == {"flashinfer"}  # derived from the capability
    assert impl(device.arch.at_least("sm90")).libraries == frozenset()
    with pytest.raises(ValueError, match="non-empty"):
        impl(device.arch.at_least("sm90"), kernel_id="  ")


# --------------------------------------------------------------------------- #
# Return conventions
# --------------------------------------------------------------------------- #


def test_return_conventions():
    """Paged attention backends own runner-scoped buffers and cannot be built
    at selection time, so an op may declare that its result *is* an object
    constructed with the runner rather than adding an ad-hoc calling convention."""
    assert returns_callable().kind == "callable" and not returns_callable().is_instance

    class ARBackend:
        pass

    returns = returns_instance(ARBackend, constructed_with=("runner",))
    assert returns.is_instance and returns.protocol is ARBackend
    assert returns.constructed_with == ("runner",)
    with pytest.raises(ValueError, match="unknown Returns kind"):
        Returns("magic")


# --------------------------------------------------------------------------- #
# Parameter contracts
# --------------------------------------------------------------------------- #


def test_layernorm_derives_fp32_gamma():
    """The value matches the old hardcoded choice (``float32 if backend ==
    "flashinfer" ...``) but is now derived: a bf16-gamma kernel changes the
    derivation instead of requiring that string test to be edited."""
    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="flashinfer.layernorm",
            op="layernorm",
            priority=Priority.OPTIMIZED + 2,
            params={"weight": fixed("fp32"), "bias": fixed("fp32")},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="phyai_kernel.layernorm",
            op="layernorm",
            priority=Priority.OPTIMIZED,
            params={"weight": any_float(), "bias": any_float()},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="torch.layernorm",
            op="layernorm",
            priority=Priority.REFERENCE,
            params={"weight": matches_activation(), "bias": matches_activation()},
        ),
    ]
    assert resolve_param_dtypes(LAYERNORM, impls, activation="bf16") == {
        "weight": "fp32",
        "bias": "fp32",
    }


def test_rmsnorm_derives_activation_dtype_params():
    """FlashInfer RMSNorm reads gamma through the input type, previously a prose
    footgun warning. This is also the case that rules out "keep *some*
    implementation eligible": fp32 gamma would satisfy the torch reference while
    disqualifying the fast kernel."""
    spec = OpSpec(name="rmsnorm", dtypes=("input", "weight"), params=("weight",))
    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="flashinfer.rmsnorm",
            op="rmsnorm",
            priority=Priority.OPTIMIZED + 2,
            params={"weight": matches_activation()},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="torch.rmsnorm",
            op="rmsnorm",
            priority=Priority.REFERENCE,
            params={"weight": any_float()},
        ),
    ]
    assert resolve_param_dtypes(spec, impls, activation="bf16") == {"weight": "bf16"}
    # With an fp32 activation the fast kernel is out of reach; the reference decides.
    assert resolve_param_dtypes(spec, impls, activation="fp32") == {"weight": "fp32"}


def test_preferences_defaults_and_unsatisfiable_contracts():
    spec = OpSpec(name="norm", dtypes=("input",), params=("weight",))
    lax = [
        impl(
            dtype.input == "bf16",
            kernel_id="lax.norm",
            op="norm",
            params={"weight": any_float()},
        )
    ]
    assert resolve_param_dtypes(
        spec, lax, activation="bf16", preferred={"weight": "bf16"}
    ) == {"weight": "bf16"}
    # No contract means no reason to deviate from the activation dtype: fp32
    # affine parameters in a bf16 model cost memory and per-forward casts.
    unconstrained = [impl(dtype.input == "bf16", kernel_id="lax.norm", op="norm")]
    assert resolve_param_dtypes(spec, unconstrained, activation="bf16") == {
        "weight": "bf16"
    }
    assert resolve_param_dtypes(spec, unconstrained, activation="fp32") == {
        "weight": "fp32"
    }
    # Better to learn at construction than at the first forward.
    only = [
        impl(
            dtype.input == "bf16",
            kernel_id="only.norm",
            op="norm",
            params={"weight": fixed("fp8_e4m3")},
        )
    ]
    with pytest.raises(ValueError, match="no dtype satisfies parameter 'weight'"):
        resolve_param_dtypes(spec, only, activation="bf16")
    with pytest.raises(ValueError, match="requires a dtype"):
        ParamContract(ParamRule.FIXED)
    with pytest.raises(ValueError, match="must not name a dtype"):
        ParamContract(ParamRule.ANY_FLOAT, "fp32")
