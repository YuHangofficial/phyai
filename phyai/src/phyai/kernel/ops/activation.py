"""Define directly selectable gated and plain activation kernels."""

from __future__ import annotations

from phyai.kernel.facts import lib, attrs, dtype, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of

ACTIVATION = OpSpec(
    name="activation",
    dims=("tokens", "hidden"),
    dtypes=("input",),
    attributes=("activation", "gated"),
    signature="(x) -> Tensor",
    doc=(
        "Elementwise activation, optionally consuming a gate half. Exact and "
        "tanh-approximated GELU are separate kernel IDs."
    ),
)

# FlashInfer supports only gated forms with half-precision inputs.
FLASHINFER_GATED = all_of(
    lib.has("flashinfer"),
    device.vendor == "nvidia",
    attrs.gated,
    dtype.input.in_({"bf16", "fp16"}),
)


def _flashinfer(function_name: str):
    """Prepare one named FlashInfer activation function."""

    def prepare(facts, params):
        from flashinfer import activation as module

        return getattr(module, function_name)

    return prepare


def _torch(name: str, *, gated: bool):
    """Prepare one exact Torch activation topology."""

    def prepare(facts, params):
        import torch

        activations = {
            "silu": torch.nn.functional.silu,
            "gelu": lambda value: torch.nn.functional.gelu(value, approximate="none"),
            "gelu_tanh": lambda value: torch.nn.functional.gelu(
                value, approximate="tanh"
            ),
        }
        activate = activations[name]

        def execute(value):
            if not gated:
                return activate(value)
            lhs, rhs = value.chunk(2, dim=-1)
            return activate(lhs) * rhs

        return execute

    return prepare


FLASHINFER_KERNELS = (
    ("silu", "silu_and_mul"),
    ("gelu", "gelu_and_mul"),
    ("gelu_tanh", "gelu_tanh_and_mul"),
)

TORCH_KERNELS = (
    ("silu", False, "silu"),
    ("gelu", False, "gelu"),
    ("gelu_tanh", False, "gelu_tanh"),
    ("silu", True, "silu_and_mul"),
    ("gelu", True, "gelu_and_mul"),
    ("gelu_tanh", True, "gelu_tanh_and_mul"),
)


def register(catalog: Catalog) -> None:
    catalog.register_op(ACTIVATION)
    catalog.register_many(
        tuple(
            Impl(
                kernel_id=f"flashinfer.activation.{function_name}",
                op="activation",
                priority=Priority.OPTIMIZED + 2,
                when=all_of(FLASHINFER_GATED, attrs.activation == activation),
                prepare=_flashinfer(function_name),
                metadata={
                    "package": "flashinfer",
                    "activation": activation,
                    "gated": True,
                },
            )
            for activation, function_name in FLASHINFER_KERNELS
        )
        + tuple(
            Impl(
                kernel_id=f"torch.activation.{kernel_name}",
                op="activation",
                priority=Priority.REFERENCE,
                reference=True,
                when=all_of(
                    attrs.activation == activation,
                    attrs.gated == gated,
                ),
                prepare=_torch(activation, gated=gated),
                metadata={
                    "package": "torch",
                    "activation": activation,
                    "gated": gated,
                },
            )
            for activation, gated, kernel_name in TORCH_KERNELS
        )
    )


__all__ = ["ACTIVATION", "register"]
