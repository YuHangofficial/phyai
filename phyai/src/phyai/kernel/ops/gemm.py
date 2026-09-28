"""Define GEMM implementations by backend and storage format."""

from __future__ import annotations

from phyai.kernel.facts import lib, dtype, quant, shape, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of


def _bench_args(facts, device):
    """Synthesize a dense GEMM call for autotune; quantized layouts decline.

    A quantized layer carries scale tensors in backend-specific layouts, so a
    faked layer would measure the wrong thing. Those calls raise, and the
    catalog's priority order stands.
    """

    import torch
    from types import SimpleNamespace

    from phyai.kernel.call import torch_dtype
    from phyai.kernel.facts import MISSING

    if facts.lookup("quant.format") not in {"bf16", "fp16"}:
        raise NotImplementedError("only dense GEMM calls are synthesized")
    m, n, k = (facts.lookup(f"shape.{name}") for name in ("M", "N", "K"))
    if MISSING in (m, n, k):
        raise NotImplementedError("benchmark needs the M, N and K dimensions")
    dt = torch_dtype(str(facts.lookup("dtype.input")))
    x = torch.randn(int(m), int(k), device=device, dtype=dt)
    weight = torch.randn(int(n), int(k), device=device, dtype=dt)
    return (SimpleNamespace(weight=weight), x, None)


GEMM = OpSpec(
    name="gemm",
    dims=("M", "N", "K"),
    dtypes=("input", "output"),
    optional_dtypes=("weight",),
    attributes=(),
    signature="(layer, x, bias) -> Tensor",
    bench_args=_bench_args,
    doc="Linear projection, possibly with quantized weights.",
)


NVIDIA_FLASHINFER = lib.has("flashinfer") & (device.vendor == "nvidia")


# Shared capability for torch FP8 scaled matrix multiplication.
TORCH_SCALED_MM = all_of(
    quant.format == "fp8_e4m3",
    device.vendor == "nvidia",
    device.arch.at_least("sm89"),
    shape.K % 16 == 0,
    shape.N % 16 == 0,
)

# Torch block-scaled FP8 uses the reference dequantize-and-matmul path.
TORCH_FP8_BLOCK = all_of(
    quant.format == "fp8_e4m3",
    quant.granularity == "block",
    device.vendor == "nvidia",
    device.arch.at_least("sm89"),
)


def _flashinfer_bf16(facts, params):
    from phyai.layers.linear.backends.flashinfer import gemm_bf16

    return gemm_bf16


def _flashinfer_fp8_block(facts, params):
    from phyai.layers.linear.backends.flashinfer import gemm_block_fp8

    return gemm_block_fp8


def _flashinfer_nvfp4(facts, params):
    from phyai.layers.linear.backends.flashinfer import gemm_nvfp4

    return gemm_nvfp4


def _torch_fp8_per_tensor(facts, params):
    from phyai.layers.linear.backends.torch import gemm_fp8_per_tensor

    return gemm_fp8_per_tensor


def _torch_fp8_per_channel(facts, params):
    from phyai.layers.linear.backends.torch import gemm_fp8_per_channel

    return gemm_fp8_per_channel


def _torch_fp8_block(facts, params):
    from phyai.layers.linear.backends.torch import gemm_fp8_block

    return gemm_fp8_block


def _torch_nvfp4_reference(facts, params):
    from phyai.layers.linear.backends.torch import gemm_nvfp4_reference

    return gemm_nvfp4_reference


def _torch_dense(facts, params):
    """Prepare the unquantized torch linear implementation."""

    import torch.nn.functional as F

    def dense(layer, x, bias):
        return F.linear(x, layer.weight, bias)

    return dense


def register(catalog: Catalog) -> None:
    catalog.register_op(GEMM)

    rows = (
        # FlashInfer.
        Impl(
            kernel_id="flashinfer.gemm.bf16",
            op="gemm",
            priority=Priority.OPTIMIZED + 2,
            when=all_of(
                NVIDIA_FLASHINFER,
                quant.format == "bf16",
                device.arch.at_least("sm80"),
                dtype.input == "bf16",
            ),
            prepare=_flashinfer_bf16,
            metadata={"package": "flashinfer", "note": "mm_bf16"},
        ),
        Impl(
            kernel_id="flashinfer.gemm.fp8_block",
            op="gemm",
            priority=Priority.SPECIALIZED,
            when=all_of(
                NVIDIA_FLASHINFER,
                quant.format == "fp8_e4m3",
                quant.granularity == "block",
                device.arch.at_least("sm100"),
            ),
            prepare=_flashinfer_fp8_block,
            metadata={"package": "flashinfer", "note": "gemm_fp8_nt_groupwise"},
        ),
        Impl(
            kernel_id="flashinfer.gemm.nvfp4_128x4",
            op="gemm",
            priority=Priority.SPECIALIZED,
            when=all_of(
                NVIDIA_FLASHINFER,
                quant.format == "nvfp4",
                quant.layout == "128x4",
                quant.block_k == 16,
                device.arch.at_least("sm100"),
                shape.K % 16 == 0,
            ),
            prepare=_flashinfer_nvfp4,
            metadata={"package": "flashinfer"},
        ),
        # Torch.
        Impl(
            kernel_id="torch.gemm.bf16",
            op="gemm",
            priority=Priority.REFERENCE,
            reference=True,
            when=quant.format.in_({"bf16", "fp16"}),
            prepare=_torch_dense,
            metadata={"package": "torch", "note": "F.linear"},
        ),
        Impl(
            kernel_id="torch.gemm.fp8_per_tensor",
            op="gemm",
            priority=Priority.GENERAL,
            when=all_of(TORCH_SCALED_MM, quant.granularity == "per_tensor"),
            prepare=_torch_fp8_per_tensor,
            metadata={"package": "torch", "note": "torch._scaled_mm"},
        ),
        Impl(
            kernel_id="torch.gemm.fp8_per_channel",
            op="gemm",
            priority=Priority.GENERAL,
            when=all_of(TORCH_SCALED_MM, quant.granularity == "per_channel"),
            prepare=_torch_fp8_per_channel,
            metadata={"package": "torch", "note": "torch._scaled_mm"},
        ),
        Impl(
            kernel_id="torch.gemm.fp8_block",
            op="gemm",
            priority=Priority.REFERENCE,
            reference=True,
            when=TORCH_FP8_BLOCK,
            prepare=_torch_fp8_block,
            metadata={"package": "torch", "note": "dequantize reference"},
        ),
        Impl(
            kernel_id="torch.gemm.nvfp4_linear",
            op="gemm",
            priority=Priority.REFERENCE,
            reference=True,
            when=all_of(
                quant.format == "nvfp4",
                quant.layout == "linear",
                quant.block_k == 16,
            ),
            prepare=_torch_nvfp4_reference,
            metadata={"package": "torch", "note": "unpack + dequantize reference"},
        ),
    )
    catalog.register_many(rows)


__all__ = ["GEMM", "register"]
