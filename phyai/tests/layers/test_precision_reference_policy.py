"""Precision contracts must survive real selector policies and input layouts."""

import os

import pytest
import torch
import torch.nn.functional as F

from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.kernel.policy import policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.layers.layer_norm import LayerNorm, RMSNorm
from phyai.layers.linear import ReplicatedLinear


def make_selector(mode, device):
    catalog = build_catalog()
    config = {"schema": "phyai.kernel/v1", "defaults": {"fallback": "reference"}}
    if mode != "default":
        key = "overrides" if mode == "strict" else "rules"
        config[key] = []
        for op, accelerated in (
            ("gemm", "flashinfer.gemm.bf16"),
            ("layernorm", "flashinfer.layernorm"),
            ("rmsnorm", "flashinfer.rmsnorm"),
        ):
            rule = {"id": op, "match": {"op": op}}
            if mode == "torch":
                rule["restrict_to"] = f"torch.{op}*"
            elif mode == "reference":
                rule["restrict_to"] = (
                    "torch.gemm.fp32_compute"
                    if op == "gemm"
                    else f"torch.{op}.explicit_fp32"
                )
            else:
                rule["use" if mode == "strict" else "prefer"] = (
                    [accelerated] if mode != "strict" else accelerated
                )
            config[key].append(rule)
        if mode == "error":
            config["defaults"]["fallback"] = "error"
    return Selector(catalog, policy_from_mapping(config, catalog), device=device)


@pytest.mark.parametrize("mode", ["default", "torch", "reference", "flashinfer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [7, 63, 129])
def test_norm_precision_policy_and_strides(mode, dtype, width):
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    torch.manual_seed(width)
    x = (torch.randn(2, 3, 2 * width, dtype=dtype, device=device) * 0.02 + 4)[..., ::2]
    assert not x.is_contiguous()
    with kernel_selector_scope(make_selector(mode, device)):
        for eps in (1e-6, 1e-3):
            ln = LayerNorm(
                width, dtype=dtype, device=device, eps=eps, explicit_fp32_math=True
            )
            rms = RMSNorm(
                width,
                dtype=dtype,
                device=device,
                eps=eps,
                explicit_fp32_math=True,
                cast_before_affine=True,
            )
            ln.weight.data.normal_()
            ln.bias.data.normal_()
            rms.weight.data.normal_()
            centered = x.float() - x.float().mean(-1, keepdim=True)
            expected_ln = (
                centered
                * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
                * ln.weight.float()
                + ln.bias.float()
            ).to(dtype)
            expected_rms = rms.weight * (
                x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            ).to(dtype)
            torch.testing.assert_close(ln(x), expected_ln, atol=0, rtol=0)
            torch.testing.assert_close(rms(x), expected_rms, atol=0, rtol=0)
            for layer, op in ((ln, "layernorm"), (rms, "rmsnorm")):
                assert {
                    h.kernel_id
                    for h in (
                        layer._call if op == "layernorm" else layer._plain_call
                    )._memo.values()
                } == {f"torch.{op}.explicit_fp32"}


@pytest.mark.parametrize("mode", ["default", "torch", "reference", "flashinfer"])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((0, 63), torch.bfloat16),
        ((1, 63), torch.float32),
        ((3, 63), torch.float16),
        ((2, 3, 63), torch.bfloat16),
    ],
)
def test_router_reference_real_layer(mode, shape, dtype, fake_mesh):
    fake_mesh()
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    with kernel_selector_scope(make_selector(mode, device)):
        layer = ReplicatedLinear(
            63,
            37,
            bias=True,
            params_dtype=torch.bfloat16,
            device=device,
            fp32_compute=True,
        )
        layer.weight.data.normal_()
        layer.bias.data.normal_()
        x = torch.randn(*shape[:-1], 126, dtype=dtype, device=device)[..., ::2]
        output, bias = layer(x)
        torch.testing.assert_close(
            output,
            F.linear(x.float(), layer.weight.float(), layer.bias.float()),
            atol=0,
            rtol=0,
        )
        assert bias is None and output.dtype == torch.float32
        assert {h.kernel_id for h in layer.gemm_call._memo.values()} == {
            "torch.gemm.fp32_compute"
        }


@pytest.mark.parametrize("mode", ["strict", "error"])
def test_reference_does_not_override_explicit_failure_policy(mode):
    from phyai.kernel.types import KernelQuery

    selector = make_selector(mode, "cpu")
    for op, dtype, attrs, shape, quant in [
        (
            "gemm",
            {"input": "fp32", "weight": "bf16", "output": "fp32"},
            {},
            {"M": 3, "N": 37, "K": 63},
            {"format": "bf16"},
        ),
        (
            "layernorm",
            {"input": "bf16", "weight": "bf16", "bias": "bf16"},
            {"math": "explicit_fp32", "bias": True},
            {"tokens": 3, "hidden": 63},
            None,
        ),
        (
            "rmsnorm",
            {"input": "bf16", "weight": "bf16"},
            {
                "math": "explicit_fp32",
                "cast_before_affine": True,
                "variant": "rms",
                "rank": 2,
            },
            {"tokens": 3, "hidden": 63},
            None,
        ),
    ]:
        query = KernelQuery.build(
            op, device="cpu", dtype=dtype, attrs=attrs, shape=shape, quant=quant
        )
        assert selector.explain(query).selected is None
