"""FlashQLA Gated Delta Net backend.

FlashQLA (https://github.com/QwenLM/FlashQLA) provides TileLang fused
kernels for GDN chunked prefill, measured 2-3x faster than the FLA Triton
kernels on Hopper and Blackwell. It ships no decode kernel, so decode
rides FLA's fused recurrent op -- the catalog row therefore requires both
libraries.

Unlike FLA's extended entry point, FlashQLA takes precomputed gates: the
forget gate in log space and an already-sigmoided beta. Its QK L2 norm
flag runs as a host-side normalization pass, not inside the fused kernel,
so it is safe to use directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from phyai.layers.attention.enums import AttnMode
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn.backends.fla import FlaGatedDeltaNetBackend


def _load_chunk_op() -> Callable[..., Any]:
    try:
        from flash_qla import chunk_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "backend='flash-qla' requires flash-qla (pip install flash-qla)."
        ) from exc
    return chunk_gated_delta_rule


@dataclass(frozen=True)
class FlashQlaGatedDeltaNetPlan(GatedDeltaNetPlanHandle):
    """FlashQLA GDN needs no separate planning object."""


class FlashQlaGatedDeltaNetBackend(GatedDeltaNetBackend):
    """Route GDN prefill through FlashQLA and decode through FLA."""

    name = "flash_qla"

    def __init__(self, runner=None) -> None:
        del runner
        # FlashQLA ships chunked prefill only; single-token decode reuses
        # FLA's fused recurrent kernel, including its state-pool handling.
        self._decode = FlaGatedDeltaNetBackend()

    def init_forward_metadata(
        self, meta: GatedDeltaNetMetadata
    ) -> GatedDeltaNetPlanHandle:
        return FlashQlaGatedDeltaNetPlan()

    def forward(
        self,
        layer: GatedDeltaNetLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ctx: GatedDeltaNetCtx,
    ) -> torch.Tensor:
        if ctx.mode == AttnMode.IDLE:
            shape = (*q.shape[:-2], layer.num_state_heads, layer.head_dim)
            return q.new_zeros(shape)
        if ctx.mode == AttnMode.DECODE:
            return self._decode.forward(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        if ctx.mode == AttnMode.PREFILL:
            return self._forward_prefill(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        raise NotImplementedError(
            f"FlashQLA GDN does not support mode={ctx.mode.name}."
        )

    def _forward_prefill(
        self,
        layer: GatedDeltaNetLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ctx: GatedDeltaNetCtx,
    ) -> torch.Tensor:
        if ctx.state_indices is not None or ctx.output_state_indices is not None:
            raise ValueError(
                "FlashQLA GDN prefill does not accept state-pool indices; "
                "pass per-batch state/output_state tensors."
            )

        ragged = ctx.layout.is_ragged()
        q_input = q.unsqueeze(0) if ragged else q
        k_input = k.unsqueeze(0) if ragged else k
        v_input = v.unsqueeze(0) if ragged else v
        a_input = a.unsqueeze(0) if ragged else a
        b_input = b.unsqueeze(0) if ragged else b
        cu_seqlens = None
        if ragged:
            if ctx.cu_seqlens is None:
                raise ValueError("FlashQLA GDN ragged prefill requires ctx.cu_seqlens.")
            cu_seqlens = ctx.cu_seqlens.to(device=q.device, dtype=torch.int64)

        # Precomputed gates: log-space decay and sigmoided beta, in fp32.
        g = -a_log.exp() * F.softplus(a_input.float() + dt_bias.float())
        beta = b_input.float().sigmoid()

        out, final_state = _load_chunk_op()(
            q_input,
            k_input,
            v_input,
            g=g,
            beta=beta,
            scale=layer.scale,
            initial_state=ctx.state,
            output_final_state=ctx.output_state is not None,
            use_qk_l2norm_in_kernel=layer.use_qk_l2norm,
            cu_seqlens=cu_seqlens,
            # phyai state pools store the recurrent state V-first.
            state_v_first=True,
        )
        if ctx.output_state is not None:
            ctx.output_state.copy_(final_state)
        if ragged:
            out = out.squeeze(0)
        if ctx.output is not None:
            ctx.output.copy_(out)
            return ctx.output
        return out


__all__ = [
    "FlashQlaGatedDeltaNetBackend",
    "FlashQlaGatedDeltaNetPlan",
]
