"""Shared lifecycle contract for the attention subsystems.

The three stacks (``paged``, ``nocache``, ``gdn``) differ in their tensor
contracts — the paged one scatters K/V into a pool, the no-cache one is the
only rectangular ``S_q != S_kv`` stack, GDN takes eight tensors — but they
share one *metadata lifecycle*: a host-side description of the step is turned
into a backend-private plan handle once, outside any captured region, and the
per-call forward only consumes it. This module states that lifecycle once;
before it, the same six-method ABC and the same metadata validation were
written out four times across the subsystem base files.

What is deliberately NOT shared: the per-call ``Ctx`` dataclasses (their value
is precise per-stack field typing) and the ``forward`` signature (it genuinely
differs per stack), so each subsystem base declares its own abstract
``forward`` and its own ctx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode


@dataclass(frozen=True)
class BaseAttnMetadata:
    """Mode / layout / count core shared by every stack's metadata.

    Subclasses add their stack-specific tensors and extend
    ``__post_init__`` (calling ``super().__post_init__()`` first) with
    their own layout rules.
    """

    mode: AttnMode
    layout: AttnLayout
    batch_size: int
    num_query_tokens: int

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.num_query_tokens < 0:
            raise ValueError(
                f"{type(self).__name__}: batch_size={self.batch_size}, "
                f"num_query_tokens={self.num_query_tokens} must be non-negative."
            )


class AttnPlanHandleBase:
    """Backend-private per-step state.

    Stability invariant
    -------------------
    Backends that support CUDA graph capture MUST keep the handle's
    tensor / wrapper references stable across replays: the graph
    captures Python identity, so substituting a fresh handle on replay
    invalidates capture. Layers thread the handle through their ctx
    opaquely — only the matching backend cracks it open.
    """


MetaT = TypeVar("MetaT", bound=BaseAttnMetadata)
PlanT = TypeVar("PlanT", bound=AttnPlanHandleBase)


class AttentionBackendBase(ABC, Generic[MetaT, PlanT]):
    """Metadata lifecycle shared by every attention backend.

    Backends are rows in the kernel catalog
    (:mod:`phyai.kernel.ops.attention`); selection constructs the class
    directly. :attr:`name` is the backend's canonical name — the first
    segment of its kernel IDs.

    Lifecycle
    ---------
    * :meth:`init_cuda_graph_state` — once at runner setup, allocates every
      static buffer touched inside a captured graph (default no-op).
    * :meth:`init_capture_metadata` — plan with a representative shape so
      capture has valid kernel state (default delegates to
      :meth:`init_forward_metadata`).
    * :meth:`replay_metadata` — refresh static buffer *contents* in place
      before a graph replay; identities must not change (default no-op).
    * :meth:`init_forward_metadata` — eagerly plan one step; the returned
      handle is written to the ctx the layer receives.

    Each subsystem base declares its own abstract ``forward`` — the call
    signatures genuinely differ (paged scatters K/V into a pool, GDN takes
    eight tensors), so a shared one would be a lie.
    """

    name: ClassVar[str]

    def supports_capture(self) -> bool:
        """Whether the per-call hot path is safe inside a captured graph."""
        return False

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: object,
    ) -> None:
        """Allocate every static buffer the backend touches inside a graph.

        Called once at runner setup. After this returns the backend
        MUST hold every device-resident tensor at a stable address —
        :meth:`replay_metadata` may then update their contents but
        not their identity. Default no-op for backends without static
        state. Subsystems may extend the signature (the paged stack
        adds ``max_paged_kv_indices``).
        """
        return None

    def init_capture_metadata(self, seed_meta: MetaT) -> PlanT:
        """Plan with a representative shape so capture has valid kernel state."""
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(self, plan: PlanT, replay_meta: MetaT) -> None:
        """Update the backend's static buffers in place. Default no-op."""
        return None

    @abstractmethod
    def init_forward_metadata(self, meta: MetaT) -> PlanT:
        """Eagerly plan one step. Returns a handle written to the ctx."""


__all__ = [
    "AttentionBackendBase",
    "AttnPlanHandleBase",
    "BaseAttnMetadata",
]
