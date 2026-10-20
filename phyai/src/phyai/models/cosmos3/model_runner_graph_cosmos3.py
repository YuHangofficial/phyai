"""Bounded CUDA graph state for a complete Cosmos3 Generator forward."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Hashable

import torch

from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph


class Cosmos3CudaGraphRunner:
    """Keep one graph and refresh request constants independently of step inputs.

    Every replay executes the complete captured function. Outputs share graph
    storage and must be consumed or cloned before the next call. A shape, dtype,
    device, layout, or ``graph_key`` change releases the previous graph.
    """

    def __init__(
        self, *, use_cuda_graph: bool = True, num_warmup_iters: int = 3
    ) -> None:
        if num_warmup_iters < 0:
            raise ValueError("num_warmup_iters must be non-negative")
        self.use_cuda_graph = bool(use_cuda_graph)
        self.num_warmup_iters = int(num_warmup_iters)
        self.graph: CudaGraph | None = None
        self.num_captures = 0
        self.num_replays = 0
        self._signature: tuple | None = None
        self._static_key: Hashable | None = None
        self._captured_fn: Callable[..., Any] | None = None

    @staticmethod
    def tensor_signature(inputs: dict[str, torch.Tensor]) -> tuple:
        """Describe tensor contracts without depending on allocation addresses."""
        return tuple(
            (name, tuple(t.shape), t.dtype, t.device, t.layout, tuple(t.stride()))
            for name, t in sorted(inputs.items())
        )

    def reset(self) -> None:
        """Release the captured executable and all static input/output buffers."""
        if self.graph is not None:
            self.graph.reset()
        self.graph = None
        self._signature = None
        self._static_key = None
        self._captured_fn = None

    @torch.no_grad()
    def forward(
        self,
        fn: Callable[..., Any],
        inputs: dict[str, torch.Tensor],
        *,
        static_inputs: dict[str, torch.Tensor] | None = None,
        static_key: Hashable | None = None,
        graph_key: Hashable | None = None,
    ) -> Any:
        """Copy inputs and run ``fn(**inputs, **static_inputs)`` or its graph.

        ``inputs`` are refreshed every call. A non-None ``static_key`` is an
        explicit revision supplied by the caller: static tensor values must not
        change without changing this key. With no key they are always copied.
        ``graph_key`` describes non-tensor arguments closed over by ``fn``;
        change it, or call ``reset()``, when those arguments or weights change.
        """
        static_inputs = {} if static_inputs is None else static_inputs
        overlap = inputs.keys() & static_inputs.keys()
        if overlap:
            raise ValueError(f"Inputs cannot also be static: {sorted(overlap)}")
        all_inputs = inputs | static_inputs
        if not all_inputs:
            raise ValueError("A Cosmos3 graph requires at least one tensor input")
        if not self.use_cuda_graph or not all(t.is_cuda for t in all_inputs.values()):
            return fn(**all_inputs)
        devices = {tensor.device for tensor in all_inputs.values()}
        if len(devices) != 1:
            raise ValueError("All inputs of a Cosmos3 graph must use the same device")
        signature = (
            graph_key,
            self.tensor_signature(inputs),
            self.tensor_signature(static_inputs),
        )
        if signature != self._signature:
            self.reset()
            graph = CudaGraph(num_warmup_iters=self.num_warmup_iters)

            def capture_forward(**buffers: torch.Tensor) -> Any:
                # Warmup must select the same capture-safe backends as capture;
                # otherwise their first-use plans would be built inside the graph.
                with graph_capture():
                    return fn(**buffers)

            try:
                with torch.cuda.device(next(iter(devices))):
                    graph.capture(capture_forward, all_inputs)
            except Exception:
                graph.reset()
                raise
            self.graph = graph
            # Closures may own attention metadata tensors read by captured
            # kernels. Keep those external allocations alive until reset.
            self._captured_fn = capture_forward
            self._signature = signature
            self._static_key = static_key
            self.num_captures += 1
        assert self.graph is not None
        updates = inputs
        if static_key is None or static_key != self._static_key:
            updates = all_inputs
        output = self.graph.replay(updates)
        self._static_key = static_key
        self.num_replays += 1
        return output


__all__ = ["Cosmos3CudaGraphRunner"]
