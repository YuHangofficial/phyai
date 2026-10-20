"""Bounded CUDA graphs for native Cosmos3 VAE encoding and decoding."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import torch

from phyai.kernel.bootstrap import get_kernel_selector
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.models.cosmos3.model_runner_vae_cosmos3 import Cosmos3VAERunner
from phyai.models.cosmos3.model_runner_graph_cosmos3 import Cosmos3CudaGraphRunner


class Cosmos3OptimizedVAERunner(Cosmos3VAERunner):
    """Replay the complete native VAE with fresh inputs and independently owned outputs.

    Encoder and decoder graphs have separate bounded caches. Move their weights
    through the corresponding stage method to release graphs before storage
    changes. Small shared normalization buffers remain on the execution device.
    Posterior sampling and distributed decoding retain the native eager path.
    """

    def __init__(
        self,
        vae: Cosmos3WanVAE,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        use_cuda_graph: bool = True,
        max_graphs_per_stage: int = 2,
    ) -> None:
        super().__init__(vae, device=device, dtype=dtype)
        if max_graphs_per_stage < 1:
            raise ValueError("max_graphs_per_stage must be positive")
        self.use_cuda_graph = bool(use_cuda_graph)
        self.max_graphs_per_stage = int(max_graphs_per_stage)
        self.encode_graphs: OrderedDict[tuple, Cosmos3CudaGraphRunner] = OrderedDict()
        self.decode_graphs: OrderedDict[tuple, Cosmos3CudaGraphRunner] = OrderedDict()
        self.num_encode_captures = 0
        self.num_decode_captures = 0

    @staticmethod
    def _clear_graphs(graphs: OrderedDict[tuple, Cosmos3CudaGraphRunner]) -> None:
        for graph in graphs.values():
            graph.reset()
        graphs.clear()

    def invalidate_cache(self) -> None:
        """Release graphs after external weight, buffer, or configuration changes."""
        self._clear_graphs(self.encode_graphs)
        self._clear_graphs(self.decode_graphs)

    def _stage(
        self,
        modules: tuple[torch.nn.Module, ...],
        graphs: OrderedDict[tuple, Cosmos3CudaGraphRunner],
        device: torch.device | str,
    ) -> None:
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        tensors = [
            tensor
            for module in modules
            for tensor in (*module.parameters(), *module.buffers())
        ]
        if all(tensor.device == target for tensor in tensors):
            return
        self._clear_graphs(graphs)
        for module in modules:
            module.to(device=target)

    def stage_encoder(self, device: torch.device | str) -> None:
        """Move encoder weights, preserving decoder graphs and the execution target."""
        self._stage((self.vae.encoder, self.vae.quant_conv), self.encode_graphs, device)

    def stage_decoder(self, device: torch.device | str) -> None:
        """Move decoder weights, preserving encoder graphs and the execution target."""
        self._stage(
            (self.vae.post_quant_conv, self.vae.decoder), self.decode_graphs, device
        )

    def _stage_normalization(self, device: torch.device) -> None:
        names = ("latents_mean", "latents_std")
        if all(getattr(self.vae, name).device == device for name in names):
            return
        # CPU-to-CUDA copies must happen before capture. Both stages use these
        # buffers, so changing their storage invalidates both graph caches.
        self.invalidate_cache()
        for name in names:
            setattr(self.vae, name, getattr(self.vae, name).to(device=device))

    def _run_graph(
        self,
        graphs: OrderedDict[tuple, Cosmos3CudaGraphRunner],
        forward: Callable[..., torch.Tensor],
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        signature = Cosmos3CudaGraphRunner.tensor_signature(inputs)
        graph = graphs.get(signature)
        if graph is None:
            if len(graphs) >= self.max_graphs_per_stage:
                _, previous = graphs.popitem(last=False)
                previous.reset()
            graph = Cosmos3CudaGraphRunner()
            graphs[signature] = graph
        else:
            graphs.move_to_end(signature)
        # Six independent encodes may reuse one graph. Their outputs must survive
        # later replays, as must a returned video when decoding another latent.
        captures_before = graph.num_captures
        selector = get_kernel_selector()
        output = graph.forward(
            forward, inputs, graph_key=(id(selector), selector.epoch)
        ).clone()
        captures = graph.num_captures - captures_before
        if graphs is self.encode_graphs:
            self.num_encode_captures += captures
        else:
            self.num_decode_captures += captures
        return output

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = pixels.to(self.device, self.dtype)
        if next(self.vae.encoder.parameters()).device != x.device:
            raise RuntimeError("Stage VAE encoder weights before encoding")
        if not self.use_cuda_graph or not x.is_cuda or sample:
            return super().encode(x, sample=sample, generator=generator)
        self._stage_normalization(x.device)
        return self._run_graph(self.encode_graphs, super().encode, {"pixels": x})

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.to(self.device, self.dtype)
        if next(self.vae.decoder.parameters()).device != x.device:
            raise RuntimeError("Stage VAE decoder weights before decoding")
        if not self.use_cuda_graph or not x.is_cuda or self._decode_parallel_size() > 1:
            return super().decode(x)
        self._stage_normalization(x.device)
        return self._run_graph(self.decode_graphs, super().decode, {"latents": x})

    def close(self) -> None:
        self.invalidate_cache()


__all__ = ["Cosmos3OptimizedVAERunner"]
