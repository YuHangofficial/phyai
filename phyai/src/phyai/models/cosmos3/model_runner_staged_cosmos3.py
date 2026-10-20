"""Separate Cosmos3 UND prefill and packed DiT execution lifetimes."""

from __future__ import annotations

from functools import partial

import torch

from phyai.runtime.model_runner import ModelRunner
from phyai.layers.attention.nocache.base import AttnCtx
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Condition,
    Cosmos3Transformer,
    Cosmos3AttentionPreparation,
    compute_mrope_position_ids_text,
)
from phyai.models.cosmos3.modeling_cosmos3_staged import Cosmos3PackedDiT
from phyai.models.cosmos3.model_runner_graph_cosmos3 import Cosmos3CudaGraphRunner
from phyai.models.cosmos3.model_runner_fusion_cosmos3 import Cosmos3AttentionFusion
from phyai.models.cosmos3.model_runner_vae_optimized_cosmos3 import (
    Cosmos3OptimizedVAERunner as Cosmos3StagedVAERunner,
)
from phyai.models.cosmos3.model_runner_attention_cosmos3 import (
    Cosmos3AttentionContexts,
)


class Cosmos3VLMRunner(ModelRunner):
    """Run native text prefill and retain its KV independently of UND weights.

    ``device`` is the execution target. Moving weights with :meth:`stage` or
    :meth:`offload` does not change it or relocate the retained KV tensors.
    """

    def __init__(
        self, transformer: Cosmos3Transformer, *, device: torch.device | str
    ) -> None:
        self.transformer = transformer
        self.device = torch.device(device)
        self.cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.text_mask: torch.Tensor | None = None

    def setup(self) -> None:
        """UND prefill needs no persistent attention plans or graph capture."""

    def stage(self, device: torch.device | str) -> None:
        """Move UND weights while retaining the previous prefill result."""
        self.transformer.language_model.to(device=device)

    def offload(self) -> None:
        """Release UND GPU weights while keeping its KV on the execution device."""
        self.stage("cpu")

    @torch.no_grad()
    def frequencies(
        self, gen_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build native GEN RoPE for ``[3,B,S]`` positions, including after offload."""
        if gen_position_ids.ndim != 3 or gen_position_ids.shape[0] != 3:
            raise ValueError("GEN position IDs must have shape [3,B,S]")
        language_model = self.transformer.language_model
        # The native rotary helper transfers its small inv_freq buffer as needed;
        # generating positions never reloads the UND weight matrices.
        cos, sin = language_model.rotary_emb.get_cos_sin(
            gen_position_ids.to(self.device)
        )
        dtype = language_model.embed_tokens.weight.dtype
        return cos.unsqueeze(2).to(dtype), sin.unsqueeze(2).to(dtype)

    @torch.no_grad()
    def forward(
        self, text_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Replace the prefill state using an equally right-padded text batch."""
        if text_ids.ndim != 2 or text_mask.shape != text_ids.shape:
            raise ValueError("Text IDs and mask must have matching [B,S] shapes")
        if min(text_ids.shape) < 1:
            raise ValueError("Text prefill requires a nonempty batch and prompt")
        mask = text_mask.detach().cpu()
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("The text mask must contain only zero and one")
        lengths = mask.sum(dim=1).long()
        real_length = int(lengths[0])
        if real_length < 1 or not torch.all(lengths == real_length):
            raise ValueError("Cosmos3 requires identical nonzero real text lengths")
        expected = torch.arange(mask.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(mask.bool(), expected):
            raise ValueError("Cosmos3 text masks must use right padding")

        ids = text_ids.to(self.device)
        language_model = self.transformer.language_model
        if language_model.embed_tokens.weight.device != ids.device:
            raise RuntimeError(
                "Stage UND weights on the execution device before prefill"
            )
        positions, _ = compute_mrope_position_ids_text(real_length, temporal_offset=0)
        if real_length < ids.shape[1]:
            positions = torch.cat(
                (
                    positions,
                    torch.zeros(3, ids.shape[1] - real_length, dtype=torch.long),
                ),
                dim=1,
            )
        positions = positions.unsqueeze(1).expand(-1, ids.shape[0], -1)
        full_kv = language_model(ids, self.frequencies(positions))
        self.cached_kv = [
            (key[:, :real_length], value[:, :real_length]) for key, value in full_kv
        ]
        self.text_mask = mask.clone()
        return self.cached_kv

    def close(self) -> None:
        self.cached_kv = []
        self.text_mask = None


class Cosmos3DiTRunner(ModelRunner):
    """Own prepared conditioning, attention metadata, and one packed DiT graph."""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        device: torch.device | str,
        use_cuda_graph: bool = False,
    ) -> None:
        self.transformer = transformer
        self.model = Cosmos3PackedDiT(transformer)
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.condition: Cosmos3Condition | None = None
        self.video_shapes: tuple[tuple[int, int, int], ...] = ()
        self.graph = Cosmos3CudaGraphRunner()
        self._contexts = Cosmos3AttentionContexts()
        self._fusion = Cosmos3AttentionFusion()
        self._condition_revision = 0
        self._attention_kernel_revision: tuple | None = None

    def setup(self) -> None:
        """Attention planning and graph capture are deferred to the first forward."""

    def stage(self, device: torch.device | str) -> None:
        """Move GEN weights, releasing graphs before any weight addresses change.

        UND weights and cached conditioning remain in place. The execution target
        configured at construction is unchanged by temporary CPU offload.
        """
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        children = [
            module
            for name, module in self.transformer.named_children()
            if name != "language_model"
        ]
        tensors = list(self.transformer.parameters(recurse=False)) + list(
            self.transformer.buffers(recurse=False)
        )
        for module in children:
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        if all(tensor.device == target for tensor in tensors):
            return
        self.graph.reset()
        self._contexts.reset()
        for module in children:
            module.to(device=target)
        # Modality embeddings are root parameters, outside the GEN child modules.
        for parameter in self.transformer.parameters(recurse=False):
            parameter.data = parameter.data.to(target)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(target)
        for name, buffer in self.transformer.named_buffers(recurse=False):
            self.transformer._buffers[name] = buffer.to(target)

    def offload(self) -> None:
        self.stage("cpu")

    def refresh_attention_contexts(self) -> None:
        from phyai.kernel.bootstrap import get_kernel_selector

        selector = get_kernel_selector()
        revision = (selector, selector.epoch)
        if revision != self._attention_kernel_revision:
            self._contexts.reset()
            self._attention_kernel_revision = revision

    def rollout_condition(self) -> Cosmos3Condition:
        """Prepare immutable attention metadata before an outer rollout capture."""
        from dataclasses import replace

        condition = self.condition
        if condition is None:
            raise RuntimeError("Prepare DiT conditioning before a rollout")
        key = condition.cached_kv[0][0]
        if key.shape[0] != 1 or not key.is_cuda:
            raise ValueError("Whole-rollout graphs require a single CUDA batch item")
        self.refresh_attention_contexts()
        contexts = self._contexts.prepare(
            self.transformer.gen_layers,
            batch_size=1,
            query_tokens=condition.freqs_gen[0].shape[1],
            text_tokens=key.shape[1],
            dtype=key.dtype,
            device=key.device,
            use_cuda_graph=True,
        )
        return replace(
            condition,
            attention_contexts=contexts,
            attention_preparation=self._fusion.prepare(
                device=key.device, use_cuda_graph=True
            ),
        )

    def forward_prepared(
        self,
        vision: tuple[torch.Tensor, ...],
        timestep: torch.Tensor,
        condition: Cosmos3Condition,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Run one complete GEN pass inside an already prepared outer graph."""
        return self.model(vision, timestep, condition, **kwargs)

    @torch.no_grad()
    def prepare(
        self,
        cached_kv: list[tuple[torch.Tensor, torch.Tensor]],
        gen_freqs: tuple[torch.Tensor, torch.Tensor],
        video_shapes: tuple[tuple[int, int, int], ...],
        action_len: int,
    ) -> Cosmos3Condition:
        """Replace immutable request conditioning without rerunning UND prefill."""
        shapes = tuple(tuple(shape) for shape in video_shapes)
        if not shapes or any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            raise ValueError("Provide nonempty (T,H,W) grids for all visual streams")
        if action_len < 0 or (action_len and not self.transformer.action_gen):
            raise ValueError("Action tokens require an action-enabled transformer")
        if len(cached_kv) != len(self.transformer.gen_layers) or not cached_kv:
            raise ValueError("Provide one UND KV pair per GEN layer")
        cos, sin = gen_freqs
        tokens = action_len
        for t, h, w in shapes:
            hp, wp, _, _ = self.transformer._pad_to_patch_size(h, w)
            tokens += t * hp * wp
        if (
            cos.ndim != 4
            or cos.shape != sin.shape
            or cos.shape[1:] != (tokens, 1, self.transformer.config.head_dim)
        ):
            raise ValueError("GEN RoPE must match the packed visual and action tokens")
        batch = cos.shape[0]
        text_tokens = cached_kv[0][0].shape[1]
        dtype = self.transformer.proj_in.weight.dtype
        for layer, (key, value) in zip(
            self.transformer.gen_layers, cached_kv, strict=True
        ):
            attention = layer.cross_attention
            expected = (
                batch,
                text_tokens,
                attention.num_local_kv_heads,
                attention.head_dim,
            )
            if key.shape != expected or value.shape != expected or text_tokens < 1:
                raise ValueError(
                    "UND KV shape must match the batch and local GEN heads"
                )
            if key.dtype != dtype or value.dtype != dtype:
                raise ValueError("UND KV dtype must match the GEN weights")
        self.condition = Cosmos3Condition(
            cached_kv=[
                (key.to(self.device), value.to(self.device)) for key, value in cached_kv
            ],
            freqs_gen=(cos.to(self.device, dtype), sin.to(self.device, dtype)),
            video_shape=shapes[0],
            action_len=action_len,
        )
        self.video_shapes = shapes
        self._condition_revision += 1
        return self.condition

    def _graph_forward(
        self,
        *,
        video_shapes: tuple[tuple[int, int, int], ...],
        action_len: int,
        num_layers: int,
        attention_contexts: tuple[AttnCtx, ...] | None,
        attention_preparation: Cosmos3AttentionPreparation | None,
        **inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        condition = Cosmos3Condition(
            cached_kv=[
                (inputs[f"key_{index}"], inputs[f"value_{index}"])
                for index in range(num_layers)
            ],
            freqs_gen=(inputs["freqs_cos"], inputs["freqs_sin"]),
            video_shape=video_shapes[0],
            action_len=action_len,
            attention_contexts=attention_contexts,
            attention_preparation=attention_preparation,
        )
        count = len(video_shapes)
        return self.model(
            tuple(inputs[f"vision_{index}"] for index in range(count)),
            inputs["timestep"],
            condition,
            video_noisy_masks=tuple(inputs[f"mask_{index}"] for index in range(count)),
            action_latents=inputs.get("action_latents"),
            action_domain_id=inputs.get("action_domain_id"),
            action_noisy_mask=inputs.get("action_noisy_mask"),
            video_role_embeddings=(
                tuple(inputs[f"role_{index}"] for index in range(count))
                if "role_0" in inputs
                else None
            ),
        )

    @torch.no_grad()
    def forward(
        self,
        vision: tuple[torch.Tensor, ...],
        timestep: torch.Tensor,
        video_noisy_masks: tuple[torch.Tensor, ...],
        action_latents: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        video_role_embeddings: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return visual velocities in input order, then the optional action velocity."""
        condition = self.condition
        if condition is None:
            raise RuntimeError("Prepare DiT conditioning before running generation")
        if tuple(tuple(tensor.shape[2:]) for tensor in vision) != self.video_shapes:
            raise ValueError("Visual streams must match the prepared grids")
        if len(video_noisy_masks) != len(vision):
            raise ValueError("Provide one noisy mask per visual stream")
        if video_role_embeddings is not None and len(video_role_embeddings) != len(
            vision
        ):
            raise ValueError("Provide one role embedding per visual stream")
        inputs = {"timestep": timestep.to(self.device)}
        for index, (latent, mask) in enumerate(
            zip(vision, video_noisy_masks, strict=True)
        ):
            inputs[f"vision_{index}"] = latent.to(self.device)
            inputs[f"mask_{index}"] = mask.to(self.device)
        for name, value in (
            ("action_latents", action_latents),
            ("action_domain_id", action_domain_id),
            ("action_noisy_mask", action_noisy_mask),
        ):
            if value is not None:
                inputs[name] = value.to(self.device)
        if video_role_embeddings is not None:
            for index, role in enumerate(video_role_embeddings):
                inputs[f"role_{index}"] = role.to(self.device)
        latent = inputs["vision_0"]
        if self.transformer.proj_in.weight.device != latent.device:
            raise RuntimeError(
                "Stage GEN weights on the execution device before generation"
            )
        # Batched FlashInfer plans are mutable; preserve the existing eager path.
        use_graph = self.use_cuda_graph and latent.is_cuda and latent.shape[0] == 1
        preparation = self._fusion.prepare(
            device=latent.device, use_cuda_graph=use_graph
        )
        contexts = None
        attention_key = None
        if latent.shape[0] == 1:
            self.refresh_attention_contexts()
            contexts = self._contexts.prepare(
                self.transformer.gen_layers,
                batch_size=1,
                query_tokens=condition.freqs_gen[0].shape[1],
                text_tokens=condition.cached_kv[0][0].shape[1],
                dtype=condition.cached_kv[0][0].dtype,
                device=latent.device,
                use_cuda_graph=use_graph,
            )
            attention_key = (self._contexts.signature, self._contexts.revision)
        constants = {
            "freqs_cos": condition.freqs_gen[0],
            "freqs_sin": condition.freqs_gen[1],
        }
        for index, (key, value) in enumerate(condition.cached_kv):
            constants[f"key_{index}"] = key
            constants[f"value_{index}"] = value
        forward = partial(
            self._graph_forward,
            video_shapes=self.video_shapes,
            action_len=condition.action_len,
            num_layers=len(condition.cached_kv),
            attention_contexts=contexts,
            attention_preparation=preparation,
        )
        if not use_graph:
            return forward(**inputs, **constants)
        output = self.graph.forward(
            forward,
            inputs,
            static_inputs=constants,
            static_key=self._condition_revision,
            graph_key=(
                self.video_shapes,
                condition.action_len,
                attention_key,
                preparation,
            ),
        )
        # UniPC retains earlier velocities, which subsequent replays must not overwrite.
        return tuple(tensor.clone() for tensor in output)

    def close(self) -> None:
        self.graph.reset()
        self._contexts.reset()
        self._fusion.reset()
        self._attention_kernel_revision = None
        self.condition = None
        self.video_shapes = ()


__all__ = ["Cosmos3VLMRunner", "Cosmos3DiTRunner", "Cosmos3StagedVAERunner"]
