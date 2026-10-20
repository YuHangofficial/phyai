"""Stateless Cosmos3 GEN forward over multiple visual latent grids."""

from __future__ import annotations

import torch
from torch import nn

from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Condition,
    Cosmos3Transformer,
)


class Cosmos3PackedDiT(nn.Module):
    """Pack visual streams and optional actions through an existing GEN tower.

    The wrapped transformer supplies all weights. UND prefill, position building,
    condition storage, graph capture, and denoising belong to the caller. GEN
    positions must follow the supplied visual stream order, with actions last.
    """

    def __init__(self, transformer: Cosmos3Transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        vision: tuple[torch.Tensor, ...],
        timestep: torch.Tensor,
        condition: Cosmos3Condition,
        *,
        video_noisy_masks: tuple[torch.Tensor, ...],
        action_latents: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        video_role_embeddings: tuple[torch.Tensor, ...] | None = None,
        time_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return each visual velocity followed by the optional action velocity.

        Visual inputs have shape ``[B,C,T,H,W]`` and may use different grids.
        Their masks are ``[B,T]`` (one for noisy frames). Actions use
        ``[B,A,action_dim]`` with an optional ``[B,A]`` noisy mask. ``timestep``
        is ``[B]`` and shares the existing transformer's units and embedding.
        ``time_embedding`` can supply its precomputed ``[B,hidden_size]`` result.

        Optional visual role embeddings are added after the input projection,
        before timestep conditioning. Each embedding is ``[hidden]``,
        ``[B,hidden]``, or ``[B,1|tokens,hidden]``. A batch size of one broadcasts.
        The returned velocities include clean positions; the scheduler applies
        condition and action-validity masks when updating the sample.
        """
        model = self.transformer
        if not vision or len(video_noisy_masks) != len(vision):
            raise ValueError(
                "Provide at least one visual stream and one mask per stream"
            )
        if video_role_embeddings is not None and len(video_role_embeddings) != len(
            vision
        ):
            raise ValueError("Provide one role embedding per visual stream")
        if condition.sound_len:
            raise ValueError("Packed visual/action generation does not support sound")
        if vision[0].ndim != 5:
            raise ValueError("Visual streams must have shape [B,C,T,H,W]")
        batch = vision[0].shape[0]
        if timestep.shape != (batch,):
            raise ValueError(
                f"timestep must have shape [{batch}], got {tuple(timestep.shape)}"
            )

        hidden_streams = []
        video_shapes = []
        video_tokens = []
        time_embed = (
            model.time_embedder(timestep * model.timestep_scale)
            if time_embedding is None
            else time_embedding
        )
        if time_embed.shape != (batch, model.config.hidden_size):
            raise ValueError("The timestep embedding must have shape [B,hidden_size]")
        for index, (latent, noisy_mask) in enumerate(
            zip(vision, video_noisy_masks, strict=True)
        ):
            if latent.ndim != 5 or latent.shape[:2] != (
                batch,
                model.latent_channel_size,
            ):
                raise ValueError(
                    "Visual streams must share batch and latent channel dimensions"
                )
            t, h, w = latent.shape[2:]
            if min(t, h, w) < 1 or noisy_mask.shape != (batch, t):
                raise ValueError(
                    "Each visual grid must be nonempty with a [B,T] noisy mask"
                )
            hp, wp, _, _ = model._pad_to_patch_size(h, w)
            patches = model.patchify(latent, t, h, w)
            hidden, _ = model.proj_in(patches.to(model.proj_in.weight.dtype))
            if video_role_embeddings is not None:
                role = video_role_embeddings[index]
                if role.ndim == 2:
                    role = role.unsqueeze(1)
                if role.ndim not in (1, 3) or role.shape[-1] != hidden.shape[-1]:
                    raise ValueError(
                        "Role embeddings must have hidden width and rank 1, 2, or 3"
                    )
                if role.ndim == 3 and (
                    role.shape[0] not in (1, batch)
                    or role.shape[1] not in (1, hidden.shape[1])
                ):
                    raise ValueError(
                        "Role embedding batch/token dimensions cannot broadcast"
                    )
                hidden = hidden + role.to(hidden.dtype)
            hidden = hidden + model._gate_timestep(
                time_embed.to(hidden.dtype), noisy_mask, hp * wp
            )
            hidden_streams.append(hidden)
            video_shapes.append((t, h, w))
            video_tokens.append(hidden.shape[1])

        hidden_gen = (
            hidden_streams[0]
            if len(hidden_streams) == 1
            else torch.cat(hidden_streams, dim=1)
        )
        num_video_tokens = hidden_gen.shape[1]
        if action_latents is not None:
            if not model.action_gen:
                raise ValueError("The transformer does not contain action projections")
            if (
                action_latents.ndim != 3
                or action_latents.shape[0] != batch
                or action_latents.shape[1] != condition.action_len
                or action_latents.shape[2] != model.action_dim
            ):
                raise ValueError(
                    "Actions must have shape [B,condition.action_len,action_dim]"
                )
            if action_domain_id is None or action_domain_id.shape != (batch,):
                raise ValueError("Actions require one domain ID per batch item")
            if (
                action_noisy_mask is not None
                and action_noisy_mask.shape != action_latents.shape[:2]
            ):
                raise ValueError("The action noisy mask must have shape [B,A]")
            action = model.action_proj_in(
                action_latents.to(model.action_proj_in.fc.dtype), action_domain_id
            )
            action = action + model.action_modality_embed.to(action.dtype)
            action = action + model._gate_timestep(
                time_embed.to(action.dtype), action_noisy_mask, 1
            )
            hidden_gen = torch.cat((hidden_gen, action), dim=1)
        elif condition.action_len:
            raise ValueError(
                "The condition expects action tokens but no action was supplied"
            )

        cos, sin = condition.freqs_gen
        expected_freqs = (batch, hidden_gen.shape[1], 1, model.config.head_dim)
        if cos.shape != expected_freqs or sin.shape != expected_freqs:
            raise ValueError(f"GEN RoPE tensors must have shape {expected_freqs}")
        if condition.attention_contexts is not None and len(
            condition.attention_contexts
        ) != len(model.gen_layers):
            raise ValueError("Provide one attention context per GEN layer")
        for index, (layer, (key, value)) in enumerate(
            zip(model.gen_layers, condition.cached_kv, strict=True)
        ):
            context = (
                condition.attention_contexts[index]
                if condition.attention_contexts is not None
                else None
            )
            hidden_gen = layer(
                hidden_gen,
                key,
                value,
                cos,
                sin,
                context,
                condition.attention_preparation,
            )
        hidden_gen = model.norm_moe_gen(hidden_gen)
        video_output, _ = model.proj_out(hidden_gen[:, :num_video_tokens])
        velocities = tuple(
            model.unpatchify(tokens, *shape)
            for tokens, shape in zip(
                video_output.split(video_tokens, dim=1), video_shapes, strict=True
            )
        )
        if action_latents is not None:
            action_output = model.action_proj_out(
                hidden_gen[:, num_video_tokens:], action_domain_id
            )
            velocities = (*velocities, action_output)
        return velocities


__all__ = ["Cosmos3PackedDiT"]
