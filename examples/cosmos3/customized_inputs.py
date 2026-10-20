"""Synthetic custom inputs for the staged Cosmos3 example.

The workload uses six goal/history/rollout visual streams and one state/action
stream. Values are synthetic, and optional role embeddings are left unset.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3Config
from phyai.models.cosmos3.modeling_cosmos3 import (
    compute_mrope_position_ids_text,
    compute_mrope_position_ids_vision,
)
from phyai.models.cosmos3.requests_staged_cosmos3 import Cosmos3DenoiseRequest


PROMPT = (
    "Tidy up the room: put the clothes into the laundry basket, "
    "then turn and drive away."
)
VISION_SHAPES = (
    (1, 48, 1, 18, 20),
    *((1, 48, 1, 10, 10),) * 3,
    (1, 48, 1, 6, 6),
    (1, 48, 4, 18, 20),
)
HISTORY_AGES_SECONDS = (8.0 / 30.0, 0.5, 1.0, 2.0)
HISTORY_POOL_FACTORS = (2, 2, 2, 4)


def tokenize_prompt(checkpoint: str | Path, prompt: str = PROMPT) -> torch.Tensor:
    """Return [1,30] IDs for the default prompt, including generation boundaries."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(checkpoint) / "text_tokenizer", local_files_only=True
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=False,
    )
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id, *ids]
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if tokenizer.eos_token_id is None or start is None:
        raise ValueError("The tokenizer lacks generation boundary tokens")
    return torch.tensor([[*ids, tokenizer.eos_token_id, start]], dtype=torch.long)


def build_gen_positions(text_tokens: int, config: Cosmos3Config) -> torch.Tensor:
    """Return [3,1,583] positions, retaining the unpooled patch coordinates."""
    _, text_end = compute_mrope_position_ids_text(text_tokens, temporal_offset=0)
    goal_origin = text_end + config.temporal_modality_margin
    common = {
        "base_fps": config.base_fps,
        "temporal_compression_factor": config.temporal_compression_factor,
        "enable_fps_modulation": config.enable_fps_modulation,
    }
    goal, rollout_origin = compute_mrope_position_ids_vision(
        1, 9, 10, temporal_offset=goal_origin, fps=7.5, **common
    )
    rollout, _ = compute_mrope_position_ids_vision(
        4, 9, 10, temporal_offset=rollout_origin, fps=7.5, **common
    )
    histories = []
    for age, factor in zip(HISTORY_AGES_SECONDS, HISTORY_POOL_FACTORS, strict=True):
        h0, w0 = (torch.arange(0, n, factor, dtype=torch.float32) for n in (9, 10))
        h = (h0 + (h0 + factor).clamp(max=9) - 1) / 2
        w = (w0 + (w0 + factor).clamp(max=10) - 1) / 2
        time = torch.full(
            (h.numel() * w.numel(),),
            rollout_origin - age * config.base_fps / config.temporal_compression_factor,
            dtype=torch.float32,
        )
        histories.append(
            torch.stack((time, h.repeat_interleave(w.numel()), w.repeat(h.numel())))
        )
    action, _ = compute_mrope_position_ids_vision(
        49,
        1,
        1,
        temporal_offset=goal_origin,
        fps=30.0,
        **(common | {"temporal_compression_factor": 1}),
        base_temporal_compression_factor=config.temporal_compression_factor,
        start_frame_offset=1,
    )
    action[0] += rollout_origin - goal_origin
    return torch.cat((goal, *histories, rollout, action), dim=1).unsqueeze(1)


def make_denoise_request(
    text_tokens: int,
    config: Cosmos3Config,
    *,
    seed: int = 9000,
    vision: tuple[torch.Tensor, ...] | None = None,
    steps: int = 10,
    offload_after: bool = False,
) -> Cosmos3DenoiseRequest:
    """Build six visual streams plus a clean state row and 48 noisy action rows."""
    if (config.latent_channel, config.latent_patch_size, config.action_dim) != (
        48,
        2,
        64,
    ):
        raise ValueError(
            "This input example requires 48 latent channels, 2x2 patches and 64 action channels"
        )
    rng = torch.Generator(device="cpu").manual_seed(seed)
    action = torch.zeros(1, 49, 64, dtype=torch.float32)
    action[0, 0, :49] = torch.randn(49, generator=rng, dtype=torch.float32)
    valid = torch.zeros_like(action, dtype=torch.bool)
    valid[:, :, :42] = True
    valid[:, :, 22] = False
    action.masked_fill_(~valid, 0)
    if vision is None:
        vision = tuple(
            torch.randn(shape, generator=rng, dtype=torch.float32)
            for shape in VISION_SHAPES
        )
    if tuple(tuple(x.shape) for x in vision) != VISION_SHAPES:
        raise ValueError(f"Expected goal/history/rollout latent shapes {VISION_SHAPES}")
    video_masks = tuple(
        torch.zeros(1, shape[2], dtype=torch.bool) for shape in VISION_SHAPES
    )
    video_masks[-1][:, 1:] = True
    action_mask = torch.ones(1, 49, dtype=torch.bool)
    action_mask[:, 0] = False
    return Cosmos3DenoiseRequest(
        vision=vision,
        video_noisy_masks=video_masks,
        gen_position_ids=build_gen_positions(text_tokens, config),
        action=action,
        action_noisy_mask=action_mask,
        action_valid_mask=valid,
        domain_id=22,
        num_inference_steps=steps,
        seed=seed,
        offload_after=offload_after,
    )


def make_fake_pixels(seed: int = 9000) -> tuple[torch.Tensor, ...]:
    """Make normalized VAE clips: five stills, then current plus 12 black frames."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    stills = tuple(
        torch.rand(1, 3, 1, 288, 320, generator=rng, dtype=torch.float32) * 2 - 1
        for _ in range(5)
    )
    rollout = torch.full((1, 3, 13, 288, 320), -1.0, dtype=torch.float32)
    rollout[:, :, :1] = (
        torch.rand(1, 3, 1, 288, 320, generator=rng, dtype=torch.float32) * 2 - 1
    )
    return (*stills, rollout)


def pool_encoded_history(latents: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Pool each history's 2x2 patch phases separately, preserving border averages."""
    expected = ((1, 48, 1, 18, 20),) * 5 + ((1, 48, 4, 18, 20),)
    if tuple(tuple(x.shape) for x in latents) != expected:
        raise ValueError(f"Expected unpooled VAE latent shapes {expected}")
    output = list(latents)
    for index, factor in enumerate(HISTORY_POOL_FACTORS, start=1):
        phases = latents[index][0, :, 0].reshape(48, 9, 2, 10, 2)
        phases = phases.permute(0, 2, 4, 1, 3).reshape(1, 48 * 4, 9, 10)
        pooled = F.avg_pool2d(
            phases, factor, stride=factor, ceil_mode=True, count_include_pad=False
        )
        h, w = pooled.shape[-2:]
        output[index] = (
            pooled.reshape(48, 2, 2, h, w)
            .permute(0, 3, 1, 4, 2)
            .reshape(1, 48, 1, h * 2, w * 2)
            .contiguous()
        )
    return tuple(output)
