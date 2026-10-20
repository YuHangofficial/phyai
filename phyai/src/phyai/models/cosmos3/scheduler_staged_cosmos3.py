"""Orchestrate independently requested Cosmos3 inference stages."""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, median
from time import perf_counter
from typing import TypeVar

import numpy as np
import torch

from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.model_runner_staged_cosmos3 import (
    Cosmos3DiTRunner,
    Cosmos3VLMRunner,
    Cosmos3StagedVAERunner,
)
from phyai.models.cosmos3.model_runner_rollout_cosmos3 import (
    Cosmos3RolloutRunner,
    Cosmos3UniPCState,
    build_native_unipc_schedule,
)
from phyai.models.cosmos3.requests_staged_cosmos3 import (
    Cosmos3DecodeRequest,
    Cosmos3DenoiseRequest,
    Cosmos3EncodeRequest,
    Cosmos3StageRequest,
    Cosmos3VLMRequest,
)
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.schedule import Scheduler
from phyai.utils import get_logger


logger = get_logger(__name__)
StageOutput = TypeVar("StageOutput")


class Cosmos3StagedScheduler(Scheduler):
    """Keep one prompt resident while staging each network independently.

    Call prefill once, encode any observations, denoise, then decode selected
    output latents. Denoising can also start directly from supplied VAE latents.
    A new prefill replaces the previous prompt and releases its DiT graph state.
    """

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        vae: Cosmos3WanVAE | None = None,
        device: torch.device | str,
        use_cuda_graph: bool = False,
        use_rollout_graph: bool = True,
        framework_compat: bool = False,
        flow_shift: float = 2.0,
        use_karras_sigmas: bool = False,
        warmup_runs: int = 0,
        timing_runs: int = 0,
    ) -> None:
        if warmup_runs < 0 or timing_runs < 0 or (timing_runs and not warmup_runs):
            raise ValueError(
                "Stage timing requires at least one warmup run and nonnegative run counts"
            )
        self.device = torch.device(device)
        self.config = transformer.config
        self.dtype = transformer.proj_in.weight.dtype
        self.flow_shift = flow_shift
        self.use_karras_sigmas = use_karras_sigmas
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
        self.use_rollout_graph = bool(use_rollout_graph)
        self.framework_compat = bool(framework_compat)
        self.timings: dict[str, dict[str, object]] = {}
        self.vlm = Cosmos3VLMRunner(transformer, device=self.device)
        self.dit = Cosmos3DiTRunner(
            transformer, device=self.device, use_cuda_graph=use_cuda_graph
        )
        self.rollout = Cosmos3RolloutRunner(self.dit, self.vlm)
        self.vae = (
            Cosmos3StagedVAERunner(
                vae,
                device=self.device,
                dtype=self.dtype,
                use_cuda_graph=use_cuda_graph,
            )
            if vae is not None
            else None
        )
        self.ready = False

    def setup(self) -> None:
        self.vlm.setup()
        self.dit.setup()
        self.rollout.setup()
        if self.vae is not None:
            self.vae.setup()
        self.ready = True

    def timed_call(
        self, operation: Callable[[], StageOutput]
    ) -> tuple[StageOutput, float]:
        """Measure synchronous wall time, including host orchestration in the stage."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        result = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return result, (perf_counter() - start) * 1000.0

    def execute_stage(
        self,
        name: str,
        load: Callable[[], object],
        run: Callable[[], StageOutput],
        offload: Callable[[], object] | None,
    ) -> StageOutput:
        """Warm and measure while weights stay resident, then optionally offload.

        Timing is opt-in. Warmups include lazy compilation and graph capture.
        Weight transfers are measured separately and happen once per stage call.
        """
        if not self.timing_runs:
            try:
                load()
                return run()
            finally:
                if offload is not None:
                    offload()
                    self.release_unused_memory()

        samples = []
        offload_ms = 0.0
        captures_before = self.stage_capture_count(name)
        try:
            _, load_ms = self.timed_call(load)
            for _ in range(self.warmup_runs):
                run()
            captures_warm = self.stage_capture_count(name)
            for _ in range(self.timing_runs):
                result, elapsed = self.timed_call(run)
                samples.append(elapsed)
            captures_timed = self.stage_capture_count(name)
        finally:
            if offload is not None:

                def release() -> None:
                    offload()
                    self.release_unused_memory()

                _, offload_ms = self.timed_call(release)
        self.timings[name] = {
            "warmup_runs": self.warmup_runs,
            "timing_runs": self.timing_runs,
            "run_ms": samples,
            "load_ms": load_ms,
            "offload_ms": offload_ms,
        }
        if name in ("dit", "vae_encode", "vae_decode"):
            self.timings[name].update(
                graph_captures_warmup=captures_warm - captures_before,
                graph_captures_timed=captures_timed - captures_warm,
            )
        logger.info_rank0(
            "Cosmos3 %s: run mean=%.3f ms median=%.3f ms min=%.3f ms "
            "(warmup=%d, measured=%d); weight load=%.3f ms offload=%.3f ms",
            name,
            mean(samples),
            median(samples),
            min(samples),
            self.warmup_runs,
            self.timing_runs,
            load_ms,
            offload_ms,
        )
        if name in ("dit", "vae_encode", "vae_decode"):
            logger.info_rank0(
                "Cosmos3 %s graph captures: warmup=%d measured=%d",
                name,
                captures_warm - captures_before,
                captures_timed - captures_warm,
            )
        return result

    @torch.inference_mode()
    def step(self, request: Cosmos3StageRequest):
        if not self.ready:
            raise RuntimeError("Call setup() before executing a Cosmos3 stage")
        if isinstance(request, Cosmos3VLMRequest):
            return self.prefill(request)
        if isinstance(request, Cosmos3EncodeRequest):
            return self.encode(request)
        if isinstance(request, Cosmos3DenoiseRequest):
            return self.denoise(request)
        if isinstance(request, Cosmos3DecodeRequest):
            return self.decode(request)
        raise TypeError(f"Unsupported Cosmos3 stage request: {type(request).__name__}")

    @torch.inference_mode()
    def prefill(self, request: Cosmos3VLMRequest) -> dict[str, int]:
        self.rollout.close()
        self.dit.close()

        def run() -> dict[str, int]:
            kv = self.vlm.forward(
                request.text_ids.to(self.device), request.text_mask.to(self.device)
            )
            return {"text_tokens": kv[0][0].shape[1], "layers": len(kv)}

        return self.execute_stage(
            "vlm",
            lambda: self.vlm.stage(self.device),
            run,
            self.vlm.offload if request.offload_after else None,
        )

    @torch.inference_mode()
    def encode(self, request: Cosmos3EncodeRequest) -> tuple[torch.Tensor, ...]:
        if not request.pixels or any(
            x.ndim != 5 or x.shape[1] != 3 for x in request.pixels
        ):
            raise ValueError("pixels must contain [B,3,T,H,W] clips")
        if self.vae is not None:
            temporal = self.vae.vae.scale_factor_temporal
            spatial = self.vae.vae.scale_factor_spatial
            for pixels in request.pixels:
                if min(pixels.shape) < 1 or (pixels.shape[2] - 1) % temporal:
                    raise ValueError(
                        f"VAE clips require a positive 1+{temporal}*k frame count"
                    )
                if pixels.shape[3] % spatial or pixels.shape[4] % spatial:
                    raise ValueError(
                        f"VAE height and width must be divisible by {spatial}"
                    )
        runner = self.vae
        if runner is None:
            raise RuntimeError("VAE stages require Cosmos3StagedArgs(load_vae=True)")
        return self.execute_stage(
            "vae_encode",
            lambda: runner.stage_encoder(self.device),
            lambda: tuple(runner.encode(x) for x in request.pixels),
            (lambda: runner.stage_encoder("cpu")) if request.offload_after else None,
        )

    @torch.inference_mode()
    def decode(self, request: Cosmos3DecodeRequest) -> torch.Tensor:
        if (
            request.latents.ndim != 5
            or request.latents.shape[1] != self.config.latent_channel
            or min(request.latents.shape) < 1
        ):
            raise ValueError("latents must have shape [B,latent_channels,T,H,W]")
        runner = self.vae
        if runner is None:
            raise RuntimeError("VAE stages require Cosmos3StagedArgs(load_vae=True)")

        def run() -> torch.Tensor:
            pixels = runner.decode(request.latents)
            return ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)

        return self.execute_stage(
            "vae_decode",
            lambda: runner.stage_decoder(self.device),
            run,
            (lambda: runner.stage_decoder("cpu")) if request.offload_after else None,
        )

    def validate_denoise(self, request: Cosmos3DenoiseRequest) -> None:
        if not self.vlm.cached_kv:
            raise RuntimeError("Run Cosmos3VLMRequest before denoising")
        if request.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not request.vision or len(request.vision) != len(request.video_noisy_masks):
            raise ValueError("Each visual stream needs its own noisy-frame mask")
        batch = self.vlm.cached_kv[0][0].shape[0]
        if self.framework_compat and batch != 1:
            raise ValueError(
                "Framework-compatible staged sampling requires batch_size=1"
            )
        tokens = 0
        patch = self.config.latent_patch_size
        for value, mask in zip(request.vision, request.video_noisy_masks, strict=True):
            if value.ndim != 5 or value.shape[:2] != (
                batch,
                self.config.latent_channel,
            ):
                raise ValueError(
                    "Each visual stream must have shape [B,latent_channels,T,H,W]"
                )
            if min(value.shape[2:]) < 1:
                raise ValueError("Visual grids must have positive dimensions")
            if mask.shape != (batch, value.shape[2]) or mask.dtype != torch.bool:
                raise ValueError("Each video noisy mask must be bool [B,T]")
            t, h, w = value.shape[2:]
            tokens += t * ((h + patch - 1) // patch) * ((w + patch - 1) // patch)
        if request.action is not None:
            action = request.action
            if not self.config.action_gen:
                raise ValueError("The checkpoint has no action adapters")
            if (
                action.ndim != 3
                or action.shape[0] != batch
                or action.shape[2] != self.config.action_dim
                or action.shape[1] < 1
            ):
                raise ValueError("Actions must have shape [B,L,action_dim]")
            if not 0 <= request.domain_id < self.config.num_embodiment_domains:
                raise ValueError(
                    "domain_id is outside the checkpoint's embodiment domains"
                )
            for mask, shape in (
                (request.action_noisy_mask, action.shape[:2]),
                (request.action_valid_mask, action.shape),
            ):
                if mask is not None and (
                    mask.shape != shape or mask.dtype != torch.bool
                ):
                    raise ValueError(
                        f"Action mask must be bool with shape {tuple(shape)}"
                    )
            tokens += action.shape[1]
        elif (
            request.action_noisy_mask is not None
            or request.action_valid_mask is not None
        ):
            raise ValueError("Action masks require action latents")
        if request.gen_position_ids.shape != (3, batch, tokens):
            raise ValueError(f"GEN position IDs must have shape {(3, batch, tokens)}")
        if request.video_role_embeddings is not None and len(
            request.video_role_embeddings
        ) != len(request.vision):
            raise ValueError("Provide one role embedding per visual stream")

    @torch.inference_mode()
    def denoise(self, request: Cosmos3DenoiseRequest) -> dict[str, object]:
        self.validate_denoise(request)
        return self.execute_stage(
            "dit",
            lambda: self.dit.stage(self.device),
            lambda: self.sample(request),
            self.offload_dit if request.offload_after else None,
        )

    def sample(self, request: Cosmos3DenoiseRequest) -> dict[str, object]:
        vision = tuple(x.to(self.device, torch.float32) for x in request.vision)
        masks = tuple(x.to(self.device) for x in request.video_noisy_masks)
        clean = list(vision)
        noise_masks = [
            m[:, None, :, None, None].expand_as(x)
            for m, x in zip(masks, vision, strict=True)
        ]
        batch = vision[0].shape[0]
        action_mask = domain = fixed_action = None
        if request.action is not None:
            action = request.action.to(self.device, torch.float32)
            valid = (
                torch.ones_like(action, dtype=torch.bool)
                if request.action_valid_mask is None
                else request.action_valid_mask.to(self.device)
            )
            action_mask = (
                torch.ones(action.shape[:2], device=self.device, dtype=torch.bool)
                if request.action_noisy_mask is None
                else request.action_noisy_mask.to(self.device)
            )
            action = action.masked_fill(~valid, 0)
            if self.framework_compat and not bool(action_mask.any().item()):
                # Forward dynamics conditions on actions without adding them to
                # the public sampler's state vector or consuming action noise.
                fixed_action = action
            else:
                clean.append(action)
                noise_masks.append(action_mask.unsqueeze(-1) & valid)
            domain = torch.full(
                (batch,), request.domain_id, device=self.device, dtype=torch.long
            )
        shapes = tuple(x.shape for x in clean)
        sizes = tuple(x[0].numel() for x in clean)
        clean_flat = torch.cat([x.reshape(batch, -1) for x in clean], dim=1)
        noisy = torch.cat([x.reshape(batch, -1) for x in noise_masks], dim=1)
        if self.framework_compat:
            # The framework restarts NumPy's CPU RNG for every visual/action
            # item, then flattens complete CTHW or action tensors in that order.
            noise = torch.cat(
                [
                    torch.from_numpy(
                        np.random.RandomState(request.seed)
                        .standard_normal(tuple(x.shape[1:]))
                        .astype(np.float32)
                    )
                    .reshape(1, -1)
                    .to(self.device)
                    for x in clean
                ],
                dim=1,
            )
            mask = noisy.to(torch.float32)
            latent = (1.0 - mask) * clean_flat + mask * noise
        else:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
            latent = torch.where(
                noisy,
                torch.randn(
                    clean_flat.shape,
                    device=self.device,
                    dtype=torch.float32,
                    generator=generator,
                ),
                clean_flat,
            )
        roles = (
            None
            if request.video_role_embeddings is None
            else tuple(
                x.to(self.device, self.dtype) for x in request.video_role_embeddings
            )
        )
        if (
            self.use_rollout_graph
            and self.dit.use_cuda_graph
            and batch == 1
            and self.device.type == "cuda"
        ):
            return self.sample_rollout(
                request,
                latent,
                clean_flat,
                noisy,
                shapes,
                masks,
                action_mask,
                domain,
                roles,
                fixed_action,
            )
        freqs = self.vlm.frequencies(request.gen_position_ids.to(self.device))
        self.dit.prepare(
            self.vlm.cached_kv,
            freqs,
            tuple(tuple(x.shape[2:]) for x in vision),
            0 if request.action is None else request.action.shape[1],
        )
        if self.framework_compat:
            schedule = build_native_unipc_schedule(
                num_steps=request.num_inference_steps,
                flow_shift=self.flow_shift,
                use_karras_sigmas=self.use_karras_sigmas,
                device=self.device,
                framework_compat=True,
            )
            sampler = Cosmos3UniPCState(schedule)
            timesteps = schedule.timesteps
        else:
            sampler = UniPCMultistepSampler(
                flow_shift=self.flow_shift, use_karras_sigmas=self.use_karras_sigmas
            )
            sampler.set_timesteps(request.num_inference_steps, device=self.device)
            timesteps = sampler.timesteps

        def unpack(flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(
                x.reshape(shape)
                for x, shape in zip(flat.split(sizes, dim=1), shapes, strict=True)
            )

        for timestep in timesteps:
            streams = unpack(latent)
            velocities = self.dit.forward(
                streams[: len(vision)],
                timestep.to(
                    self.device, torch.float32 if self.framework_compat else self.dtype
                )
                .reshape(1)
                .expand(batch),
                video_noisy_masks=masks,
                action_latents=(
                    fixed_action
                    if fixed_action is not None
                    else streams[-1]
                    if request.action is not None
                    else None
                ),
                action_domain_id=domain,
                action_noisy_mask=action_mask,
                video_role_embeddings=roles,
            )
            velocity = torch.cat(
                [
                    (x if self.framework_compat else x.float()).reshape(batch, -1)
                    for x in velocities[: len(shapes)]
                ],
                dim=1,
            )
            if self.framework_compat:
                velocity = velocity * noisy.to(velocity.dtype)
                latent = sampler.step(velocity, latent)
            else:
                velocity = velocity.masked_fill(~noisy, 0)
                latent = torch.where(
                    noisy, sampler.step(velocity, timestep, latent), clean_flat
                )
        streams = unpack(latent)
        result: dict[str, object] = {"vision": streams[: len(vision)]}
        if request.action is not None:
            result["action"] = fixed_action if fixed_action is not None else streams[-1]
        return result

    def sample_rollout(
        self,
        request: Cosmos3DenoiseRequest,
        latent: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        shapes: tuple[tuple[int, ...], ...],
        masks: tuple[torch.Tensor, ...],
        action_mask: torch.Tensor | None,
        domain: torch.Tensor | None,
        roles: tuple[torch.Tensor, ...] | None,
        fixed_action: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Capture the complete native loop, refreshing all request data on replay."""
        num_vision = len(masks)
        has_action = request.action is not None
        plan = self.rollout.prepare(
            request.gen_position_ids,
            shapes,
            num_vision=num_vision,
            num_steps=request.num_inference_steps,
            flow_shift=self.flow_shift,
            use_karras_sigmas=self.use_karras_sigmas,
            framework_compat=self.framework_compat,
            action_len=0 if request.action is None else request.action.shape[1],
        )
        inputs = {"initial": latent, "clean": clean, "noisy": noisy}
        inputs.update({f"mask_{i}": mask for i, mask in enumerate(masks)})
        if action_mask is not None and domain is not None:
            inputs.update(action_mask=action_mask, domain=domain)
        if fixed_action is not None:
            inputs["fixed_action"] = fixed_action
        if roles is not None:
            inputs.update({f"role_{i}": role for i, role in enumerate(roles)})

        def unpack(flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(
                x.reshape(shape)
                for x, shape in zip(
                    flat.split(plan.sizes, dim=1), plan.shapes, strict=True
                )
            )

        def complete_rollout(**buffers: torch.Tensor) -> torch.Tensor:
            state = Cosmos3UniPCState(plan.schedule)
            current = buffers["initial"]
            frame_masks = tuple(buffers[f"mask_{i}"] for i in range(num_vision))
            role_embeddings = (
                None
                if roles is None
                else tuple(buffers[f"role_{i}"] for i in range(num_vision))
            )
            for index, timestep in enumerate(plan.timesteps):
                streams = unpack(current)
                velocities = self.dit.forward_prepared(
                    streams[:num_vision],
                    timestep,
                    plan.condition,
                    video_noisy_masks=frame_masks,
                    action_latents=(
                        buffers["fixed_action"]
                        if fixed_action is not None
                        else streams[-1]
                        if has_action
                        else None
                    ),
                    action_domain_id=buffers.get("domain"),
                    action_noisy_mask=buffers.get("action_mask"),
                    video_role_embeddings=role_embeddings,
                    time_embedding=plan.time_embeddings[index],
                )
                velocity = torch.cat(
                    [
                        (x if self.framework_compat else x.float()).reshape(
                            shapes[0][0], -1
                        )
                        for x in velocities[: len(shapes)]
                    ],
                    dim=1,
                )
                velocity = (
                    velocity * buffers["noisy"].to(velocity.dtype)
                    if self.framework_compat
                    else velocity.masked_fill(~buffers["noisy"], 0)
                )
                updated = state.step(velocity, current)
                current = (
                    updated
                    if self.framework_compat
                    else torch.where(buffers["noisy"], updated, buffers["clean"])
                )
            return current

        streams = unpack(self.rollout.forward(complete_rollout, inputs, plan=plan))
        result: dict[str, object] = {"vision": streams[:num_vision]}
        if has_action:
            result["action"] = fixed_action if fixed_action is not None else streams[-1]
        return result

    def dit_capture_count(self) -> int:
        return self.dit.graph.num_captures + self.rollout.graph.num_captures

    def stage_capture_count(self, name: str) -> int:
        if name == "dit":
            return self.dit_capture_count()
        if self.vae is not None:
            if name == "vae_encode":
                return self.vae.num_encode_captures
            if name == "vae_decode":
                return self.vae.num_decode_captures
        return 0

    def offload_dit(self) -> None:
        self.rollout.close()
        self.dit.offload()

    def release_unused_memory(self) -> None:
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def close(self) -> None:
        self.rollout.close()
        self.dit.close()
        self.vlm.close()
        if self.vae is not None:
            self.vae.close()
        self.ready = False


__all__ = ["Cosmos3StagedScheduler"]
