"""End-to-end Cosmos3 generation on one or more GPUs.

Drives the ``cosmos3`` engine plugin on a Cosmos3-Nano checkpoint: tokenizes a
prompt, runs the diffusion model, and decodes the result to a single mp4 (with an
AAC audio track muxed in when ``--sound`` requests a joint audio stream).

Example (T2V — defaults are 720x1280, 189 frames, 35 steps, matching the reference)::

    uv run python examples/cosmos3/run_cosmos3.py \\
        --checkpoint /path/to/Cosmos3-Nano \\
        --prompt "A red sports car driving along a coastal road at sunset." \\
        --out .cache/cosmos3_t2v

    # the default is a full-size generation (slow); for a quick smoke test pass e.g.
    #   --num-frames 49 --height 480 --width 832 --steps 10

Text-to-audio-video (T2AV) — add a jointly denoised sound stream::

    uv run python examples/cosmos3/run_cosmos3.py --checkpoint /path/to/Cosmos3-Nano \\
        --prompt "ocean waves crashing on rocks" --sound --out .cache/cosmos3_t2av

Defaults follow the cosmos-framework native generation config:
  * ``flow_shift=10`` + linear-flow UniPC (``--use-karras-sigmas false``); steps=35 /
    guidance=6.0 / fps match the reference sample args.
  * the positive prompt is auto-appended with duration/resolution metadata, and the
    negative prompt defaults to the reference structured "bad-quality" negative
    (pass ``--negative-prompt ""`` for an empty negative, or your own string).

Native-parity note (read before benchmarking against the reference):
  The sampler schedule, the per-modality initial noise (numpy ``RandomState``), and the
  prompt STRINGS are aligned to cosmos-framework native and verified bit-/byte-exact.
  End-to-end, phyai reproduces the *diffusers*-cosmos3 reference to >0.99 cosine. Against
  the *cosmos-framework native* repo the final-latent cosine is ~0.95 (NOT a config/OOD
  effect — it persists in-distribution). The residual is the text conditioning: phyai
  tokenizes with the checkpoint ``text_tokenizer`` and appends a ``<|vision_start|>``
  text/media boundary token, whereas native uses the Qwen3-VL tokenizer (+ extra special
  tokens) and marks the boundary via packed-sequence structure. The large structured
  negative prompt (high CFG leverage) is byte-identical as a string but tokenizes to a
  different length, which dominates the gap. That difference lives in the tokenizer /
  modeling-convention layer, not in the sampler/noise or this example. (Image/video
  conditioning — I2V / I2AV — sets ``cond_latents`` + ``cond_frame_indexes`` on the
  request from VAE-encoded clean frames; see ``Cosmos3T2VScheduler.encode``.)

Requires CUDA and a Cosmos3-Nano checkpoint.

Outputs (prefix set by ``--out``, default ``.cache/cosmos3_t2v``):
  * ``<out>.mp4``   decoded video — with an AAC audio track muxed in when ``--sound``
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

import torch


@contextlib.contextmanager
def _timed(label: str, store: dict[str, float], *, synchronize_cuda: bool = False):
    """Time a region in seconds (CUDA-synchronized) into ``store[label]``."""
    if synchronize_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if synchronize_cuda:
            torch.cuda.synchronize()
        store[label] = time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Cosmos3-Nano checkpoint dir"
    )
    parser.add_argument(
        "--prompt", default="A red sports car driving along a coastal road at sunset."
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Omit to use the native structured default; pass '' for none.",
    )
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument(
        "--use-karras-sigmas",
        choices=("auto", "true", "false"),
        default="false",
        help="UniPC sigma schedule. 'false' (default) = native linear-flow + flow_shift; "
        "'true' = Karras (diffusers); 'auto' reads the checkpoint scheduler_config.json.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Also generate a joint audio stream (T2AV).",
    )
    parser.add_argument("--out", default=".cache/cosmos3_t2v")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--cfg",
        type=int,
        default=1,
        help="CFG rank-group size (1 or 2).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for managed workers to load the model.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if args.cfg not in (1, 2):
        raise SystemExit("--cfg must be 1 or 2.")
    if args.tp < 1:
        raise SystemExit("--tp must be positive.")

    from phyai import DeploymentConfig, Engine, EngineArgs
    from phyai.engine_config import (
        DeviceConfig,
        EngineConfig,
        AttentionParallelConfig,
        DenseParallelConfig,
        OuterParallelConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from phyai.models.cosmos3 import Cosmos3T2VRequest, pixel_to_latent_shape
    from phyai.models.cosmos3.main_cosmos3 import Cosmos3Args
    from phyai.server import WorkerSupervisorConfig
    from phyai_utils_tools.models.cosmos3 import (
        Cosmos3GenerationPostProcessor,
        Cosmos3Processor,
    )

    # The engine picks the executor: cfg=tp=1 runs inline; anything else spawns
    # one managed worker per rank on the first visible GPUs (select physical
    # GPUs with CUDA_VISIBLE_DEVICES on this launcher).
    deployment = DeploymentConfig(
        process_config=WorkerSupervisorConfig(startup_timeout_s=args.startup_timeout),
    )
    dtype = torch.bfloat16
    use_karras = {"auto": None, "true": True, "false": False}[args.use_karras_sigmas]

    timings: dict[str, float] = {}

    print("[engine] creating cosmos3 generation engine...")
    with _timed("model_load", timings):
        engine = Engine(
            EngineArgs(
                plugin="cosmos3",
                plugin_args=Cosmos3Args(
                    checkpoint_dir=args.checkpoint,
                    flow_shift=args.flow_shift,
                    use_karras_sigmas=use_karras,
                    load_sound=(True if args.sound else None),
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=dtype),
                    parallel=ParallelConfig(
                        outer=OuterParallelConfig(cfg_size=args.cfg),
                        dense=DenseParallelConfig(tp_size=args.tp),
                        attention=AttentionParallelConfig(tp_size=args.tp),
                    ),
                    runtime=RuntimeConfig(use_cuda_graph=False),
                ),
            ),
            deployment=deployment,
        )

    try:
        # Managed workers receive requests over a process pipe, so inputs are
        # built on the CPU; the inline engine takes them on the GPU directly.
        managed = engine.mode != "inline"
        request_device = "cpu" if managed else "cuda"
        with _timed("preprocess", timings):
            # Native-aligned prompt: metadata-appended positive + structured negative
            # (built by the processor; see the native-parity note in the docstring).
            processor = Cosmos3Processor(
                f"{args.checkpoint}/text_tokenizer",
                fps=args.fps,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                append_metadata=True,
            )
            cond, uncond = processor.tokenize_pair(
                args.prompt,
                negative_prompt=args.negative_prompt,
                device=request_device,
            )
            video_shape = pixel_to_latent_shape(
                args.num_frames, args.height, args.width
            )
            # Audio runs at sound_latent_fps (25); video at fps (see Cosmos3T2VRequest).
            sound_frames = (
                math.ceil(args.num_frames / args.fps * 25.0) if args.sound else None
            )
            request = Cosmos3T2VRequest(
                text_ids=cond.text_ids,
                text_mask=cond.text_mask,
                neg_text_ids=uncond.text_ids,
                neg_text_mask=uncond.text_mask,
                video_shape=video_shape,
                fps=args.fps,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                sound_frames=sound_frames,
            )

        print(
            f"[run] T2{'AV' if args.sound else 'V'} latent={video_shape} "
            f"steps={args.steps} guidance={args.guidance_scale} "
            f"cfg={args.cfg} tp={args.tp} mode={engine.mode}"
        )
        with _timed("inference", timings, synchronize_cuda=not managed):
            result = engine.step(request)

        postprocessor = Cosmos3GenerationPostProcessor(fps=args.fps)
        out_mp4 = f"{args.out}.mp4"
        with _timed("to_cpu", timings):
            media = postprocessor.postprocess(result)
        with _timed("encode", timings):
            postprocessor.save_mp4(media, out_mp4)
        print(
            f"[saved] -> {out_mp4}"
            + (
                f" (+{media.sample_rate} Hz audio)"
                if media.waveform is not None and media.sample_rate is not None
                else ""
            )
        )

        # --- timing breakdown (model_load = JuiceFS weight read; inference = denoise
        # + VAE decode; to_cpu = GPU->CPU pixel transfer; encode = ffmpeg mp4) ---
        print("\n=== timing (seconds) ===")
        for label in ("model_load", "preprocess", "inference", "to_cpu", "encode"):
            if label in timings:
                print(f"  {label:<11s}{timings[label]:9.2f}")
        if timings.get("inference") and args.steps > 0:
            print(
                f"  {'per-step':<11s}{timings['inference'] / args.steps:9.3f}"
                f"   ({args.steps} steps)"
            )
        print(f"  {'TOTAL':<11s}{sum(timings.values()):9.2f}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
