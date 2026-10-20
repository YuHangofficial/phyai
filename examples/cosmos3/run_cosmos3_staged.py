"""Run Cosmos3 as VLM -> VAE encoder -> DiT -> VAE decoder on one GPU.

    CUDA_VISIBLE_DEVICES=0 uv run python examples/cosmos3/run_cosmos3_staged.py \\
        --checkpoint /path/to/Cosmos3-Nano \\
        --sample examples/cosmos3/assets/agibot_forward_dynamics.json \\
        --offload --output .cache/cosmos3/agibot.pt

The bundled AgiBot sample uses the first frame and first action chunk in assets/.
It generates a 17-frame video at 10 FPS with 30 denoise steps. The sample JSON
resolves its image and action paths relative to itself. See README.md or
README_ZH.md for the recorded Framework/PHYAI frame comparison.

--offload moves each stage's weights back to CPU after use. VLM K/V stay on GPU
until the next prefill or engine.close(). CUDA graphs are enabled by default for
the VAE encoder, complete DiT denoise loop, and VAE decoder. --no-cuda-graph runs
these stages eagerly. Moving weights releases the corresponding graphs first.

Default inputs are synthetic RGB clips. --latent-input skips the VAE encoder and
uses fake latents directly. Both routes use the prompt and six latent shapes in
customized_inputs.py, totaling 583 GEN tokens. The example runs native PHYAI UniPC
with 10 steps, shift=2 and no CFG, then saves [48,49] actions and a 13-frame rollout.
These fake observations demonstrate the API; they do not evaluate policy quality.
Optional visual role embeddings are left unset.

--sample accepts a Framework action sample whose vision_path and action_path
point to local files, as in assets/agibot_forward_dynamics.json. It supplies
its prompt, resolution, embodiment, FPS, seed and sampling settings; --steps and
--seed can override them. This path supports guidance=1 and uses the public
Framework's noise and timestep conventions. Choose a checkpoint trained for that
sample's domain. The output includes a tensor file and an MP4 beside it.
Forward-dynamics samples predict video from supplied actions; their saved actions
are the input conditions. Each invocation generates one action chunk.

Each stage runs --warmup times before --repeat timed runs. Weights stay resident
through those runs and are offloaded afterwards. DiT graph capture is therefore
excluded from run timings. Synchronized wall times include stage input preparation
and, for DiT, the complete denoise loop. Weight load/offload times are separate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from phyai import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3Config
from phyai.models.cosmos3.main_cosmos3_staged import Cosmos3StagedArgs
from phyai.models.cosmos3.requests_staged_cosmos3 import (
    Cosmos3DecodeRequest,
    Cosmos3DenoiseRequest,
    Cosmos3EncodeRequest,
    Cosmos3VLMRequest,
)
from phyai.utils import get_logger, load_config
from phyai_utils_tools.models.cosmos3.processor_cosmos3 import (
    Cosmos3GenerationPostProcessor,
    Cosmos3PolicyProcessor,
)

from customized_inputs import (
    PROMPT,
    make_denoise_request,
    make_fake_pixels,
    pool_encoded_history,
    tokenize_prompt,
)


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--sample",
        type=Path,
        help="Framework policy/forward-dynamics sample JSON with local input assets",
    )
    parser.add_argument(
        "--action-chunk-index",
        type=int,
        default=0,
        help="Select one action chunk; vision_path must supply its starting observation",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Offload each stage's weights after execution",
    )
    parser.add_argument(
        "--offload-vlm",
        action="store_true",
        help="Offload only VLM weights after prefill",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture VAE stages and the complete DiT rollout (default: enabled)",
    )
    parser.add_argument(
        "--rollout-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one graph for all DiT steps; disabling uses one graph per step",
    )
    parser.add_argument(
        "--latent-input",
        action="store_true",
        help="Start from fake VAE latents and skip encoding",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--warmup", type=int, default=3, help="Untimed full runs per stage"
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Measured full runs per stage"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/cosmos3_staged/output.pt"),
        help="Tensor output (.pt); an MP4 is saved with the same stem",
    )
    args = parser.parse_args()
    if min(args.steps or 10, args.warmup, args.repeat) < 1 or args.steps == 0:
        parser.error("--steps, --warmup and --repeat must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if args.output.suffix != ".pt":
        parser.error("--output must end in .pt; the MP4 is saved beside it")

    torch.set_num_threads(2)
    sample = None
    if args.sample is not None:
        if args.latent_input or args.prompt is not None:
            parser.error("--sample supplies its own RGB observations and prompt")
        sample = Cosmos3PolicyProcessor.preprocess_sample(
            args.sample, args.checkpoint, chunk_index=args.action_chunk_index
        )
        if sample.guidance_scale != 1.0:
            parser.error("The staged example currently requires sample guidance=1")
        text, text_mask = sample.text_ids, sample.text_mask
        prompt = sample.prompt
    else:
        prompt = args.prompt or PROMPT
        text = tokenize_prompt(args.checkpoint, prompt)
        text_mask = torch.ones_like(text)
    steps = (
        args.steps if args.steps is not None else (sample.num_steps if sample else 10)
    )
    seed = args.seed if args.seed is not None else (sample.seed if sample else 9000)
    config = load_config(args.checkpoint / "transformer", Cosmos3Config)
    engine = Engine(
        EngineArgs(
            plugin="cosmos3_staged",
            plugin_args=Cosmos3StagedArgs(
                checkpoint_dir=args.checkpoint,
                flow_shift=sample.flow_shift if sample else 2.0,
                framework_compat=sample is not None,
                use_rollout_graph=args.rollout_graph,
                warmup_runs=args.warmup,
                timing_runs=args.repeat,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=args.cuda_graph, num_threads=2),
            ),
        )
    )
    try:
        # Stage 1: UND text prefill. The engine owns K/V independently of weights.
        info = engine.step(
            Cosmos3VLMRequest(
                text,
                text_mask,
                offload_after=args.offload or args.offload_vlm,
            )
        )
        logger.info("VLM prefill: %s", info)

        # Stage 2: RGB [-1,1] -> normalized conditioning latents.
        vision = None
        if not args.latent_input:
            encoded = engine.step(
                Cosmos3EncodeRequest(
                    sample.pixels if sample else make_fake_pixels(seed),
                    offload_after=args.offload,
                )
            )
            vision = encoded if sample else pool_encoded_history(encoded)

        # Stage 3: all DiT steps reuse the cached text K/V, including after offload.
        request = (
            Cosmos3DenoiseRequest(
                **Cosmos3PolicyProcessor.prepare_denoise_inputs(
                    sample, vision, model_config=asdict(config)
                ),
                num_inference_steps=steps,
                seed=seed,
                offload_after=args.offload,
            )
            if sample
            else make_denoise_request(
                text.shape[1],
                config,
                seed=seed,
                vision=vision,
                steps=steps,
                offload_after=args.offload,
            )
        )
        result = engine.step(request)

        # Stage 4: decode the generated visual stream.
        pixels = engine.step(
            Cosmos3DecodeRequest(result["vision"][-1], offload_after=args.offload)
        )
        if sample:
            processed = Cosmos3PolicyProcessor.postprocess_sample(
                {"video": pixels, "action": result["action"]}, sample
            )
            pixels, actions = processed["video"], processed["action"][0]
        else:
            actions = result["action"][0, 1:, :49].cpu()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt": prompt,
                "action": actions,
                "video": pixels.cpu(),
                "video_latents": tuple(x.cpu() for x in result["vision"]),
            },
            args.output,
        )
        # H.264 4:2:0 needs even dimensions; preserve an odd crop in the tensor file.
        video_for_file = F.pad(
            pixels,
            (0, pixels.shape[-1] % 2, 0, pixels.shape[-2] % 2, 0, 0),
            mode="replicate",
        )
        postprocessor = Cosmos3GenerationPostProcessor(fps=sample.fps if sample else 24)
        video_path = args.output.with_suffix(".mp4")
        postprocessor.save_mp4(postprocessor.postprocess(video_for_file), video_path)
        logger.info(
            "Saved action %s and video %s to %s",
            tuple(actions.shape),
            tuple(pixels.shape),
            args.output,
        )
        logger.info("Saved video to %s", video_path)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
