"""Cosmos3 processors — text-to-video tokenizer + action/policy processor"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as image_transforms

from phyai_utils_tools.models.cosmos3.positions_cosmos3 import (
    compute_mrope_position_ids_vision,
)

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.transition import IMAGES, TASK, Transition
from phyai_utils_tools.tokenizer import get_tokenizer

from phyai_utils_tools.models.cosmos3.steps_cosmos3 import (
    ACTION_CHUNK,
    COND_ACTION,
    DOMAIN_ID,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    MODE,
    NEG_TEXT_IDS,
    NEG_TEXT_MASK,
    RAW_ACTION_DIM,
    TEXT_IDS,
    TEXT_MASK,
    VIDEO_SHAPE,
    Cosmos3ActionPadStep,
    Cosmos3DomainResolveStep,
    Cosmos3ImagePreprocessStep,
    Cosmos3TextTokenizeStep,
    cosmos3_default_negative_prompt,
    cosmos3_generation_caption,
    find_closest_target_size,
    resolve_domain_id,
    resolve_raw_action_dim,
)
from phyai_utils_tools.processing.transition import PIXEL_VALUES


COSMOS3_VISION_START_TOKEN = "<|vision_start|>"


def _resolve_sample_asset(value: str, sample_path: Path) -> Path:
    """Resolve a local asset relative to its sample JSON."""
    if "://" in value:
        raise ValueError(
            "Download sample assets first and replace URLs with local paths: " + value
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = sample_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_sample_frames(path: Path, maximum: int) -> torch.Tensor:
    """Read RGB uint8 frames as [C,T,H,W], without temporal resampling."""
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        with Image.open(path) as source:
            frames = [np.array(source.convert("RGB"))]
    else:
        import av

        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == maximum:
                    break
    if not frames:
        raise ValueError(f"No video frames found in {path}")
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).contiguous()


def _resize_sample_frames(
    frames: torch.Tensor, resolution: str
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Apply Framework's uint8 bicubic resize and bottom/right padding."""
    height, width = frames.shape[-2:]
    target_w, target_h = find_closest_target_size(height, width, resolution)
    scale = min(target_w / width, target_h / height, 1.0)
    crop_h, crop_w = int(height * scale + 0.5), int(width * scale + 0.5)
    if (crop_h, crop_w) != (height, width):
        frames = image_transforms.resize(
            frames,
            [crop_h, crop_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    pad_h, pad_w = target_h - crop_h, target_w - crop_w
    if pad_h or pad_w:
        mode = "edge" if pad_h >= crop_h or pad_w >= crop_w else "reflect"
        frames = image_transforms.pad(frames, [0, 0, pad_w, pad_h], padding_mode=mode)
    return frames, (crop_h, crop_w)


def _flatten_chat_ids(out) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to ``list[int]``.

    Different transformers/tokenizers versions return a ``list[int]``, a nested
    ``[[int, ...]]``, or a ``BatchEncoding`` of ``tokenizers.Encoding`` objects.
    """
    # BatchEncoding / list whose first element exposes ``.ids`` (Encoding).
    first = out[0] if len(out) > 0 else None
    if hasattr(first, "ids"):
        return list(first.ids)
    if isinstance(first, (list, tuple)):
        return [int(x) for x in first]
    # Flat list of ints.
    return [int(x) for x in out]


@dataclass
class Cosmos3TokenizedPrompt:
    """Batch-1 tokenized prompt tensors."""

    text_ids: torch.Tensor  # [1, S] int64
    text_mask: torch.Tensor  # [1, S] int64 (all ones — no padding)


@dataclass
class Cosmos3GenerationOutput:
    """CPU-ready media output from the Cosmos3 generation plugin."""

    frames: torch.Tensor  # [T, H, W, 3] uint8 RGB, CPU
    video: torch.Tensor  # [B, 3, T, H, W] or [3, T, H, W], CPU float in [0, 1]
    waveform: torch.Tensor | None = None  # CPU float in [-1, 1]
    sample_rate: int | None = None


class Cosmos3Processor:
    """Qwen2 chat-template tokenizer for Cosmos3 T2V/I2V/T2AV prompts.

    When the generation dims (``fps``/``num_frames``/``height``/``width``) are given
    and ``append_metadata`` is set, the positive prompt gets native-style
    duration/resolution metadata appended (see :func:`cosmos3_generation_caption`).
    The negative prompt defaults to the native structured "bad video" negative
    (:func:`cosmos3_default_negative_prompt`); pass an explicit string (e.g. ``""``)
    to override.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        *,
        use_system_prompt: bool = False,
        fps: float | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        aspect_ratio: str | None = None,
        append_metadata: bool = True,
        negative_prompt: str | None = None,
    ) -> None:
        self.tokenizer = get_tokenizer(tokenizer_name_or_path)
        self.use_system_prompt = use_system_prompt
        self.fps = fps
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.aspect_ratio = aspect_ratio
        self.append_metadata = bool(append_metadata)
        self._negative_prompt = negative_prompt
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(COSMOS3_VISION_START_TOKEN)
        )

    def _augment(self, prompt: str) -> str:
        """Append native duration/resolution metadata when the dims are known."""
        if not self.append_metadata:
            return prompt
        if None in (self.fps, self.num_frames, self.height, self.width):
            return prompt
        return cosmos3_generation_caption(
            prompt,
            fps=self.fps,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            aspect_ratio=self.aspect_ratio,
        )

    def tokenize(
        self,
        prompt: str,
        *,
        device: torch.device | str = "cpu",
        augment: bool = True,
    ) -> Cosmos3TokenizedPrompt:
        """Tokenize one prompt -> ``[1, S]`` ids + all-ones mask.

        ``augment`` controls whether duration/resolution metadata is appended (on
        for the positive prompt; off for the negative, matching native).
        """
        content = self._augment(prompt) if augment else prompt
        conversation = []
        if self.use_system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant who will generate videos from a given prompt.",
                }
            )
        conversation.append({"role": "user", "content": content})
        out = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        ids = _flatten_chat_ids(out)
        ids = ids + [self.eos_token_id, self.vision_start_token_id]
        text_ids = torch.tensor([ids], dtype=torch.long, device=device)
        text_mask = torch.ones_like(text_ids)
        return Cosmos3TokenizedPrompt(text_ids=text_ids, text_mask=text_mask)

    def tokenize_pair(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Cosmos3TokenizedPrompt, Cosmos3TokenizedPrompt]:
        """Tokenize the conditional + unconditional prompts.

        ``negative_prompt=None`` falls back to the processor's ``negative_prompt``,
        then to the native structured default. Pass ``""`` for an empty negative.
        The positive prompt is metadata-augmented; the negative is not.
        """
        if negative_prompt is None:
            negative_prompt = self._negative_prompt
        if negative_prompt is None:
            negative_prompt = cosmos3_default_negative_prompt()
        return (
            self.tokenize(prompt, device=device, augment=True),
            self.tokenize(negative_prompt, device=device, augment=False),
        )


class Cosmos3GenerationPostProcessor:
    """Postprocess and save Cosmos3 generation media.

    ``cosmos3`` engine outputs are already VAE-decoded by the plugin:
    video-only requests return pixels in ``[0, 1]`` and T2AV requests return a
    ``{"video", "sound", "sample_rate"}`` dict. This class handles the output-side
    media glue: move tensors to CPU, convert video pixels to uint8 RGB frames, and
    optionally mux video + audio into one mp4 via PyAV.
    """

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)

    @staticmethod
    def _to_uint8_frames(video: torch.Tensor) -> torch.Tensor:
        """Convert ``[1,3,T,H,W]`` or ``[3,T,H,W]`` pixels to CPU uint8 frames."""
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(
                "Expected video shaped [1, 3, T, H, W] or [3, T, H, W], got "
                f"{tuple(video.shape)}."
            )
        return (
            (video.clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .contiguous()
        )

    def postprocess(
        self, output: torch.Tensor | dict[str, torch.Tensor | int]
    ) -> Cosmos3GenerationOutput:
        """Move generation output to CPU and prepare frames for media encoding."""
        if isinstance(output, dict):
            video = output["video"]
            waveform = output.get("sound")
            sample_rate = output.get("sample_rate")
        else:
            video = output
            waveform = None
            sample_rate = None
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected video tensor, got {type(video)!r}.")

        frames = self._to_uint8_frames(video)
        video_cpu = video.detach().cpu()
        waveform_cpu = (
            waveform.detach().clamp(-1.0, 1.0).float().cpu()
            if isinstance(waveform, torch.Tensor)
            else None
        )
        sample_rate_int = int(sample_rate) if sample_rate is not None else None
        return Cosmos3GenerationOutput(
            frames=frames,
            video=video_cpu,
            waveform=waveform_cpu,
            sample_rate=sample_rate_int,
        )

    def save_mp4(
        self,
        output: Cosmos3GenerationOutput,
        path: str | Path,
        *,
        crf: str = "18",
    ) -> None:
        """Encode frames, and optional waveform, into one mp4 via PyAV."""
        import av

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = output.frames.numpy()  # [T, H, W, 3] uint8 RGB
        with av.open(str(path), mode="w") as container:
            video_stream = container.add_stream(
                "h264", rate=Fraction(self.fps).limit_denominator(10000)
            )
            video_stream.width = int(arr.shape[2])
            video_stream.height = int(arr.shape[1])
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": crf}

            audio_stream = None
            audio_samples = None
            audio_layout = "stereo"
            if output.waveform is not None and output.sample_rate is not None:
                wav = output.waveform
                if wav.ndim == 3:
                    wav = wav[0]
                if wav.ndim == 1:
                    wav = wav.reshape(1, -1)
                if wav.ndim != 2:
                    raise ValueError(
                        "Expected waveform shaped [1, channels, samples], "
                        "[channels, samples], or [samples], got "
                        f"{tuple(output.waveform.shape)}."
                    )
                audio_samples = wav.numpy()
                audio_layout = "stereo" if audio_samples.shape[0] >= 2 else "mono"
                audio_stream = container.add_stream("aac", rate=int(output.sample_rate))
                audio_stream.layout = audio_layout

            for frame_data in arr:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            if audio_stream is not None:
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_samples, format="fltp", layout=audio_layout
                )
                audio_frame.sample_rate = int(output.sample_rate)
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)


@dataclass
class Cosmos3PolicyProcessedInputs:
    """Preprocessed inputs for the Cosmos3 action/policy path."""

    pixel_values: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    cond_action: torch.Tensor | None
    domain_id: int
    mode: str
    action_chunk: int
    raw_action_dim: int
    video_shape: tuple[int, int, int]
    cond_frame_indexes: tuple[int, ...] | None = None


@dataclass
class Cosmos3PolicySample:
    """CPU inputs and sampling settings for one action/video chunk."""

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    pixels: tuple[torch.Tensor, ...]
    action: torch.Tensor
    fps: float
    num_steps: int
    flow_shift: float
    guidance_scale: float
    seed: int
    prompt: str
    raw_action_dim: int
    mode: str
    domain_id: int
    action_chunk_size: int
    num_frames: int
    crop_size: tuple[int, int]
    latent_crop_size: tuple[int, int]
    source: Path
    condition_frame_indexes: tuple[int, ...]


class Cosmos3PolicyProcessor(BaseModelProcessor):
    """Cosmos3 action/policy pre/post processor.

    Preprocessing: image resize/normalize, text tokenize, action pad, domain resolve.
    Postprocessing: slice action to raw_action_dim, move to CPU.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        mode: str = "policy",
        domain_name: str | int = "agibotworld",
        action_chunk_size: int = 16,
        raw_action_dim: int | None = None,
        action_dim: int = 64,
        negative_prompt: str = "",
        fps: float = 24.0,
        image_size: int | None = None,
        append_metadata: bool = True,
        prompt_format: str = "plain",
        view_point: str = "ego_view",
        cond_frame_indexes: tuple[int, ...] | None = None,
        action_stats_path: str | None = None,
        action_normalization: str = "minmax",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.mode = mode
        self.domain_name = domain_name
        self.action_chunk_size = int(action_chunk_size)
        # Resolve the embodiment's raw action width: explicit value wins (and is
        # cross-checked against the embodiment table when both are known);
        # otherwise it is looked up from the domain name.
        self.raw_action_dim = self._resolve_raw_action_dim(raw_action_dim, domain_name)
        self.action_dim = int(action_dim)
        self.negative_prompt = negative_prompt
        self.fps = float(fps)
        self.image_size = int(image_size) if image_size is not None else None
        self.append_metadata = bool(append_metadata)
        self.prompt_format = prompt_format
        self.view_point = view_point
        self.cond_frame_indexes = (
            tuple(cond_frame_indexes) if cond_frame_indexes is not None else None
        )
        self.action_normalization = action_normalization
        self.device = device
        self.params_dtype = params_dtype
        # Optional output denormalization: tensors built once from an external stats
        # JSON, applied in postprocess.
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._action_min: torch.Tensor | None = None
        self._action_range: torch.Tensor | None = None
        if action_stats_path is not None:
            self._load_action_stats(action_stats_path)
        super().__init__()

    @classmethod
    def preprocess_sample(
        cls, sample_path: str | Path, checkpoint: str | Path, *, chunk_index: int = 0
    ) -> Cosmos3PolicySample:
        """Load a local Framework action sample and preprocess one chunk.

        Image/video and action paths are relative to the sample JSON. Released
        AgiBot ``action_chunks`` files support selection with ``chunk_index``.
        Conditioning actions are already in the model's normalized coordinates.
        """
        sample_path, checkpoint = Path(sample_path), Path(checkpoint)
        sample = json.loads(sample_path.read_text())
        if not isinstance(sample, dict):
            raise ValueError("The Framework sample must be a JSON object")
        action_data = None
        if sample.get("action_path"):
            action_path = _resolve_sample_asset(str(sample["action_path"]), sample_path)
            action_data = json.loads(action_path.read_text())
            if isinstance(action_data, dict):
                metadata = action_data
                chunks = metadata.get("action_chunks")
                if not isinstance(chunks, list) or not 0 <= chunk_index < len(chunks):
                    raise ValueError("Invalid action_chunks JSON or chunk_index")
                action_data = chunks[chunk_index]
                sample = {**metadata, **sample}
            elif chunk_index != 0:
                raise ValueError("chunk_index requires an action_chunks JSON file")
        elif chunk_index != 0:
            raise ValueError("chunk_index requires an action_chunks JSON file")

        mode = str(sample.get("model_mode", "wam"))
        mode = "policy" if mode == "wam" else mode
        if mode not in {"policy", "forward_dynamics", "inverse_dynamics"}:
            raise ValueError(f"Unsupported Framework action mode: {mode}")
        if sample.get("native_prompt_upsampling", False) or sample.get(
            "prompt_upsampling", False
        ):
            raise ValueError("Samples require a final prompt without upsampling")
        if sample.get("enable_sound", False) or sample.get("autoregressive", False):
            raise ValueError("Samples support single-chunk video/action generation")
        if int(sample.get("num_outputs", 1)) != 1:
            raise ValueError("Samples support one output per request")
        if sample.get("condition_video_keep", "first") != "first":
            raise ValueError("Samples use the first observation frames")
        chunk_size = int(sample.get("action_chunk_size", 16))
        if chunk_size <= 0 or chunk_size % 4:
            raise ValueError("action_chunk_size must be positive and divisible by four")
        transformer_config = json.loads(
            (checkpoint / "transformer/config.json").read_text()
        )
        action_dim = int(transformer_config["action_dim"])
        vae_config = json.loads((checkpoint / "vae/config.json").read_text())
        spatial_factor = int(vae_config.get("scale_factor_spatial", 16))
        raw_dim = sample.get("raw_action_dim")
        if mode == "forward_dynamics":
            if action_data is None:
                raise ValueError("forward_dynamics requires an action_path")
            actions = np.asarray(action_data, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] != chunk_size:
                raise ValueError(
                    "Conditioning actions must have shape [action_chunk_size,D]"
                )
            if not np.isfinite(actions).all() or not 0 < actions.shape[1] <= action_dim:
                raise ValueError(
                    "Conditioning actions must be finite with a valid width"
                )
            if raw_dim is not None and int(raw_dim) != actions.shape[1]:
                raise ValueError("raw_action_dim does not match the action file")
            raw_dim = actions.shape[1]
            action_data = actions.tolist()

        num_frames = chunk_size + 1
        frames = _read_sample_frames(
            _resolve_sample_asset(str(sample["vision_path"]), sample_path), num_frames
        )
        # Framework's action entry passes image_size as its resolution bucket.
        resolution = str(sample.get("image_size", 256))
        frames, crop_size = _resize_sample_frames(frames, resolution)
        height, width = frames.shape[-2:]
        if frames.shape[1] < num_frames:
            frames = torch.cat(
                (
                    frames,
                    frames[:, -1:].expand(-1, num_frames - frames.shape[1], -1, -1),
                ),
                dim=1,
            )
        latent_frames = (num_frames - 1) // 4 + 1
        clean = sample.get("condition_frame_indexes_vision")
        if clean is None:
            clean = list(range(latent_frames)) if mode == "inverse_dynamics" else [0]
        if (
            not isinstance(clean, list)
            or not clean
            or any(
                not isinstance(index, int) or not 0 <= index < latent_frames
                for index in clean
            )
        ):
            raise ValueError(
                "condition_frame_indexes_vision contains invalid latent indexes"
            )
        # The causal VAE only needs the prefix ending at the last clean latent.
        frames = frames[:, : 1 + 4 * max(clean)]
        fps = float(sample.get("fps", 24.0))
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError("fps must be positive and finite")
        prompt = str(sample.get("prompt", ""))
        processor = cls(
            tokenizer_name_or_path=str(checkpoint / "text_tokenizer"),
            height=height,
            width=width,
            num_frames=num_frames,
            mode=mode,
            domain_name=sample["domain_name"],
            action_chunk_size=chunk_size,
            raw_action_dim=None if raw_dim is None else int(raw_dim),
            action_dim=action_dim,
            fps=fps,
            prompt_format="json",
            view_point=str(sample.get("view_point", "ego_view")),
            negative_prompt=str(sample.get("negative_prompt", "")),
        )
        # Reuse the text/action/domain steps with already resized tensor pixels.
        pipeline = ProcessorPipeline(
            steps=processor.preprocessor.steps[1:],
            to_output=processor.preprocessor.to_output,
        )
        processed = pipeline(
            {
                PIXEL_VALUES: (frames.float() / 127.5 - 1.0).unsqueeze(0),
                VIDEO_SHAPE: (frames.shape[1], height, width),
                TASK: prompt,
                COND_ACTION: action_data if mode == "forward_dynamics" else None,
                DOMAIN_ID: sample["domain_name"],
                MODE: mode,
            }
        )
        tokenizer = get_tokenizer(str(checkpoint / "text_tokenizer"))
        if tokenizer.bos_token_id is not None:
            bos = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long)
            processed.text_ids = torch.cat((bos, processed.text_ids), dim=1)
            processed.text_mask = torch.ones_like(processed.text_ids)
        action = processed.cond_action
        if action is None:
            action = torch.zeros((1, chunk_size, action_dim), dtype=torch.float32)
        return Cosmos3PolicySample(
            text_ids=processed.text_ids,
            text_mask=processed.text_mask,
            pixels=(processed.pixel_values,),
            action=action,
            fps=fps,
            num_steps=int(sample.get("num_steps", 30)),
            flow_shift=float(sample.get("shift", 10.0)),
            guidance_scale=float(sample.get("guidance", 1.0)),
            seed=int(sample.get("seed", 0)),
            prompt=prompt,
            raw_action_dim=processed.raw_action_dim,
            mode=mode,
            domain_id=processed.domain_id,
            action_chunk_size=chunk_size,
            num_frames=num_frames,
            crop_size=crop_size,
            latent_crop_size=tuple(
                max(size // spatial_factor, 1) for size in crop_size
            ),
            source=sample_path,
            condition_frame_indexes=tuple(clean),
        )

    @staticmethod
    def prepare_denoise_inputs(
        sample: Cosmos3PolicySample,
        encoded: tuple[torch.Tensor, ...],
        *,
        model_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Prepare latents, masks and positions from a completed VAE encode.

        ``model_config`` contains the canonical transformer configuration fields.
        The returned dictionary contains model inputs; the caller supplies
        inference settings when constructing its request.
        """
        if sample.guidance_scale != 1.0:
            raise ValueError("Sample denoising currently requires guidance=1")
        if len(encoded) != 1:
            raise ValueError("Framework action samples require one encoded video")
        observed = encoded[0]
        if observed.ndim != 5 or observed.shape[0] != 1:
            raise ValueError("Encoded video must have shape [1,C,T,H,W]")
        # Framework crops encoder padding in latent space before GEN packing.
        crop_h, crop_w = sample.latent_crop_size
        observed = observed[..., :crop_h, :crop_w].contiguous()
        _, channels, observed_frames, height, width = observed.shape
        temporal_factor = model_config["temporal_compression_factor"]
        frames = (sample.num_frames - 1) // temporal_factor + 1
        if channels != model_config["latent_channel"]:
            raise ValueError("Encoded channel count does not match the transformer")
        if max(sample.condition_frame_indexes) >= observed_frames:
            raise ValueError("The encoded video is missing a conditioning frame")
        vision = observed.new_zeros((1, channels, frames, height, width))
        video_noisy = torch.ones((1, frames), dtype=torch.bool)
        for index in sample.condition_frame_indexes:
            vision[:, :, index] = observed[:, :, index]
            video_noisy[:, index] = False

        patch = model_config["latent_patch_size"]
        offset = sample.text_ids.shape[1] + model_config["temporal_modality_margin"]
        video_positions, _ = compute_mrope_position_ids_vision(
            frames,
            (height + patch - 1) // patch,
            (width + patch - 1) // patch,
            temporal_offset=offset,
            fps=sample.fps,
            base_fps=model_config["base_fps"],
            temporal_compression_factor=temporal_factor,
            enable_fps_modulation=model_config["enable_fps_modulation"],
        )
        action_positions, _ = compute_mrope_position_ids_vision(
            sample.action_chunk_size,
            1,
            1,
            temporal_offset=offset,
            fps=sample.fps,
            base_fps=model_config["base_fps"],
            temporal_compression_factor=1,
            base_temporal_compression_factor=temporal_factor,
            enable_fps_modulation=model_config["enable_fps_modulation"],
            start_frame_offset=1,
        )
        action = sample.action.to(dtype=torch.float32)
        if action.shape[-1] != model_config["action_dim"]:
            raise ValueError("Sample action padding does not match the transformer")
        action_valid = torch.zeros_like(action, dtype=torch.bool)
        action_valid[..., : sample.raw_action_dim] = True
        action_noisy = torch.full(
            (1, sample.action_chunk_size),
            sample.mode != "forward_dynamics",
            dtype=torch.bool,
        )
        return {
            "vision": (vision,),
            "video_noisy_masks": (video_noisy,),
            "gen_position_ids": torch.cat((video_positions, action_positions), dim=1)
            .to(dtype=torch.float32)
            .unsqueeze(1),
            "action": action,
            "action_noisy_mask": action_noisy,
            "action_valid_mask": action_valid,
            "domain_id": sample.domain_id,
        }

    @staticmethod
    def postprocess_sample(
        output: dict[str, Any], sample: Cosmos3PolicySample
    ) -> dict[str, Any]:
        """Crop decoded pixels, slice normalized actions and move tensors to CPU."""
        result = dict(output)
        if isinstance(result.get("action"), torch.Tensor):
            result["action"] = (
                result["action"][..., : sample.raw_action_dim].detach().cpu()
            )
        height, width = sample.crop_size
        for key in ("video", "pixels"):
            if isinstance(result.get(key), torch.Tensor):
                result[key] = (
                    result[key][..., :height, :width].detach().cpu().contiguous()
                )
        return result

    def _resolve_raw_action_dim(
        self, raw_action_dim: int | None, domain_name: str | int
    ) -> int:
        """Resolve raw_action_dim from the embodiment, cross-checking if explicit."""
        if raw_action_dim is None:
            return resolve_raw_action_dim(domain_name)
        raw_action_dim = int(raw_action_dim)
        # Cross-check against the table when the domain name is resolvable.
        if isinstance(domain_name, str):
            try:
                expected = resolve_raw_action_dim(domain_name)
            except ValueError:
                expected = None
            if expected is not None and expected != raw_action_dim:
                raise ValueError(
                    f"raw_action_dim={raw_action_dim} conflicts with the table value "
                    f"{expected} for domain_name={domain_name!r}; pass the matching "
                    f"value or omit raw_action_dim to auto-resolve."
                )
        return raw_action_dim

    def _load_action_stats(self, stats_path: str) -> None:
        """Load output-denormalization tensors from an external stats JSON.

        The JSON is a dict, optionally with a ``"global"`` (or ``"global_raw"`` for
        ``quantile_rot``) block. ``meanstd`` uses ``mean``/``std``; ``minmax`` uses
        ``min``/``max``; ``quantile``/``quantile_rot`` use ``q01``/``q99``.
        """
        import json

        with open(stats_path) as f:
            raw_stats = json.load(f)
        if not isinstance(raw_stats, dict):
            raise ValueError(f"Action stats file must contain a dict: {stats_path}")
        stats_key = (
            "global_raw" if self.action_normalization == "quantile_rot" else "global"
        )
        stats = raw_stats.get(stats_key, raw_stats)
        if self.action_normalization == "meanstd":
            self._action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self._action_std = torch.clamp(
                torch.tensor(stats["std"], dtype=torch.float32), min=1e-8
            )
        elif self.action_normalization in ("quantile", "quantile_rot"):
            self._action_min = torch.tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"], dtype=torch.float32)
            self._action_range = torch.clamp(q99 - self._action_min, min=1e-6)
        elif self.action_normalization == "minmax":
            self._action_min = torch.tensor(stats["min"], dtype=torch.float32)
            amax = torch.tensor(stats["max"], dtype=torch.float32)
            self._action_range = torch.clamp(amax - self._action_min, min=1e-6)
        else:
            raise ValueError(
                "action_normalization must be one of 'meanstd', 'minmax', "
                f"'quantile', 'quantile_rot'; got {self.action_normalization!r}."
            )

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert the configured normalization on the raw-dim action channels.

        Slice to the stats width, then ``x*std+mean`` (meanstd) or
        ``(x+1)/2*range+min`` (minmax / quantile). No-op when no stats are loaded.
        """
        if self._action_mean is not None and self._action_std is not None:
            dim = self._action_mean.shape[0]
            mean = self._action_mean.to(action.device)
            std = self._action_std.to(action.device)
            return action[..., :dim] * std + mean
        if self._action_min is not None and self._action_range is not None:
            dim = self._action_min.shape[0]
            amin = self._action_min.to(action.device)
            arange = self._action_range.to(action.device)
            return (action[..., :dim] + 1.0) / 2.0 * arange + amin
        return action

    def _to_transition(self, raw: dict[str, Any]) -> Transition:
        """Adapt caller's raw dict into the canonical transition."""
        t: Transition = {}
        t[IMAGES] = raw.get("images")
        t[TASK] = raw.get("task", raw.get("prompt", ""))
        t[COND_ACTION] = raw.get("action") or raw.get("cond_action")
        t[DOMAIN_ID] = raw.get("domain_name", raw.get("domain_id", self.domain_name))
        t[MODE] = raw.get("mode", self.mode)
        return t

    def _to_output(self, transition: Transition) -> Cosmos3PolicyProcessedInputs:
        """Extract typed output from the final transition."""
        return Cosmos3PolicyProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            text_ids=transition[TEXT_IDS],
            text_mask=transition[TEXT_MASK],
            neg_text_ids=transition[NEG_TEXT_IDS],
            neg_text_mask=transition[NEG_TEXT_MASK],
            cond_action=transition.get(COND_ACTION),
            domain_id=transition[DOMAIN_ID],
            mode=transition[MODE],
            action_chunk=transition[ACTION_CHUNK],
            raw_action_dim=transition[RAW_ACTION_DIM],
            video_shape=transition[VIDEO_SHAPE],
            cond_frame_indexes=self.cond_frame_indexes,
        )

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = [
            Cosmos3ImagePreprocessStep(
                height=self.height,
                width=self.width,
                mode=self.mode,
                image_size=self.image_size,
            ),
            Cosmos3TextTokenizeStep(
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                negative_prompt=self.negative_prompt,
                append_metadata=self.append_metadata,
                prompt_format=self.prompt_format,
                view_point=self.view_point,
                # Duration in the caption uses the rollout length = chunk + 1
                # (the target frame count), not the observation frame count.
                fps=self.fps,
                num_frames=self.action_chunk_size + 1,
            ),
            Cosmos3ActionPadStep(
                action_chunk_size=self.action_chunk_size,
                raw_action_dim=self.raw_action_dim,
                action_dim=self.action_dim,
                mode=self.mode,
            ),
            Cosmos3DomainResolveStep(),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="cosmos3_policy_preprocessor",
            to_transition=self._to_transition,
            to_output=self._to_output,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[],
            name="cosmos3_policy_postprocessor",
            to_transition=lambda x: x,
            to_output=lambda x: x,
        )

    def postprocess(self, output: dict[str, Any] | torch.Tensor) -> dict[str, Any]:
        """Slice action to raw_action_dim, optionally denormalize, move to CPU.

        When action stats were loaded (``action_stats_path``), the sliced action is
        denormalized back to physical units before moving to CPU; otherwise it is
        returned in the model's (normalized) action space.
        """
        if isinstance(output, torch.Tensor):
            action = self._denormalize_action(output[:, :, : self.raw_action_dim])
            return {"action": action.cpu()}
        result: dict[str, Any] = {}
        if "action" in output:
            action = self._denormalize_action(
                output["action"][:, :, : self.raw_action_dim]
            )
            result["action"] = action.cpu()
        if "pixels" in output:
            result["pixels"] = output["pixels"].cpu()
        if "video" in output:
            result["video"] = output["video"].cpu()
        return result


__all__ = [
    "Cosmos3GenerationOutput",
    "Cosmos3GenerationPostProcessor",
    "Cosmos3PolicyProcessedInputs",
    "Cosmos3PolicyProcessor",
    "Cosmos3PolicySample",
    "Cosmos3Processor",
    "Cosmos3TokenizedPrompt",
    "COSMOS3_VISION_START_TOKEN",
    "EMBODIMENT_TO_DOMAIN_ID",
    "EMBODIMENT_TO_RAW_ACTION_DIM",
    "cosmos3_default_negative_prompt",
    "cosmos3_generation_caption",
    "resolve_domain_id",
    "resolve_raw_action_dim",
]
