"""Engine plugin for native Emu3.5 image decoding with local weights."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.xiaomi_u0.configuration_vision import U0VisionConfig
from phyai.models.xiaomi_u0.modeling_vision import (
    U0VisionTokenizer,
    vision_decode_weight_remap,
)
from phyai.models.xiaomi_u0.model_runner_vision import U0VisionRunner
from phyai.models.xiaomi_u0.scheduler_vision import U0VisionScheduler
from phyai.weights.loader import load_pretrained


@dataclass
class U0VisionArgs(EntryArgs):
    """Local VisionTokenizer checkpoint directory for the image decoder."""

    checkpoint_dir: str = ""


@Engine.register
class U0VisionEntry(Entry):
    """Load only the native codebook and decoder used for single-image generation."""

    name: ClassVar[str] = "xiaomi_u0_decode"
    args_cls: ClassVar[type[EntryArgs]] = U0VisionArgs

    def __init__(self) -> None:
        """Create an entry without allocating model parameters."""
        self.model: U0VisionTokenizer | None = None
        self.scheduler: U0VisionScheduler | None = None

    def setup(self, args: U0VisionArgs) -> None:
        """Build the FP32 decoder and load its required tensors strictly."""
        if self.model is not None:
            raise RuntimeError("The vision entry is already set up.")
        checkpoint = Path(args.checkpoint_dir)
        if not args.checkpoint_dir or not checkpoint.is_dir():
            raise ValueError(
                "A local VisionTokenizer checkpoint directory is required."
            )
        engine = get_engine_config()
        if engine.runtime.use_cuda_graph:
            raise ValueError("Native U0 image decoding requires CUDA Graph disabled.")
        if engine.device.params_dtype != torch.float32:
            raise ValueError("The released VisionTokenizer requires FP32 parameters.")
        config = U0VisionConfig.from_json(checkpoint / "config.json")
        if (
            config.codebook_size != 131072
            or config.embed_dim != 256
            or config.spatial_factor != 16
        ):
            raise ValueError(
                "The U0 example requires the released Emu3.5 codebook and 16x spatial scale."
            )
        try:
            self.model = U0VisionTokenizer(config, device=engine.device.target).eval()
            report = load_pretrained(
                self.model, checkpoint, remap=vision_decode_weight_remap, strict=True
            )
            if report.casts:
                raise ValueError(
                    "VisionTokenizer weights must load in FP32 without precision casts."
                )
            self.scheduler = U0VisionScheduler(U0VisionRunner(self.model))
            self.scheduler.setup()
        except Exception:
            self.close()
            raise

    def step(self, codes: torch.Tensor) -> torch.Tensor:
        """Delegate one image-code grid to the decoder scheduler."""
        if self.scheduler is None:
            raise RuntimeError("Set up the vision entry before decoding.")
        return self.scheduler.step(codes)

    def close(self) -> None:
        """Release the scheduler and native decoder model."""
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        """Expose the loaded decoder to engine inspection tools."""
        return {} if self.model is None else {"vision": self.model}
