"""Single-image scheduling for the native U0 pixel decoder."""

import torch

from phyai.models.xiaomi_u0.model_runner_vision import U0VisionRunner
from phyai.runtime.schedule import Scheduler


class U0VisionScheduler(Scheduler):
    """Validate one released 64-by-64 code grid and decode it through the runner."""

    def __init__(self, runner: U0VisionRunner) -> None:
        """Retain the decoder runner for image requests."""
        self.runner = runner

    def setup(self) -> None:
        """Prepare the image decoder runner."""
        self.runner.setup()

    @torch.inference_mode()
    def step(self, codes: torch.Tensor) -> torch.Tensor:
        """Return finite FP32 RGB pixels for a single grid of valid codebook IDs."""
        if not isinstance(codes, torch.Tensor) or codes.shape != (1, 64, 64):
            raise ValueError("The U0 image decoder expects a (1, 64, 64) code grid.")
        if (
            codes.dtype != torch.int64
            or (codes < 0).any().item()
            or (codes >= 131072).any().item()
        ):
            raise ValueError("Image code grids must contain int64 IDs in [0, 131072).")
        pixels = self.runner.forward(codes)
        if (
            pixels.shape != (1, 3, 1024, 1024)
            or not torch.isfinite(pixels).all().item()
        ):
            raise RuntimeError("The native image decoder returned invalid RGB pixels.")
        return pixels.float()

    def close(self) -> None:
        """Release the runner's request lifecycle state."""
        self.runner.close()
