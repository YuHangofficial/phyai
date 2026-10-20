"""Single-GPU Cosmos3 entry with explicit VLM, VAE, and DiT stages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config, set_engine_config
from phyai.layers.quant.active import resolve_quant_plan
from phyai.models.cosmos3.configuration_cosmos3 import (
    Cosmos3Config,
    Cosmos3WanVAEConfig,
)
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Transformer,
    cosmos3_weight_remap,
)
from phyai.models.cosmos3.requests_staged_cosmos3 import Cosmos3StageRequest
from phyai.models.cosmos3.scheduler_staged_cosmos3 import Cosmos3StagedScheduler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, cosmos3_vae_weight_remap
from phyai.utils import get_logger, load_config
from phyai.weights import load_pretrained


logger = get_logger(__name__)


@dataclass
class Cosmos3StagedArgs(EntryArgs):
    """Load weights on CPU; each stage request controls its GPU residency.

    The staged path supports unquantized video/action checkpoints on one GPU.
    Existing ``cosmos3`` and ``cosmos3_policy`` plugins retain their APIs.
    """

    checkpoint_dir: str | Path | None = None
    config: Cosmos3Config | None = None
    load_vae: bool = True
    flow_shift: float = 2.0
    use_karras_sigmas: bool = False
    weight_strict: bool = True
    use_rollout_graph: bool = True
    framework_compat: bool = False
    # Optional stage benchmarks: warmups and repeats run before any offload.
    warmup_runs: int = 0
    timing_runs: int = 0


@Engine.register
class Cosmos3StagedEntry(Entry):
    name: ClassVar[str] = "cosmos3_staged"
    args_cls: ClassVar[type[EntryArgs]] = Cosmos3StagedArgs

    def __init__(self) -> None:
        self.transformer: Cosmos3Transformer | None = None
        self.vae: Cosmos3WanVAE | None = None
        self.scheduler: Cosmos3StagedScheduler | None = None

    def setup(self, args: Cosmos3StagedArgs) -> None:
        if args.checkpoint_dir is None:
            raise ValueError("Cosmos3StagedArgs.checkpoint_dir is required")
        checkpoint = Path(args.checkpoint_dir)
        engine = get_engine_config()
        if torch.device(engine.device.target).type != "cuda":
            raise ValueError("The Cosmos3 staged entry requires a CUDA device")
        if resolve_quant_plan(checkpoint / "transformer") is not None:
            raise ValueError(
                "Stage migration currently supports unquantized Cosmos3 weights"
            )
        config = args.config or load_config(checkpoint / "transformer", Cosmos3Config)

        # Shared layer constructors read the engine device. Restore it even if
        # loading fails; network weights do not need to visit CUDA during setup.
        set_engine_config(engine.replace(device=replace(engine.device, target="cpu")))
        try:
            self.transformer = (
                Cosmos3Transformer(
                    config, params_dtype=engine.device.params_dtype, device="cpu"
                )
                .eval()
                .requires_grad_(False)
            )
            if args.framework_compat:
                # The public generation path evaluates the timestep MLP in FP32.
                self.transformer.time_embedder.to(dtype=torch.float32)
            report = load_pretrained(
                self.transformer,
                checkpoint / "transformer",
                remap=cosmos3_weight_remap,
                strict=args.weight_strict,
            )
            logger.info_rank0("Cosmos3 staged transformer: %s", report.summary())
            if args.load_vae:
                self.vae = (
                    Cosmos3WanVAE(load_config(checkpoint / "vae", Cosmos3WanVAEConfig))
                    .to(device="cpu", dtype=engine.device.params_dtype)
                    .eval()
                    .requires_grad_(False)
                )
                report = load_pretrained(
                    self.vae,
                    checkpoint / "vae",
                    remap=cosmos3_vae_weight_remap,
                    strict=args.weight_strict,
                )
                logger.info_rank0("Cosmos3 staged VAE: %s", report.summary())
        finally:
            set_engine_config(engine)
        self.scheduler = Cosmos3StagedScheduler(
            self.transformer,
            vae=self.vae,
            device=engine.device.target,
            use_cuda_graph=engine.runtime.use_cuda_graph,
            use_rollout_graph=args.use_rollout_graph,
            framework_compat=args.framework_compat,
            flow_shift=args.flow_shift,
            use_karras_sigmas=args.use_karras_sigmas,
            warmup_runs=args.warmup_runs,
            timing_runs=args.timing_runs,
        )
        self.scheduler.setup()

    def step(self, request: Cosmos3StageRequest):
        if self.scheduler is None:
            raise RuntimeError("Call setup() before executing a Cosmos3 stage")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.vae = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {} if self.transformer is None else {"transformer": self.transformer}


__all__ = ["Cosmos3StagedArgs", "Cosmos3StagedEntry"]
