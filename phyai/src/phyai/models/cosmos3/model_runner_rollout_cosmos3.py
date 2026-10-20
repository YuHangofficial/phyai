"""Exact native UniPC constants and runner-owned Cosmos3 rollout graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import torch

from phyai.kernel.bootstrap import get_kernel_selector
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Condition
from phyai.models.cosmos3.model_runner_graph_cosmos3 import Cosmos3CudaGraphRunner
from phyai.models.cosmos3.model_runner_staged_cosmos3 import (
    Cosmos3DiTRunner,
    Cosmos3VLMRunner,
)
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class UniPCUpdate:
    sigma_ratio: torch.Tensor
    alpha_phi: torch.Tensor
    alpha_b: torch.Tensor
    rks: tuple[torch.Tensor, ...]
    rhos: torch.Tensor | None


@dataclass(frozen=True)
class UniPCStep:
    sigma: torch.Tensor
    predictor: UniPCUpdate
    corrector: UniPCUpdate | None


@dataclass(frozen=True)
class Cosmos3UniPCSchedule:
    timesteps: torch.Tensor
    steps: tuple[UniPCStep, ...]
    framework_compat: bool = False


def native_update_coefficients(
    sampler: UniPCMultistepSampler,
    index: int,
    order: int,
    *,
    corrector: bool,
    device: torch.device,
    dtype: torch.dtype,
    scalar_device: torch.device | None = None,
) -> UniPCUpdate:
    """Evaluate native scalar arithmetic on its original device, once per schedule."""
    sigmas = sampler.sigmas
    scalar_device = device if scalar_device is None else scalar_device
    target, source = (index, index - 1) if corrector else (index + 1, index)
    sigma_t, sigma_s0 = (
        sigmas[target].to(scalar_device),
        sigmas[source].to(scalar_device),
    )
    alpha_t, sigma_t = sampler._alpha_sigma(sigma_t)
    alpha_s0, sigma_s0 = sampler._alpha_sigma(sigma_s0)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0
    rks = []
    for offset in range(1, order):
        previous = index - offset - 1 if corrector else index - offset
        alpha_si, sigma_si = sampler._alpha_sigma(sigmas[previous].to(scalar_device))
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rks.append((lambda_si - lambda_s0) / h)
    hh = -h
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)
    if corrector and order > 1:
        all_rks = torch.stack([*rks, torch.tensor(1.0, device=scalar_device)]).to(
            device
        )
        h_phi_k = h_phi_1 / hh - 1
        rows, rhs = [], []
        factorial_i = 1
        for row in range(1, order + 1):
            rows.append(torch.pow(all_rks, row - 1))
            rhs.append(h_phi_k * factorial_i / b_h)
            factorial_i *= row + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        rhos = torch.linalg.solve(torch.stack(rows), torch.stack(rhs).to(device)).to(
            device=device, dtype=dtype
        )
    elif corrector or order > 1:
        rhos = torch.tensor([0.5], dtype=dtype, device=device)
    else:
        rhos = None
    return UniPCUpdate(
        sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * b_h, tuple(rks), rhos
    )


def build_native_unipc_schedule(
    *,
    num_steps: int,
    flow_shift: float,
    use_karras_sigmas: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    framework_compat: bool = False,
) -> Cosmos3UniPCSchedule:
    """Use the native schedule and preserve order-2 predictor/corrector rounding."""
    if framework_compat and use_karras_sigmas:
        raise ValueError(
            "Framework-compatible staged sampling supports the linear flow schedule"
        )
    sampler = UniPCMultistepSampler(
        flow_shift=flow_shift,
        use_karras_sigmas=use_karras_sigmas,
        base_shift=1.0 if framework_compat else None,
    )
    sampler.set_timesteps(num_steps, device=device)
    steps = []
    previous_order = 0
    for index in range(num_steps):
        corrector = (
            native_update_coefficients(
                sampler,
                index,
                previous_order,
                corrector=True,
                device=device,
                dtype=dtype,
                scalar_device=torch.device("cpu") if framework_compat else device,
            )
            if index
            else None
        )
        order = min(sampler.solver_order, num_steps - index, index + 1)
        predictor = native_update_coefficients(
            sampler,
            index,
            order,
            corrector=False,
            device=device,
            dtype=dtype,
            scalar_device=torch.device("cpu") if framework_compat else device,
        )
        steps.append(
            UniPCStep(
                sampler.sigmas[index].to(
                    device="cpu" if framework_compat else device, dtype=dtype
                ),
                predictor,
                corrector,
            )
        )
        previous_order = order
    return Cosmos3UniPCSchedule(sampler.timesteps, tuple(steps), framework_compat)


class Cosmos3UniPCState:
    """Fresh per-rollout history; only immutable schedule tensors are reused."""

    def __init__(self, schedule: Cosmos3UniPCSchedule) -> None:
        self.schedule = schedule
        self.outputs: list[torch.Tensor | None] = [None, None]
        self.last_sample: torch.Tensor | None = None
        self.index = 0

    def step(self, velocity: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        coefficients = self.schedule.steps[self.index]
        converted = sample - coefficients.sigma * velocity
        m0 = self.outputs[-1]
        corrector = coefficients.corrector
        if corrector is not None:
            differences = [
                (self.outputs[-offset - 2] - m0) / rk
                for offset, rk in enumerate(corrector.rks)
            ]
            stacked = torch.stack(differences, dim=1) if differences else None
            x_t = corrector.sigma_ratio * self.last_sample - corrector.alpha_phi * m0
            corr_res = (
                torch.einsum("k,bkc...->bc...", corrector.rhos[:-1], stacked)
                if stacked is not None
                else 0.0
            )
            delta = converted - m0
            sample = (
                x_t - corrector.alpha_b * (corr_res + corrector.rhos[-1] * delta)
            ).to(sample.dtype)
        self.outputs = [self.outputs[-1], converted]
        self.last_sample = sample
        predictor = coefficients.predictor
        differences = [
            (self.outputs[-offset - 2] - converted) / rk
            for offset, rk in enumerate(predictor.rks)
        ]
        x_t = predictor.sigma_ratio * sample - predictor.alpha_phi * converted
        if differences:
            prediction = torch.einsum(
                "k,bkc...->bc...", predictor.rhos, torch.stack(differences, dim=1)
            )
            x_t = x_t - predictor.alpha_b * prediction
        elif self.schedule.framework_compat:
            x_t = x_t - predictor.alpha_b * 0
        self.index += 1
        return x_t.to(sample.dtype)


@dataclass(frozen=True)
class Cosmos3RolloutPlan:
    key: Hashable
    condition: Cosmos3Condition
    schedule: Cosmos3UniPCSchedule
    timesteps: tuple[torch.Tensor, ...]
    time_embeddings: tuple[torch.Tensor, ...]
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]


def tensor_revision(tensor: torch.Tensor) -> tuple:
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        version,
    )


class Cosmos3RolloutRunner(ModelRunner):
    """Bounded immutable plans and one graph containing every native denoise step.

    Fresh initial noise and conditioning values enter every replay. No DiT
    activations or velocity history are reused between requests. Call close()
    before offloading weights, changing untracked inference weights, or prefill.
    """

    def __init__(self, dit: Cosmos3DiTRunner, vlm: Cosmos3VLMRunner) -> None:
        self.dit = dit
        self.vlm = vlm
        self.graph = Cosmos3CudaGraphRunner()
        self.plans: OrderedDict[Hashable, Cosmos3RolloutPlan] = OrderedDict()
        self.schedules: OrderedDict[Hashable, Cosmos3UniPCSchedule] = OrderedDict()
        self.plan_builds = 0
        self.plan_hits = 0
        self.schedule_builds = 0
        self.time_embedding_calls = 0

    def setup(self) -> None:
        pass

    @torch.no_grad()
    def prepare(
        self,
        positions: torch.Tensor,
        shapes: tuple[tuple[int, ...], ...],
        *,
        num_vision: int,
        num_steps: int,
        flow_shift: float,
        use_karras_sigmas: bool,
        framework_compat: bool = False,
        action_len: int | None = None,
    ) -> Cosmos3RolloutPlan:
        model = self.dit.transformer
        device, dtype = model.proj_in.weight.device, model.proj_in.weight.dtype
        schedule_key = (
            num_steps,
            flow_shift,
            use_karras_sigmas,
            device,
            torch.get_default_dtype(),
            framework_compat,
        )
        # Position values, not tensor identities, determine the RoPE table. This
        # also detects in-place edits to tensors created under inference_mode.
        position_key = (
            positions.dtype,
            tuple(positions.shape),
            tuple(positions.detach().cpu().reshape(-1).tolist()),
        )
        weights = tuple(
            tensor_revision(p)
            for name, p in model.named_parameters()
            if not name.startswith("language_model.")
        )
        kv_key = tuple(tensor_revision(t) for pair in self.vlm.cached_kv for t in pair)
        selector = get_kernel_selector()
        key = (
            schedule_key,
            position_key,
            shapes,
            num_vision,
            action_len,
            dtype,
            weights,
            kv_key,
            repr(model.config),
            (selector, selector.epoch),
        )
        if key in self.plans:
            self.plans.move_to_end(key)
            self.plan_hits += 1
            return self.plans[key]
        schedule = self.schedules.get(schedule_key)
        if schedule is None:
            schedule = build_native_unipc_schedule(
                num_steps=num_steps,
                flow_shift=flow_shift,
                use_karras_sigmas=use_karras_sigmas,
                device=device,
                framework_compat=framework_compat,
            )
            self.schedules[schedule_key] = schedule
            self.schedule_builds += 1
            if len(self.schedules) > 4:
                self.schedules.popitem(last=False)
        else:
            self.schedules.move_to_end(schedule_key)
        if action_len is None:
            action_len = shapes[-1][1] if len(shapes) > num_vision else 0
        self.dit.prepare(
            self.vlm.cached_kv,
            self.vlm.frequencies(positions),
            tuple(shape[2:] for shape in shapes[:num_vision]),
            action_len,
        )
        condition = self.dit.rollout_condition()
        timesteps = tuple(
            t.to(device, torch.float32 if framework_compat else dtype)
            .reshape(1)
            .expand(shapes[0][0])
            for t in schedule.timesteps
        )
        embeddings = tuple(
            model.time_embedder(t * model.timestep_scale) for t in timesteps
        )
        self.time_embedding_calls += len(embeddings)
        sizes = tuple(torch.Size(shape[1:]).numel() for shape in shapes)
        plan = Cosmos3RolloutPlan(
            key, condition, schedule, timesteps, embeddings, shapes, sizes
        )
        self.plans[key] = plan
        self.plan_builds += 1
        if len(self.plans) > 2:
            self.plans.popitem(last=False)
        return plan

    @torch.no_grad()
    def forward(
        self,
        function: Callable[..., torch.Tensor],
        inputs: dict[str, torch.Tensor],
        *,
        plan: Cosmos3RolloutPlan,
    ) -> torch.Tensor:
        return self.graph.forward(function, inputs, graph_key=plan.key).clone()

    def close(self) -> None:
        self.graph.reset()
        self.plans.clear()
        self.schedules.clear()


__all__ = [
    "Cosmos3RolloutRunner",
    "Cosmos3RolloutPlan",
    "Cosmos3UniPCState",
    "build_native_unipc_schedule",
]
