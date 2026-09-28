"""Write-side process environment setup: env vars, rlimits, crash handlers.

:mod:`phyai.env` is the *read* side — a typed registry of ``PHYAI_*``
variables phyai consults. This module is the *write* side: the handful of
third-party variables (CUDA, flashinfer, OpenMP) that want a
phyai-appropriate value before any kernel launches, plus the two
process-level knobs that have nothing to do with torch (file-descriptor
limits and a fault handler).

The two sides stay in separate modules on purpose. A reader looking for
"what does phyai consult?" should not have to skip over code that mutates
the environment out from under it.

Contract
--------
:func:`init_env` never overwrites a variable the caller already set. A
launch script, a Slurm prologue or a ``docker run -e`` always wins — the
tuned values here are defaults for the common case, not policy. Every
value actually written is returned *and* logged, so a run's environment
is reconstructable from its log. ``PHYAI_SKIP_ENV_SETUP=1`` disables the
whole thing.

Why not just document the values and let users export them? Because the
failure mode of a missing ``CUDA_DEVICE_MAX_CONNECTIONS`` is not an
error, it is a quietly serialized stream — the kind of thing that shows
up as a 20% throughput mystery three months later.
"""

from __future__ import annotations

import os
import resource
import faulthandler
from typing import Callable
from dataclasses import dataclass

import setproctitle

from phyai.env import envs
from phyai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TunedVar:
    """One third-party env var phyai wants a specific value for.

    Fields
    ------
    name / value:
        What to export. ``value`` is always a string — this is
        ``os.environ``, not a config object.
    why:
        One line explaining what breaks or degrades without it. Written
        into the log line when the variable is applied, so the reason
        travels with the run rather than living only here.
    applies_when:
        Predicate over ``(world_size, device_type)``. A variable whose
        predicate is false is skipped entirely — it stays in this table
        as documentation of the recommended value without being imposed.
    """

    name: str
    value: str
    why: str
    applies_when: Callable[[int, str], bool]


def _cuda_only(world_size: int, device_type: str) -> bool:
    del world_size
    return device_type == "cuda"


def _never(world_size: int, device_type: str) -> bool:
    """Recommended, but not applied until measured on this hardware."""
    del world_size, device_type
    return False


#: Third-party variables phyai sets when they are unset.
#:
#: Ordering is documentation-only; each entry is independent.
TUNED_ENV_VARS: tuple[TunedVar, ...] = (
    TunedVar(
        name="CUDA_DEVICE_MAX_CONNECTIONS",
        value="8",
        why=(
            "hardware work queues per device; a low value serializes streams "
            "that should overlap (green-context multi-replica, CFG-parallel)"
        ),
        applies_when=_cuda_only,
    ),
    TunedVar(
        name="CUDA_MODULE_LOADING",
        value="LAZY",
        why=(
            "load cubins on first use -- cuts startup time and device memory "
            "for the large flashinfer fatbins and phyai-kernel JIT modules"
        ),
        applies_when=_cuda_only,
    ),
    TunedVar(
        name="TRTLLM_ENABLE_PDL",
        value="1",
        why=(
            "flashinfer's trtllm-gen attention and quant kernels read this to "
            "enable programmatic dependent launch"
        ),
        applies_when=_cuda_only,
    ),
    TunedVar(
        name="CUTE_DSL_LOG_LEVEL",
        value="30",
        why="quiet the flashinfer cute-dsl path down to warnings",
        applies_when=_cuda_only,
    ),
    TunedVar(
        name="CUTE_DSL_LOG_TO_CONSOLE",
        value="1",
        why="cute-dsl ignores its log level unless console logging is on",
        applies_when=_cuda_only,
    ),
    TunedVar(
        name="OMP_NUM_THREADS",
        value="1",
        why=(
            "inherited by spawned children (server worker group, self-managed "
            "DP) so each GPU process does not open a core-count-sized OpenMP "
            "pool; this process is handled by torch.set_num_threads in "
            "phyai.utils.torch_setup.init_threads, which OMP_NUM_THREADS "
            "cannot affect once torch is imported"
        ),
        applies_when=_cuda_only,
    ),
    # -- Recommended but NOT applied. ------------------------------------ #
    # These three are what sglang sets (entrypoints/engine.py
    # _set_envs_and_config). Every one of them changes how NCCL allocates or
    # which algorithm it picks, and phyai runs collectives *inside* captured
    # CUDA graphs via PyNCCL -- a regime where NCCL_GRAPH_MIXING_SUPPORT and
    # the cuMem allocator interact with capture in version-dependent ways.
    # Copying sglang's values unmeasured would be guessing. They stay here
    # with their rationale so the A/B is a one-line change to applies_when.
    TunedVar(
        name="NCCL_CUMEM_ENABLE",
        value="0",
        why=(
            "RECOMMENDED-ONLY: avoids NCCL's cuMem-based buffers; measure "
            "against captured-graph PyNCCL collectives before enabling"
        ),
        applies_when=_never,
    ),
    TunedVar(
        name="NCCL_NVLS_ENABLE",
        value="0",
        why=(
            "RECOMMENDED-ONLY: disables NVLink SHARP; only a win on hardware "
            "without working NVLS -- measure per box"
        ),
        applies_when=_never,
    ),
    TunedVar(
        name="NCCL_GRAPH_MIXING_SUPPORT",
        value="0",
        why=(
            "RECOMMENDED-ONLY: helps symmetric kernels but relaxes an "
            "assumption phyai leans on (collectives inside captured graphs)"
        ),
        applies_when=_never,
    ),
    TunedVar(
        name="PYTORCH_CUDA_ALLOC_CONF",
        value="expandable_segments:True",
        why=(
            "RECOMMENDED-ONLY: cuts fragmentation, but expandable segments "
            "are VMM-backed and interact with CUDA-graph capture and custom "
            "all-reduce; measure before enabling"
        ),
        applies_when=_never,
    ),
)


def init_env(*, world_size: int, device_type: str) -> dict[str, str]:
    """Export phyai's tuned values for any :data:`TUNED_ENV_VARS` still unset.

    Args:
        world_size: total rank count for this run. Passed to each
            variable's ``applies_when`` so multi-rank-only tuning can
            stay out of single-process runs.
        device_type: ``"cuda"`` / ``"cpu"`` / ... — the *type* only, no
            index.

    Returns:
        The variables actually written, ``{name: value}``. Empty when
        ``PHYAI_SKIP_ENV_SETUP`` is set or every variable was already
        present.
    """
    if envs.PHYAI_SKIP_ENV_SETUP.get():
        logger.info_rank0(
            "PHYAI_SKIP_ENV_SETUP is set: leaving the process environment "
            "untouched. See phyai.utils.env_setup.TUNED_ENV_VARS for the "
            "values phyai would otherwise export."
        )
        return {}

    applied: dict[str, str] = {}
    for var in TUNED_ENV_VARS:
        if not var.applies_when(world_size, device_type):
            continue
        if var.name in os.environ:
            continue
        os.environ[var.name] = var.value
        applied[var.name] = var.value

    if applied:
        logger.info_rank0(
            "env setup applied %d var(s): %s",
            len(applied),
            ", ".join((f"{k}={v}" for k, v in applied.items())),
        )
        for name, value in applied.items():
            why = next(v.why for v in TUNED_ENV_VARS if v.name == name)
            logger.debug_rank0("  %s=%s -- %s", name, value, why)
    return applied


def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file and stack soft limits toward inference-friendly values.

    An inference process holds a lot of descriptors: safetensors shards,
    IPC sockets, NCCL's per-peer file handles, one profiler trace per
    rank. The default 1024 on many distros is enough right up until a
    multi-rank run fails to open a shard with a misleading error.

    Only *raises* limits, never lowers them, and treats every failure as
    a warning: a container may cap the hard limit below the target, which
    is the operator's decision and not a reason to refuse to start.
    """
    for resource_type, target, label in (
        (resource.RLIMIT_NOFILE, target_soft_limit, "RLIMIT_NOFILE"),
        (resource.RLIMIT_STACK, 1024 * target_soft_limit, "RLIMIT_STACK"),
    ):
        try:
            current_soft, current_hard = resource.getrlimit(resource_type)
        except (OSError, ValueError) as e:  # pragma: no cover - platform dependent
            logger.warning_rank0("cannot read %s: %s", label, e)
            continue
        if current_soft >= target:
            continue
        try:
            resource.setrlimit(resource_type, (target, current_hard))
        except (OSError, ValueError) as e:
            logger.warning_rank0(
                "cannot raise %s from %s to %s: %s", label, current_soft, target, e
            )
        else:
            logger.debug_rank0(
                "raised %s soft limit %s -> %s", label, current_soft, target
            )


def init_process_debug(*, title: str | None = None) -> None:
    """Enable fault handling and optionally set the process title."""
    faulthandler.enable()
    if title is not None:
        setproctitle.setproctitle(title)


__all__ = [
    "TUNED_ENV_VARS",
    "TunedVar",
    "init_env",
    "init_process_debug",
    "set_ulimit",
]
