"""Logical parallel domains of one model replica.

The physical replica world (the launcher's rank count) is not part of this
configuration. Each domain describes how it factorizes the rank pool that
remains after the outer pipeline and CFG dimensions; :meth:`ParallelConfig.resolve`
binds the domains to an actual world and fills in omitted TP sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm
from typing import Final


OUTER_DOMAINS: Final[tuple[str, ...]] = ("pipeline", "cfg")
MODEL_DOMAINS: Final[tuple[str, ...]] = ("dense", "attention", "moe")
DOMAINS: Final[tuple[str, ...]] = OUTER_DOMAINS + MODEL_DOMAINS


def _validate_parallel_sizes(
    obj: object,
    names: tuple[str, ...],
    *,
    allow_none: tuple[str, ...] = (),
) -> None:
    for name in names:
        value = getattr(obj, name)
        if value is None and name in allow_none:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"{type(obj).__name__}.{name} must be a positive int, got {value!r}."
            )


@dataclass(frozen=True)
class OuterParallelConfig:
    """Physical dimensions shared by every model-parallel domain."""

    pipeline_size: int = 1
    cfg_size: int = 1

    def __post_init__(self) -> None:
        _validate_parallel_sizes(self, ("pipeline_size", "cfg_size"))


@dataclass(frozen=True)
class DenseParallelConfig:
    """Dense-layer tensor and data parallelism."""

    tp_size: int | None = None
    dp_size: int = 1
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        _validate_parallel_sizes(self, ("tp_size", "dp_size"), allow_none=("tp_size",))
        if not isinstance(self.sequence_parallel, bool):
            raise ValueError("DenseParallelConfig.sequence_parallel must be a bool.")

    @property
    def is_tp_only(self) -> bool:
        """True when the domain uses nothing beyond tensor parallelism."""
        return self.dp_size == 1 and not self.sequence_parallel


@dataclass(frozen=True)
class AttentionParallelConfig:
    """Attention TP/CP/DP, with optional decode CP inside each TP group."""

    tp_size: int | None = None
    cp_size: int = 1
    dp_size: int = 1
    decode_cp_size: int = 1

    def __post_init__(self) -> None:
        _validate_parallel_sizes(
            self,
            ("tp_size", "cp_size", "dp_size", "decode_cp_size"),
            allow_none=("tp_size",),
        )

    @property
    def is_tp_only(self) -> bool:
        """True when the domain uses nothing beyond tensor parallelism."""
        return self.cp_size == 1 and self.dp_size == 1 and self.decode_cp_size == 1


@dataclass(frozen=True)
class MoeParallelConfig:
    """Mixture-of-Experts tensor, expert and data parallelism."""

    tp_size: int | None = None
    ep_size: int = 1
    dp_size: int = 1

    def __post_init__(self) -> None:
        _validate_parallel_sizes(
            self, ("tp_size", "ep_size", "dp_size"), allow_none=("tp_size",)
        )

    @property
    def is_tp_only(self) -> bool:
        """True when the domain uses nothing beyond tensor parallelism."""
        return self.ep_size == 1 and self.dp_size == 1


@dataclass(frozen=True)
class ParallelConfig:
    """Logical parallel domains for one serving replica.

    The physical replica world is supplied by deployment or torch.distributed;
    it is deliberately not part of this configuration. Domain dimensions are
    resolved against that runtime world by :meth:`resolve`.
    """

    outer: OuterParallelConfig = field(default_factory=OuterParallelConfig)
    dense: DenseParallelConfig = field(default_factory=DenseParallelConfig)
    attention: AttentionParallelConfig = field(default_factory=AttentionParallelConfig)
    moe: MoeParallelConfig = field(default_factory=MoeParallelConfig)

    def __post_init__(self) -> None:
        for name, expected in (
            ("outer", OuterParallelConfig),
            ("dense", DenseParallelConfig),
            ("attention", AttentionParallelConfig),
            ("moe", MoeParallelConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"ParallelConfig.{name} must be a {expected.__name__}.")

    def infer_replica_world_size(self) -> int:
        """Infer the smallest physical world implied by explicit dimensions.

        The launcher or deployment remains authoritative. This helper is used
        only before a process group exists, for example when selecting the
        local executor mode.
        """
        outer_size = self.outer.pipeline_size * self.outer.cfg_size
        dimensions = (
            (self.dense.tp_size, self.dense.dp_size),
            (self.attention.tp_size, self.attention.cp_size * self.attention.dp_size),
            (self.moe.tp_size, self.moe.ep_size * self.moe.dp_size),
        )
        explicit = {tp * other for tp, other in dimensions if tp is not None}
        if len(explicit) > 1:
            raise ValueError(
                "parallel domains imply different model-scope sizes: "
                f"{sorted(explicit)!r}. Set compatible domain dimensions."
            )
        minimum = lcm(
            *(other for _, other in dimensions),
            self.attention.cp_size
            * self.attention.dp_size
            * self.attention.decode_cp_size,
        )
        scope_size = next(iter(explicit)) if explicit else minimum
        world = outer_size * scope_size
        self.resolve(world)
        return world

    def resolve(self, replica_world_size: int) -> "ResolvedParallelConfig":
        """Resolve all omitted domain sizes against the actual replica world."""
        if (
            not isinstance(replica_world_size, int)
            or isinstance(replica_world_size, bool)
            or replica_world_size < 1
        ):
            raise ValueError(
                "replica_world_size must be a positive int, "
                f"got {replica_world_size!r}."
            )
        outer_size = self.outer.pipeline_size * self.outer.cfg_size
        if replica_world_size % outer_size:
            raise ValueError(
                "replica_world_size must be divisible by the outer parallel "
                f"size {outer_size}; got {replica_world_size}."
            )
        scope_size = replica_world_size // outer_size

        dense_tp = _resolve_dimension(
            scope_size, self.dense.tp_size, self.dense.dp_size
        )
        attention_tp = _resolve_dimension(
            scope_size,
            self.attention.tp_size,
            self.attention.cp_size,
            self.attention.dp_size,
        )
        moe_tp = _resolve_dimension(
            scope_size,
            self.moe.tp_size,
            self.moe.ep_size,
            self.moe.dp_size,
        )

        if scope_size != dense_tp * self.dense.dp_size:
            raise ValueError(
                "dense parallel sizes must cover the model scope: "
                f"tp={dense_tp}, dp={self.dense.dp_size}, scope={scope_size}."
            )
        if scope_size != attention_tp * self.attention.cp_size * self.attention.dp_size:
            raise ValueError(
                "attention parallel sizes must cover the model scope: "
                f"tp={attention_tp}, cp={self.attention.cp_size}, "
                f"dp={self.attention.dp_size}, scope={scope_size}."
            )
        if scope_size != moe_tp * self.moe.ep_size * self.moe.dp_size:
            raise ValueError(
                "MoE parallel sizes must cover the model scope: "
                f"tp={moe_tp}, ep={self.moe.ep_size}, "
                f"dp={self.moe.dp_size}, scope={scope_size}."
            )
        if attention_tp % self.attention.decode_cp_size:
            raise ValueError(
                f"attention.decode_cp_size={self.attention.decode_cp_size} "
                f"must divide attention tp size {attention_tp}."
            )

        return ResolvedParallelConfig(
            replica_world_size=replica_world_size,
            outer=self.outer,
            dense=DenseParallelConfig(
                tp_size=dense_tp,
                dp_size=self.dense.dp_size,
                sequence_parallel=self.dense.sequence_parallel,
            ),
            attention=AttentionParallelConfig(
                tp_size=attention_tp,
                cp_size=self.attention.cp_size,
                dp_size=self.attention.dp_size,
                decode_cp_size=self.attention.decode_cp_size,
            ),
            moe=MoeParallelConfig(
                tp_size=moe_tp,
                ep_size=self.moe.ep_size,
                dp_size=self.moe.dp_size,
            ),
        )


def _resolve_dimension(scope_size: int, tp_size: int | None, *others: int) -> int:
    """Return the TP size, deriving it from the scope when it was omitted."""
    if tp_size is not None:
        return tp_size
    product = 1
    for size in others:
        product *= size
    if scope_size % product:
        raise ValueError(
            f"parallel dimensions {others!r} do not divide scope {scope_size}."
        )
    return scope_size // product


@dataclass(frozen=True)
class ResolvedParallelConfig:
    """Parallel configuration after binding it to one replica's rank pool."""

    replica_world_size: int
    outer: OuterParallelConfig
    dense: DenseParallelConfig
    attention: AttentionParallelConfig
    moe: MoeParallelConfig

    @property
    def scope_size(self) -> int:
        """Ranks of one pipeline/CFG partition, shared by every model domain."""
        return self.replica_world_size // (
            self.outer.pipeline_size * self.outer.cfg_size
        )

    def domain(
        self, name: str
    ) -> DenseParallelConfig | AttentionParallelConfig | MoeParallelConfig:
        """Return the resolved config of one model domain by name."""
        if name not in MODEL_DOMAINS:
            raise KeyError(f"unknown model domain {name!r}; valid: {MODEL_DOMAINS!r}.")
        return getattr(self, name)


__all__ = [
    "AttentionParallelConfig",
    "DOMAINS",
    "DenseParallelConfig",
    "MODEL_DOMAINS",
    "MoeParallelConfig",
    "OUTER_DOMAINS",
    "OuterParallelConfig",
    "ParallelConfig",
    "ResolvedParallelConfig",
]
