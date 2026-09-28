"""Explicit rank layouts for PhyAI's logical parallel domains."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, Mapping

from phyai.parallel.config import ParallelConfig, ResolvedParallelConfig


WORLD: Final[str] = "world"
GROUP_NAMES: Final[tuple[str, ...]] = (
    WORLD,
    "pipeline",
    "cfg",
    "dense_tp",
    "dense_dp",
    "attention_tp",
    "attention_cp",
    "attention_dp",
    "attention_decode_cp",
    "moe_tp",
    "moe_ep",
    "moe_dp",
    "moe_tp_ep",
)

Group = tuple[int, ...]
GroupMemberships = dict[str, tuple[Group, ...]]


@dataclass(frozen=True)
class RankLayout:
    """Resolved rank memberships for one replica."""

    replica_world_size: int
    groups: Mapping[str, tuple[Group, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groups",
            MappingProxyType(
                {
                    name: tuple(tuple(members) for members in groups)
                    for name, groups in self.groups.items()
                }
            ),
        )
        if (
            not isinstance(self.replica_world_size, int)
            or isinstance(self.replica_world_size, bool)
            or self.replica_world_size < 1
        ):
            raise ValueError("replica_world_size must be a positive int.")
        unknown = set(self.groups) - set(GROUP_NAMES)
        if unknown:
            raise ValueError(f"unknown parallel groups: {sorted(unknown)!r}.")
        world = self.groups.get(WORLD)
        if world != (tuple(range(self.replica_world_size)),):
            raise ValueError("world group must contain every replica rank once.")
        for name, groups in self.groups.items():
            for members in groups:
                if not members or any(
                    not isinstance(rank, int)
                    or isinstance(rank, bool)
                    or rank < 0
                    or rank >= self.replica_world_size
                    for rank in members
                ):
                    raise ValueError(
                        f"invalid members for group {name!r}: {members!r}."
                    )
                if len(set(members)) != len(members):
                    raise ValueError(f"group {name!r} contains duplicate ranks.")
                # torch.new_group sorts ranks, so logical rank order must agree.
                if members != tuple(sorted(members)):
                    raise ValueError(
                        f"group {name!r} members must be in ascending rank order."
                    )
            flattened = [rank for members in groups for rank in members]
            if sorted(flattened) != list(range(self.replica_world_size)):
                raise ValueError(
                    f"group family {name!r} must partition every replica rank exactly once."
                )

    def __reduce__(self):
        return type(self), (self.replica_world_size, dict(self.groups))

    def groups_for(self, name: str) -> tuple[Group, ...]:
        try:
            return self.groups[name]
        except KeyError as error:
            raise KeyError(
                f"unknown parallel group {name!r}; valid groups: {GROUP_NAMES!r}."
            ) from error

    def members_for(self, name: str, rank: int) -> Group:
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < self.replica_world_size
        ):
            raise ValueError(f"rank {rank} is outside [0, {self.replica_world_size}).")
        for members in self.groups_for(name):
            if rank in members:
                return members
        raise ValueError(f"rank {rank} does not belong to group {name!r}.")

    def group_rank(self, name: str, rank: int) -> int:
        return self.members_for(name, rank).index(rank)


def build_rank_layout(
    parallel: ParallelConfig | ResolvedParallelConfig,
    replica_world_size: int | None = None,
) -> RankLayout:
    """Build all deterministic group memberships for one replica."""
    if isinstance(parallel, ParallelConfig):
        if replica_world_size is None:
            replica_world_size = parallel.infer_replica_world_size()
        resolved = parallel.resolve(replica_world_size)
    elif isinstance(parallel, ResolvedParallelConfig):
        resolved = parallel
        if (
            replica_world_size is not None
            and replica_world_size != resolved.replica_world_size
        ):
            raise ValueError(
                "resolved parallel config world does not match the requested world: "
                f"{resolved.replica_world_size} != {replica_world_size}."
            )
    else:
        raise TypeError(
            "build_rank_layout expects a ParallelConfig or ResolvedParallelConfig."
        )

    world = resolved.replica_world_size
    pipeline_size = resolved.outer.pipeline_size
    cfg_size = resolved.outer.cfg_size
    scope_size = world // (pipeline_size * cfg_size)
    groups: GroupMemberships = {WORLD: (tuple(range(world)),)}

    groups["pipeline"] = tuple(
        tuple(
            (pipeline_rank * cfg_size + cfg_rank) * scope_size + local_rank
            for pipeline_rank in range(pipeline_size)
        )
        for cfg_rank in range(cfg_size)
        for local_rank in range(scope_size)
    )

    groups["cfg"] = tuple(
        tuple(
            (pipeline_rank * cfg_size + cfg_rank) * scope_size + local_rank
            for cfg_rank in range(cfg_size)
        )
        for pipeline_rank in range(pipeline_size)
        for local_rank in range(scope_size)
    )

    def add_domain(
        name: str, dimensions: tuple[tuple[str, int], ...], target: int
    ) -> None:
        groups[name] = tuple(
            member_group
            for pipeline_rank in range(pipeline_size)
            for cfg_rank in range(cfg_size)
            for member_group in _dimension_groups(
                base=(pipeline_rank * cfg_size + cfg_rank) * scope_size,
                dimensions=dimensions,
                target=target,
            )
        )

    dense_dims = (("tp", resolved.dense.tp_size), ("dp", resolved.dense.dp_size))
    attention_dims = (
        ("tp", resolved.attention.tp_size),
        ("cp", resolved.attention.cp_size),
        ("dp", resolved.attention.dp_size),
    )
    moe_dims = (
        ("tp", resolved.moe.tp_size),
        ("ep", resolved.moe.ep_size),
        ("dp", resolved.moe.dp_size),
    )
    add_domain("dense_tp", dense_dims, 0)
    add_domain("dense_dp", dense_dims, 1)
    add_domain("attention_tp", attention_dims, 0)
    add_domain("attention_cp", attention_dims, 1)
    add_domain("attention_dp", attention_dims, 2)
    add_domain("moe_tp", moe_dims, 0)
    add_domain("moe_ep", moe_dims, 1)
    add_domain("moe_dp", moe_dims, 2)
    add_domain(
        "moe_tp_ep",
        (
            ("tp_ep", resolved.moe.tp_size * resolved.moe.ep_size),
            ("dp", resolved.moe.dp_size),
        ),
        0,
    )
    groups["attention_decode_cp"] = _partition_groups(
        groups["attention_tp"], resolved.attention.decode_cp_size
    )
    return RankLayout(world, groups)


def _dimension_groups(
    *, base: int, dimensions: tuple[tuple[str, int], ...], target: int
) -> tuple[Group, ...]:
    """Return groups for dimensions whose first item is fastest-changing."""
    sizes = tuple(size for _, size in dimensions)
    groups: list[Group] = []
    if not 0 <= target < len(dimensions):
        raise ValueError(f"group dimension index {target} is out of range.")
    target_size = sizes[target]
    fixed_ranges = [range(size) for index, size in enumerate(sizes) if index != target]
    for fixed in product(*fixed_ranges):
        fixed_iter = iter(fixed)
        coords = [
            0 if index == target else next(fixed_iter) for index in range(len(sizes))
        ]
        members: list[int] = []
        for value in range(target_size):
            coords[target] = value
            local = 0
            stride = 1
            for coordinate, size in zip(coords, sizes, strict=True):
                local += coordinate * stride
                stride *= size
            members.append(base + local)
        groups.append(tuple(members))
    return tuple(groups)


def _partition_groups(groups: tuple[Group, ...], size: int) -> tuple[Group, ...]:
    out: list[Group] = []
    for members in groups:
        if len(members) % size:
            raise ValueError(f"cannot partition group {members!r} by size {size}.")
        out.extend(
            members[start : start + size] for start in range(0, len(members), size)
        )
    return tuple(out)


def memberships(layout: RankLayout, rank: int) -> tuple[tuple[str, int], ...]:
    """Return ``(group, rank_in_group)`` for every group containing ``rank``."""
    return tuple((name, layout.group_rank(name, rank)) for name in layout.groups)


__all__ = [
    "GROUP_NAMES",
    "Group",
    "GroupMemberships",
    "RankLayout",
    "WORLD",
    "build_rank_layout",
    "memberships",
]
