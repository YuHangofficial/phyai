"""Rank translation helpers for torch.distributed process groups."""

from __future__ import annotations

import torch.distributed as dist


def group_rank_to_global(pg: dist.ProcessGroup, rank: int) -> int:
    """Translate a group-local rank to the corresponding global rank."""
    world_size = dist.get_world_size(group=pg)
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError(f"group rank must be an int, got {rank!r}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"group rank {rank} is outside [0, {world_size}) for this process group."
        )
    return dist.get_global_rank(pg, rank)


__all__ = ["group_rank_to_global"]
