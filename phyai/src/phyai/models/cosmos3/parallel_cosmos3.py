"""Parallel-domain contract for one Cosmos3 model replica."""

from __future__ import annotations

from phyai.engine_config import ParallelConfig
from phyai.parallel import default_mesh


def group_rank_size(group: str) -> tuple[int, int]:
    """Return the local rank and size for a Cosmos3 communication group.

    EngineCore initializes the default mesh before constructing a scheduler.
    Let lookup errors propagate so a malformed multi-rank deployment cannot
    silently execute a different (single-rank) algorithm.
    """
    mesh = default_mesh()
    return mesh.group_rank(group), mesh.group_size(group)


def validate_cosmos3_parallel(
    parallel: ParallelConfig, replica_world_size: int | None = None
) -> None:
    """Reject parallel sizes Cosmos3 does not implement.

    Cosmos3 supports CFG parallelism (cfg_size 1 or 2) and dense tensor
    parallelism; the tiled VAE decode runs on the ``world`` group.
    """
    if parallel.outer.cfg_size not in (1, 2):
        raise ValueError(
            "Cosmos3 CFG parallelism supports cfg_size=1 or 2, got "
            f"{parallel.outer.cfg_size}."
        )
    resolved = parallel.resolve(
        parallel.infer_replica_world_size()
        if replica_world_size is None
        else replica_world_size
    )
    if (
        resolved.dense.dp_size != 1
        or resolved.dense.sequence_parallel
        or resolved.attention.cp_size != 1
        or resolved.attention.dp_size != 1
        or resolved.attention.decode_cp_size != 1
        or resolved.attention.tp_size != resolved.dense.tp_size
    ):
        raise ValueError(
            "Cosmos3 requires matching dense/attention TP groups, dense DP=1, "
            "attention DP/CP=1, and sequence_parallel=False."
        )


__all__ = ["group_rank_size", "validate_cosmos3_parallel"]
