"""Deterministic creation and ownership of shared torch process groups."""

from __future__ import annotations

import torch.distributed as dist

from phyai.parallel.layout import RankLayout


class ProcessGroupPool:
    """Share handles by ordered membership and backend within one mesh.

    Logical groups with identical members share one ``ProcessGroup`` (and,
    through ``PyNCCLBackend``, one NCCL communicator), so the communicator
    count follows the number of distinct memberships rather than the number
    of group names. Sharing is safe because PhyAI issues collectives
    sequentially on one stream. A feature that needs concurrent collectives
    over the same ranks, an EPLB-style side channel for example, must create
    its own group instead of borrowing an alias.

    Every rank walks the same ordered layout and calls ``new_group`` for
    every membership, members and nonmembers alike, as torch requires. Only
    groups this rank belongs to are kept and later destroyed here; torch's
    default group belongs to the launcher.
    """

    def __init__(self, layout: RankLayout, *, backend: str, build_cpu: bool) -> None:
        self._groups: dict[tuple[str, tuple[int, ...]], dist.ProcessGroup] = {}
        self._owned: list[dist.ProcessGroup] = []
        backends = tuple(dict.fromkeys((backend, "gloo") if build_cpu else (backend,)))
        world_ranks = tuple(range(layout.replica_world_size))
        # A default group created without an explicit backend is composite and
        # reports "undefined"; it serves every device type, so it stands in for
        # each requested backend instead of being duplicated.
        world_backend = str(dist.get_backend())
        world_aliases = (
            backends if world_backend == dist.Backend.UNDEFINED else (world_backend,)
        )
        for name in world_aliases:
            self._groups[(name, world_ranks)] = dist.group.WORLD
        try:
            for memberships in layout.groups.values():
                for ranks in memberships:
                    if len(ranks) <= 1:
                        continue
                    for name in backends:
                        key = (name, ranks)
                        if key in self._groups:
                            continue
                        group = dist.new_group(ranks=list(ranks), backend=name)
                        self._groups[key] = group
                        if dist.get_rank() in ranks:
                            self._owned.append(group)
        except BaseException:
            self.close()
            raise

    def get(self, ranks: tuple[int, ...], backend: str) -> dist.ProcessGroup:
        return self._groups[(backend, ranks)]

    def close(self) -> None:
        """Release only the groups created here; the launcher owns WORLD."""
        error: BaseException | None = None
        while self._owned:
            group = self._owned.pop()
            if dist.is_initialized():
                try:
                    dist.destroy_process_group(group)
                except BaseException as caught:
                    if error is None:
                        error = caught
        self._groups.clear()
        if error is not None:
            raise error


__all__ = ["ProcessGroupPool"]
