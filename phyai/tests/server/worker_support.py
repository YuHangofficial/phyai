"""Importable spawn targets for worker-supervisor tests."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from phyai.server.deployment import WorkerPlacement


class FakeRuntime:
    def __init__(self, placement: WorkerPlacement, args: Mapping[str, Any]) -> None:
        self.placement = placement
        self.args = dict(args)
        if self.args.get("startup_fail_worker") == placement.worker_id:
            raise RuntimeError("requested startup failure")

    def metadata(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.placement.worker_id,
            "device_index": self.placement.device_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "rank": os.environ.get("RANK"),
            "world_size": os.environ.get("WORLD_SIZE"),
        }

    def execute(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            sleep_by_worker = payload.get("sleep_by_worker", {})
            if delay := sleep_by_worker.get(self.placement.worker_id):
                time.sleep(delay)
            if payload.get("fail_worker") == self.placement.worker_id:
                raise RuntimeError("requested execute failure")
            if payload.get("exit_worker") == self.placement.worker_id:
                os._exit(17)
            if payload.get("unpicklable_output") == self.placement.worker_id:
                return lambda: None
        return {"worker_id": self.placement.worker_id, "payload": payload}

    def close(self) -> None:
        if self.args.get("shutdown_fail_worker") == self.placement.worker_id:
            raise RuntimeError("requested shutdown failure")


def create_fake_runtime(
    placement: WorkerPlacement, args: Mapping[str, Any] | None
) -> FakeRuntime:
    return FakeRuntime(placement, args or {})


class GlooRuntime:
    def __init__(
        self, placement: WorkerPlacement, args: Mapping[str, Any] | None
    ) -> None:
        from datetime import timedelta

        import torch
        import torch.distributed as dist

        self.placement = placement
        self.torch = torch
        self.dist = dist
        self.args = dict(args or {})
        self.parallel = None
        self.warmed: tuple[str, ...] = ()
        dist.init_process_group("gloo", timeout=timedelta(seconds=20))
        if args and args.get("init_mesh"):
            import phyai.parallel as parallel

            self.parallel = parallel
            parallel.init(
                args["parallel"],
                device="cpu",
                backend="gloo",
                enable_pynccl=False,
            )
            # Exercises the gloo warmup path on hosts that also have CUDA:
            # the warmup tensors must follow the group's backend (CPU here).
            self.warmed = parallel.warmup_collectives()

    def metadata(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.placement.worker_id,
            "rank": self.dist.get_rank(),
            "world_size": self.dist.get_world_size(),
            "warmed": self.warmed,
        }

    def execute(self, payload: Any) -> Any:
        if self.args.get("check_domains"):
            return self._check_domains()
        cfg_source = None
        if self.parallel is not None:
            cfg_value = self.parallel.broadcast(
                self.torch.tensor(float(self.dist.get_rank())),
                group="cfg",
                src=0,
            )
            cfg_source = cfg_value.item()
            expected = float(self.placement.group_rank("dense_tp"))
            if cfg_source != expected:
                raise RuntimeError(
                    f"CFG broadcast returned {cfg_source}, expected {expected}."
                )
        value = self.torch.tensor(float(payload) + self.dist.get_rank())
        self.dist.all_reduce(value)
        return {
            "rank": self.dist.get_rank(),
            "sum": value.item(),
            "cfg_source": cfg_source,
        }

    def _check_domains(self) -> dict[str, Any]:
        parallel = self.parallel
        rank = self.dist.get_rank()
        offset = 100 * self.placement.replica_id
        for _ in range(2):
            mesh = parallel.default_mesh()
            assert mesh.group("dense_tp") is self.dist.group.WORLD
            assert mesh.group("moe_tp_ep") is mesh.group("dense_tp")
            assert mesh.group("attention_dp") is mesh.group("moe_ep")
            for group in mesh.group_names:
                members = mesh.group_members(group)
                value = self.torch.tensor([float(offset + rank)])
                reduced = parallel.all_reduce(value, group=group)
                assert reduced.item() == sum(offset + member for member in members)
                gathered = parallel.all_gather(value, group=group)
                assert gathered.tolist() == [offset + member for member in members]
                source = parallel.broadcast(value, group=group, src=len(members) - 1)
                assert source.item() == offset + members[-1]
            parallel.shutdown()
            assert self.dist.is_initialized()
            assert len(self.dist.distributed_c10d._world.pg_map) == 1
            self.dist.barrier()
            parallel.init(
                self.args["parallel"], device="cpu", backend="gloo", enable_pynccl=False
            )
        return {
            "replica_id": self.placement.replica_id,
            "world": self.dist.get_world_size(),
        }

    def close(self) -> None:
        if self.parallel is not None:
            self.parallel.shutdown()
        if self.dist.is_initialized():
            self.dist.destroy_process_group()


def create_gloo_runtime(
    placement: WorkerPlacement, args: Mapping[str, Any] | None
) -> GlooRuntime:
    return GlooRuntime(placement, args)


class FakeCore:
    """Stand-in for EngineCore inside managed workers (no model, no CUDA)."""

    def __init__(self, engine_args: Any) -> None:
        self.args = engine_args
        self.config = engine_args.config

    def step(self, payload: Any) -> Any:
        return {"echo": payload, "plugin": self.args.plugin}

    def close(self) -> None:
        return None
