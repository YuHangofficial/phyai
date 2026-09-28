"""Small CUDA runtimes for executor and rank-group integration tests."""

from __future__ import annotations

import dataclasses
import os
from datetime import timedelta
from time import sleep
from typing import Any, Mapping

from phyai.server.deployment import WorkerPlacement


class CudaRuntime:
    """A one-device worker with optional PhyAI collectives."""

    def __init__(
        self,
        placement: WorkerPlacement,
        args: Mapping[str, Any] | None,
    ) -> None:
        import torch

        self.placement = placement
        self.args = dict(args or {})
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the spawned test worker.")
        torch.cuda.set_device(placement.device_index)
        self.dist = None
        self.parallel = None
        self.mesh = None
        if self.args.get("collective", False):
            import torch.distributed as dist

            import phyai.parallel as parallel

            self.parallel = parallel
            dist.init_process_group(
                "nccl",
                timeout=timedelta(seconds=float(self.args.get("dist_timeout_s", 60))),
            )
            self.mesh = parallel.init(
                self.args["parallel"],
                device="cuda",
                backend="nccl",
                enable_pynccl=bool(self.args.get("enable_pynccl", True)),
                pynccl_groups=["dense_tp"]
                if self.args.get("mode") == "graph_world"
                else None,
            )
            self.dist = dist

    def metadata(self) -> Mapping[str, Any]:
        torch = self.torch
        metadata = {
            "worker_id": self.placement.worker_id,
            "replica_id": self.placement.replica_id,
            "replica_rank": self.placement.replica_rank,
            "device_index": self.placement.device_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "rank": os.environ.get("RANK"),
            "world_size": os.environ.get("WORLD_SIZE"),
        }
        if self.parallel is not None:
            backends = self.parallel.get_dispatcher().registry.all()
            metadata["collective_backends"] = tuple(b.name for b in backends)
            metadata["topology"] = dataclasses.asdict(self.mesh.topology())
            # (mesh, group) -> device string for every pynccl communicator, so
            # tests can assert each worker attached NCCL to its own GPU.
            metadata["pynccl_attach_devices"] = {
                f"{mesh}:{group}": str(device)
                for backend in backends
                if backend.name == "pynccl"
                for (mesh, group), device in backend.attached().items()
            }
        return metadata

    def execute(self, payload: Any) -> Any:
        torch = self.torch
        if not isinstance(payload, Mapping):
            payload = {"value": payload}
        if (delay := payload.get("sleep_s")) is not None:
            sleep(float(delay))
        if payload.get("fail"):
            raise RuntimeError(
                f"requested CUDA worker failure (worker={self.placement.worker_id})"
            )

        mode = self.args.get("mode", "replica")
        if mode == "replica":
            value = float(payload.get("value", 0.0))
            tensor = torch.tensor([value], dtype=torch.float32, device="cuda")
            result = tensor * 2.0 + self.placement.replica_id
            torch.cuda.synchronize()
            return {
                "worker_id": self.placement.worker_id,
                "replica_id": self.placement.replica_id,
                "device": str(result.device),
                "value": float(result.item()),
            }

        if mode == "cuda_tensor":
            value = float(payload.get("value", 0.0))
            result = torch.tensor([value], dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            return result

        if self.parallel is None or self.dist is None or self.mesh is None:
            raise RuntimeError(f"collective mode {mode!r} was not initialized.")

        rank = self.dist.get_rank()
        if mode == "sequence_parallel":
            return self._sequence_parallel()
        if mode == "transformer_sequence_parallel":
            return self._transformer_sequence_parallel()
        if mode == "domains":
            return self._overlapping_domains()
        if mode == "tp":
            value = torch.tensor([float(rank + 1)], device="cuda")
            reduced = self.parallel.all_reduce(value, group="dense_tp")
            gathered = self.parallel.all_gather(value, group="dense_tp", dim=0)
            scatter_input = torch.tensor(
                [float(rank + 1), float(rank + 1)], device="cuda"
            )
            scattered = self.parallel.reduce_scatter(
                scatter_input,
                group="dense_tp",
                dim=0,
            )
            source = torch.tensor([42.0 if rank == 0 else -1.0], device="cuda")
            broadcast = self.parallel.broadcast(source, group="dense_tp", src=0)
            torch.cuda.synchronize()
            return {
                "rank": rank,
                "tensor_rank": self.mesh.group_rank("dense_tp"),
                "reduced": float(reduced.item()),
                "gathered": [float(item) for item in gathered.cpu().tolist()],
                "scattered": float(scattered.item()),
                "broadcast": float(broadcast.item()),
            }

        if mode == "cfg":
            cfg_rank = self.mesh.group_rank("cfg")
            branch = torch.tensor([10.0 if cfg_rank == 0 else 20.0], device="cuda")
            pair = self.parallel.all_gather(branch, group="cfg", dim=0)
            scale = float(payload.get("guidance_scale", 3.0))
            guided = pair[1] + scale * (pair[0] - pair[1])
            torch.cuda.synchronize()
            return {
                "rank": rank,
                "cfg_rank": cfg_rank,
                "pair": [float(item) for item in pair.cpu().tolist()],
                "guided": float(guided.item()),
            }

        if mode == "cfg_tensor":
            cfg_rank = self.mesh.group_rank("cfg")
            tensor_rank = self.mesh.group_rank("dense_tp")
            value = torch.tensor([float(rank + 1)], device="cuda")
            tensor_sum = self.parallel.all_reduce(value, group="dense_tp")
            branch = torch.tensor([10.0 if cfg_rank == 0 else 20.0], device="cuda")
            pair = self.parallel.all_gather(branch, group="cfg", dim=0)
            torch.cuda.synchronize()
            return {
                "rank": rank,
                "cfg_rank": cfg_rank,
                "tensor_rank": tensor_rank,
                "tensor_sum": float(tensor_sum.item()),
                "pair": [float(item) for item in pair.cpu().tolist()],
            }

        if mode in ("graph", "graph_world"):
            return self._graph_captured_collectives(
                group="dense_tp" if mode == "graph" else "world"
            )

        raise ValueError(f"unknown CUDA test runtime mode {mode!r}.")

    def _overlapping_domains(self) -> dict[str, Any]:
        torch = self.torch
        mesh = self.mesh
        rank = self.dist.get_rank()
        assert mesh.group("dense_tp") is mesh.group("attention_dp")
        assert mesh.group("attention_dp") is mesh.group("moe_ep")
        backend = next(
            b
            for b in self.parallel.get_dispatcher().registry.all()
            if b.name == "pynccl"
        )
        assert (
            backend._comms[(mesh.name, "dense_tp")]
            is backend._comms[(mesh.name, "moe_ep")]
        )
        for group in mesh.group_names:
            members = mesh.group_members(group)
            value = torch.tensor([float(rank + 1)], device="cuda")
            assert self.parallel.all_reduce(value, group=group).item() == sum(
                r + 1 for r in members
            )
            assert self.parallel.all_gather(value, group=group).tolist() == [
                r + 1 for r in members
            ]
            assert (
                self.parallel.broadcast(value, group=group, src=len(members) - 1).item()
                == members[-1] + 1
            )
            exchanged = self.parallel.all_to_all(
                value.repeat(len(members)), group=group
            )
            assert exchanged.tolist() == [r + 1 for r in members]
        self.parallel.shutdown()
        assert self.dist.is_initialized()
        assert len(self.dist.distributed_c10d._world.pg_map) == 1
        self.mesh = self.parallel.init(
            self.args["parallel"], device="cuda", backend="nccl"
        )
        return {"rank": rank, "world": self.dist.get_world_size()}

    def _sequence_parallel(self) -> dict[str, Any]:
        from phyai.layers.linear import ColumnParallelLinear, RowParallelLinear

        torch = self.torch
        torch.manual_seed(17)
        rank = self.dist.get_rank()
        world = self.dist.get_world_size()
        w1 = torch.randn(32, 16, device="cuda") * 0.1
        w2 = torch.randn(16, 32, device="cuda") * 0.1
        bias = torch.randn(16, device="cuda") * 0.1
        x = torch.randn(world * 2, 16, device="cuda")
        col = ColumnParallelLinear(
            16,
            32,
            bias=False,
            sequence_parallel=True,
            prefix="col",
            params_dtype=torch.float32,
        )
        row = RowParallelLinear(
            32,
            16,
            bias=True,
            sequence_parallel=True,
            prefix="row",
            params_dtype=torch.float32,
        )
        col.weight.weight_loader(col.weight, w1, None)
        row.weight.weight_loader(row.weight, w2, None)
        row.bias.weight_loader(row.bias, bias, None)
        intermediate, _ = col(x.chunk(world)[rank])
        result, _ = row(intermediate)
        expected = torch.nn.functional.linear(
            torch.nn.functional.linear(x, w1), w2, bias
        ).chunk(world)[rank]
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        return {"rank": rank, "shape": tuple(result.shape)}

    def _transformer_sequence_parallel(self) -> dict[str, Any]:
        from phyai.engine_config import EngineConfig, ParallelConfig, init_engine_config
        from phyai.layers.transformer_block import TransformerBlock
        from phyai.parallel.mesh import Mesh
        from phyai.parallel.layout import build_rank_layout

        torch = self.torch
        torch.manual_seed(31)
        init_engine_config(EngineConfig(parallel=self.args["parallel"]))
        self.parallel.register_mesh(
            Mesh(build_rank_layout(ParallelConfig()), name="reference")
        )
        kwargs = dict(
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            mlp_gated=False,
            mlp_activation="gelu",
            attn_backend="sdpa",
            norm_backend="phyai-kernel",
            params_dtype=torch.float32,
            prefix="block",
        )
        reference = TransformerBlock(
            **kwargs, mesh="reference", sequence_parallel=False
        )
        distributed = TransformerBlock(**kwargs)
        assert distributed.qkv_proj.sequence_parallel
        weights = {}
        for param in reference.parameters():
            param.data.normal_(std=0.1)
            pieces = param.detach().chunk(len(param.hf_keys), dim=0)
            for (key, leg), value in zip(param.hf_keys, pieces, strict=True):
                weights[(key, leg)] = value
        for param in distributed.parameters():
            for key, leg in param.hf_keys:
                param.weight_loader(param, weights[(key, leg)], leg)
        rank = self.dist.get_rank()
        world = self.dist.get_world_size()
        x = torch.randn(world * 2, 4, 32, device="cuda")
        with torch.inference_mode():
            expected = reference(x).chunk(world)[rank]
            result = distributed(x.chunk(world)[rank])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        return {"rank": rank, "shape": tuple(result.shape)}

    def _graph_captured_collectives(self, *, group: str) -> dict[str, Any]:
        """Capture collectives in a CUDA graph and compare replay with eager.

        Eager collectives go through torch.distributed (pynccl declines eager
        mode), so they are an independent reference for the captured path.
        This test attaches only ``dense_tp`` in the ``graph_world`` mode to
        exercise the fallback to NcclBackend under capture.
        """
        torch = self.torch
        parallel = self.parallel
        assert parallel is not None
        from phyai.parallel.backend import Op  # noqa: PLC0415

        rank = self.dist.get_rank()
        x = torch.tensor(
            [[rank + 1.0, rank + 2.0], [rank + 3.0, rank + 4.0]], device="cuda"
        )
        gather = group == "dense_tp"
        eager_reduced = parallel.all_reduce(x, group=group)
        # A column slice is non-contiguous and non-dense: raw-pointer NCCL
        # paths must stage it instead of reading its storage linearly.
        eager_reduced_view = parallel.all_reduce(x[:, :1], group=group)
        eager_gathered = parallel.all_gather(x, group=group, dim=-1) if gather else None

        parallel.warmup_collectives()
        static = torch.zeros_like(x)
        graph = torch.cuda.CUDAGraph()
        with parallel.graph_capture(), torch.cuda.graph(graph):
            captured_reduced = parallel.all_reduce(static, group=group)
            captured_reduced_view = parallel.all_reduce(static[:, :1], group=group)
            captured_gathered = (
                parallel.all_gather(static, group=group, dim=-1) if gather else None
            )
            # Pure cache lookup (no collective is issued): which backend the
            # captured all_reduce on this group was dispatched to.
            selected = parallel.get_dispatcher().select(
                op=Op.ALL_REDUCE, mesh=self.mesh, group=group, tensor=static
            )
        static.copy_(x)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(captured_reduced, eager_reduced)
        torch.testing.assert_close(captured_reduced_view, eager_reduced_view)
        if gather:
            torch.testing.assert_close(captured_gathered, eager_gathered)

        def rows(tensor):
            return None if tensor is None else tensor.cpu().tolist()

        return {
            "rank": rank,
            "group": group,
            "captured_backend": selected.name,
            "eager_reduced": rows(eager_reduced),
            "captured_reduced": rows(captured_reduced),
            "eager_reduced_view": rows(eager_reduced_view),
            "captured_reduced_view": rows(captured_reduced_view),
            "eager_gathered": rows(eager_gathered),
            "captured_gathered": rows(captured_gathered),
        }

    def close(self) -> None:
        if self.parallel is not None:
            # Destroys the direct NCCL communicators pynccl attached; must run
            # before the torch process group goes away.
            self.parallel.shutdown()
        if self.dist is not None and self.dist.is_initialized():
            self.dist.destroy_process_group()


def create_cuda_runtime(
    placement: WorkerPlacement,
    args: Mapping[str, Any] | None,
) -> CudaRuntime:
    return CudaRuntime(placement, args)


__all__ = ["CudaRuntime", "create_cuda_runtime"]
