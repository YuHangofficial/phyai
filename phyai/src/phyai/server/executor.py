"""Execution backends for one complete model replica."""

from __future__ import annotations

import abc
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from phyai.server.lifecycle import (
    EngineUnavailableError,
    LifecycleState,
    SupervisorProtocol,
)


ResultT = TypeVar("ResultT")


class ReplicaExecutor(abc.ABC):
    """Execute requests on exactly one complete model replica."""

    replica_id: int
    mode: str
    rank_count: int

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def step(self, request: Any) -> Any: ...

    @abc.abstractmethod
    def submit(self, request: Any) -> Future[Any]: ...

    @property
    @abc.abstractmethod
    def healthy(self) -> bool:
        """Whether this replica can accept another request.

        A backend that has not started yet must report ``True``: managed
        executors support lazy startup on their first request, so a
        created-but-unstarted replica is a valid routing target.
        """

    @property
    @abc.abstractmethod
    def inflight(self) -> int:
        """Number of accepted requests that have not completed yet."""

    @abc.abstractmethod
    def close(self) -> None: ...


@dataclass(slots=True)
class _InlineWork(Generic[ResultT]):
    request: Any
    future: Future[ResultT] | None


class InlineExecutor(ReplicaExecutor):
    """Run one rank in the caller process with lazy asynchronous support."""

    mode = "inline"
    rank_count = 1

    def __init__(
        self,
        core_factory: Callable[..., Any],
        args: Any,
        *,
        replica_id: int = 0,
    ) -> None:
        self.replica_id = replica_id
        self.core = core_factory(args)
        self._condition = threading.Condition()
        self._queue: deque[_InlineWork[Any]] = deque()
        self._active = False
        self._closed = False
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise EngineUnavailableError("cannot start a closed inline executor.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed inline executor.")

    def _bind_device(self) -> None:
        config = getattr(self.core, "config", None)
        device_config = getattr(config, "device", None)
        target = getattr(device_config, "target", None)
        if target is None:
            return
        import torch
        from phyai.utils.cuda import resolve_device

        # The executing thread may differ from the one that built the core, so
        # re-pin per request. Resolve like init_cuda does: a bare "cuda" target
        # (the config default) has no index and torch.cuda.set_device rejects
        # it, so fold in LOCAL_RANK / device 0 first.
        device = resolve_device(target)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    def _execute(self, request: Any) -> Any:
        self._bind_device()
        return self.core.step(request)

    def step(self, request: Any) -> Any:
        """Enter the core from this caller thread in FIFO order."""
        work = _InlineWork[Any](request=request, future=None)
        with self._condition:
            self._ensure_open()
            self._queue.append(work)
            self._condition.notify_all()
            while self._active or not self._queue or self._queue[0] is not work:
                if self._closed:
                    try:
                        self._queue.remove(work)
                    except ValueError:
                        pass
                    raise EngineUnavailableError(
                        "inline executor closed while waiting."
                    )
                self._condition.wait()
            self._queue.popleft()
            self._active = True
        try:
            # A malformed or otherwise request-local failure must not poison an
            # inline replica: the exception propagates as-is and the executor
            # keeps serving the next request.
            return self._execute(request)
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def submit(self, request: Any) -> Future[Any]:
        future: Future[Any] = Future()
        with self._condition:
            try:
                self._ensure_open()
            except BaseException as error:
                future.set_exception(error)
                return future
            self._queue.append(_InlineWork(request=request, future=future))
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"phyai-inline-{self.replica_id}",
                    daemon=True,
                )
                self._worker.start()
            self._condition.notify_all()
        return future

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._closed and not self._active and not self._queue:
                        return
                    if (
                        not self._active
                        and self._queue
                        and self._queue[0].future is not None
                    ):
                        work = self._queue.popleft()
                        self._active = True
                        break
                    self._condition.wait()
            future = work.future
            assert future is not None
            if future.set_running_or_notify_cancel():
                try:
                    result = self._execute(work.request)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
            with self._condition:
                self._active = False
                self._condition.notify_all()

    @property
    def healthy(self) -> bool:
        # Inline failures are request-local by design: they propagate to the
        # caller and never poison the replica, so health is simply "open".
        with self._condition:
            return not self._closed

    @property
    def inflight(self) -> int:
        with self._condition:
            return len(self._queue) + int(self._active)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            queued = tuple(self._queue)
            self._queue.clear()
            self._condition.notify_all()
            while self._active:
                self._condition.wait()
            worker = self._worker
        for work in queued:
            if work.future is not None:
                work.future.cancel()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5.0)
        self.core.close()


class ExternalExecutor(InlineExecutor):
    """Run one externally launched rank without spawning helper processes."""

    mode = "external"

    def __init__(
        self,
        core_factory: Callable[..., Any],
        args: Any,
        *,
        world_size: int,
        rank: int,
        output_rank: int = 0,
    ) -> None:
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size < 1
        ):
            raise ValueError(f"world_size must be a positive int, got {world_size!r}.")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < world_size
        ):
            raise ValueError(
                f"external rank must be in [0, {world_size}), got {rank!r}."
            )
        if (
            not isinstance(output_rank, int)
            or isinstance(output_rank, bool)
            or not 0 <= output_rank < world_size
        ):
            raise ValueError(
                f"output_rank must be in [0, {world_size}), got {output_rank!r}."
            )
        self.rank_count = world_size
        self._rank = rank
        self._output_rank = output_rank
        super().__init__(core_factory, args, replica_id=0)

    def _execute(self, request: Any) -> Any:
        result = super()._execute(request)
        return result if self._rank == self._output_rank else None


class _SharedManager:
    def __init__(self, manager: SupervisorProtocol, references: int) -> None:
        self.manager = manager
        self._remaining = references
        self._started = False
        self._closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise EngineUnavailableError("managed execution backend is closed.")
            if self._started:
                return
            self.manager.start()
            self._started = True

    def release(self) -> None:
        with self._lock:
            if self._remaining <= 0:
                return
            self._remaining -= 1
            close = self._remaining == 0 and not self._closed
            if close:
                self._closed = True
        if close:
            self.manager.close()


class MultiprocessExecutor(ReplicaExecutor):
    """Submit one replica's requests through a shared worker supervisor.

    Requests go straight to the supervisor: each worker process consumes its
    pipe strictly in order, so per-replica FIFO holds without a local queue or
    helper thread. Two consequences of the direct path: a managed future is
    never cancellable (the supervisor marks it running on creation), and the
    supervisor's ``execution_timeout_s`` clock starts at submit time.
    """

    mode = "multiprocess"

    def __init__(
        self,
        shared: _SharedManager,
        *,
        replica_id: int,
        rank_count: int,
    ) -> None:
        self._shared = shared
        self.replica_id = replica_id
        self.rank_count = rank_count
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise EngineUnavailableError("cannot start a closed replica executor.")
        self._shared.start()

    def step(self, request: Any) -> Any:
        return self.submit(request).result()

    def submit(self, request: Any) -> Future[Any]:
        if self._closed:
            failed: Future[Any] = Future()
            failed.set_exception(
                EngineUnavailableError("cannot execute on a closed executor.")
            )
            return failed
        try:
            # Lazy startup on the first request keeps auto_start=False cheap.
            self.start()
            return self._shared.manager.submit(self.replica_id, request)
        except BaseException as error:
            failed = Future()
            failed.set_exception(error)
            return failed

    @property
    def healthy(self) -> bool:
        if self._closed:
            return False
        manager = self._shared.manager
        state = manager.state
        if state in (LifecycleState.CREATED, LifecycleState.STARTING):
            return True
        if state != LifecycleState.RUNNING:
            return False
        per_replica = manager.healthy
        return self.replica_id < len(per_replica) and per_replica[self.replica_id]

    @property
    def inflight(self) -> int:
        if self._closed:
            return 0
        per_replica = self._shared.manager.inflight
        if self.replica_id >= len(per_replica):
            return 0
        return per_replica[self.replica_id]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shared.release()


def managed_executors(
    manager: SupervisorProtocol,
    *,
    replica_count: int,
    rank_count: int,
) -> tuple[MultiprocessExecutor, ...]:
    """Create single-replica views over one shared worker supervisor."""
    shared = _SharedManager(manager, replica_count)
    return tuple(
        MultiprocessExecutor(shared, replica_id=replica_id, rank_count=rank_count)
        for replica_id in range(replica_count)
    )


__all__ = [
    "EngineUnavailableError",
    "ExternalExecutor",
    "InlineExecutor",
    "MultiprocessExecutor",
    "ReplicaExecutor",
    "managed_executors",
]
