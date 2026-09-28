"""Spawn-based GPU worker lifecycle and request supervision."""

from __future__ import annotations

import atexit
import ctypes
import importlib
import itertools
import math
import os
import signal
import threading
import time
import traceback as traceback_module
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any, Mapping, Protocol

import torch.multiprocessing as mp

from phyai.server.deployment import DeploymentPlan, WorkerPlacement
from phyai.server.lifecycle import EngineUnavailableError, LifecycleState
from phyai.server.process_protocol import (
    ExecuteRequest,
    ShutdownRequest,
    WorkerError,
    WorkerHello,
    WorkerReady,
    WorkerResult,
    WorkerStopped,
)
from phyai.utils import get_logger

logger = get_logger(__name__)
_MISSING = object()


class WorkerRuntime(Protocol):
    """Runtime constructed inside one spawned worker process.

    Result transport contract: the output rank's return value travels back
    over a ``torch.multiprocessing`` pipe, so CUDA tensors are shared with
    the parent as CUDA-IPC views of this worker's memory — not copies.
    Return freshly allocated tensors (never aliases of CUDA-graph or other
    persistent buffers, which the next step would overwrite) and synchronize
    the device before returning so the parent cannot observe unfinished
    writes. Shared results stay valid only while this worker process lives.
    """

    def metadata(self) -> Mapping[str, Any]: ...

    def execute(self, payload: Any) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerFactorySpec:
    """Registered import reference and arguments for a worker factory."""

    factory: str
    args: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.factory, str) or ":" not in self.factory:
            raise ValueError(
                "worker factory must use 'module:qualified_name' syntax, got "
                f"{self.factory!r}."
            )
        module_name, qualified_name = self.factory.split(":", 1)
        if not module_name or not qualified_name:
            raise ValueError(
                "worker factory module and qualified name must not be empty."
            )


@dataclass(frozen=True, slots=True)
class WorkerSupervisorConfig:
    """Lifecycle and rendezvous settings for one node's worker manager."""

    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    base_port: int = 29500
    startup_timeout_s: float = 600.0
    shutdown_timeout_s: float = 10.0
    execution_timeout_s: float | None = None
    extra_env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node_rank, int)
            or isinstance(self.node_rank, bool)
            or self.node_rank < 0
        ):
            raise ValueError(
                f"node_rank must be a non-negative int, got {self.node_rank!r}."
            )
        if not isinstance(self.master_addr, str):
            raise TypeError(f"master_addr must be a string, got {self.master_addr!r}.")
        master_addr = self.master_addr.strip()
        if not master_addr or "\0" in master_addr:
            raise ValueError("master_addr must not be empty.")
        object.__setattr__(self, "master_addr", master_addr)
        if (
            not isinstance(self.base_port, int)
            or isinstance(self.base_port, bool)
            or not 1 <= self.base_port <= 65535
        ):
            raise ValueError(
                f"base_port must be in [1, 65535], got {self.base_port!r}."
            )
        for name in ("startup_timeout_s", "shutdown_timeout_s"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        timeout = self.execution_timeout_s
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "execution_timeout_s must be None or finite and positive, got "
                f"{timeout!r}."
            )
        normalized_env = tuple((str(key), str(value)) for key, value in self.extra_env)
        if len({key for key, _ in normalized_env}) != len(normalized_env):
            raise ValueError("extra_env contains duplicate variable names.")
        if any(not key or "=" in key or "\0" in key for key, _ in normalized_env):
            raise ValueError("extra_env contains an invalid variable name.")
        if any("\0" in value for _, value in normalized_env):
            raise ValueError("extra_env values must not contain null bytes.")
        forbidden = {"PYTHONPATH", "PYTHONHOME"} & {
            key.upper() for key, _ in normalized_env
        }
        if forbidden:
            raise ValueError(
                f"extra_env cannot change Python import paths: {sorted(forbidden)!r}."
            )
        if any(key.upper() == "CUDA_VISIBLE_DEVICES" for key, _ in normalized_env):
            raise ValueError(
                "extra_env cannot set CUDA_VISIBLE_DEVICES: workers inherit the "
                "parent's device visibility and bind their placement index; a "
                "per-worker mask desynchronizes device numbering across "
                "processes and breaks CUDA-IPC tensor transport."
            )
        rendezvous = {
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
        } & {key.upper() for key, _ in normalized_env}
        if rendezvous:
            raise ValueError(
                f"extra_env cannot set {sorted(rendezvous)!r}: the supervisor "
                "derives the rendezvous and rank variables from the placement."
            )
        object.__setattr__(self, "extra_env", normalized_env)


@dataclass(slots=True)
class _WorkerHandle:
    placement: WorkerPlacement
    process: BaseProcess
    connection: Connection
    startup_state: str = "new"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PendingRequest:
    future: Future[Any]
    replica_id: int
    remaining_workers: set[int]
    output_worker_id: int
    deadline: float | None
    output: Any = _MISSING


def _resolve_factory(reference: str):
    module_name, qualified_name = reference.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(
            f"worker factory {reference!r} resolved to non-callable {obj!r}."
        )
    return obj


def _set_parent_death_signal() -> None:
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except (AttributeError, OSError):
        return


def _set_process_title(placement: WorkerPlacement) -> None:
    try:
        import setproctitle  # noqa: PLC0415

        memberships = "_".join(
            f"{name.upper()}{rank}" for name, rank in placement.group_ranks if rank
        )
        suffix = f"_{memberships}" if memberships else ""
        setproctitle.setproctitle(f"phyai::worker_R{placement.replica_id}{suffix}")
    except ImportError:
        return


def _send_error(
    connection: Connection,
    placement: WorkerPlacement,
    phase: str,
    error: BaseException,
    request_id: int | None,
) -> None:
    try:
        connection.send(
            WorkerError(
                worker_id=placement.worker_id,
                phase=phase,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback_module.format_exc(),
                request_id=request_id,
            )
        )
    except (BrokenPipeError, EOFError, OSError, ValueError):
        pass


def _worker_main(
    placement: WorkerPlacement,
    connection: Connection,
    factory_spec: WorkerFactorySpec,
    environment: dict[str, str],
) -> None:
    """Spawn target. Model and runtime modules are imported only by the factory."""
    # The rendezvous/identity environment is applied here, in the child,
    # before anything reads it — never by mutating the parent's os.environ.
    os.environ.update(environment)
    _set_parent_death_signal()
    _set_process_title(placement)
    runtime: WorkerRuntime | None = None
    phase = "hello"
    request_id: int | None = None
    clean_shutdown = False
    try:
        connection.send(
            WorkerHello(
                worker_id=placement.worker_id,
                replica_id=placement.replica_id,
                replica_rank=placement.replica_rank,
                pid=os.getpid(),
            )
        )
        phase = "startup"
        factory = _resolve_factory(factory_spec.factory)
        runtime = factory(placement, factory_spec.args)
        metadata = dict(runtime.metadata())
        connection.send(WorkerReady(placement.worker_id, metadata))

        phase = "execute"
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break
            if isinstance(command, ShutdownRequest):
                clean_shutdown = True
                break
            if not isinstance(command, ExecuteRequest):
                raise TypeError(f"unknown worker command {type(command).__name__}.")
            request_id = command.request_id
            output = runtime.execute(command.payload)
            connection.send(
                WorkerResult(
                    worker_id=placement.worker_id,
                    request_id=request_id,
                    payload=output if placement.is_output_rank else None,
                )
            )
            request_id = None
    except BaseException as error:
        _send_error(connection, placement, phase, error, request_id)
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as error:
                _send_error(connection, placement, "shutdown", error, request_id)
                if clean_shutdown:
                    raise error
        if clean_shutdown:
            try:
                connection.send(WorkerStopped(placement.worker_id))
            except (BrokenPipeError, EOFError, OSError):
                pass
        connection.close()


class WorkerSupervisor:
    """Own and supervise all GPU workers assigned to one local node."""

    def __init__(
        self,
        plan: DeploymentPlan,
        factory: WorkerFactorySpec,
        config: WorkerSupervisorConfig | None = None,
    ) -> None:
        self.plan = plan
        self.factory = factory
        self.config = config or WorkerSupervisorConfig()
        if self.config.node_rank >= len(plan.nodes):
            raise ValueError(
                f"node_rank {self.config.node_rank} is absent from the parallel plan."
            )
        if self.config.base_port + plan.replica_count - 1 > 65535:
            raise ValueError(
                "base_port plus replica count exceeds the valid TCP port range."
            )
        self._placements = plan.workers_for_node(self.config.node_rank)
        if not self._placements:
            raise ValueError(
                f"parallel plan assigns no workers to node {self.config.node_rank}."
            )
        self._context = mp.get_context("spawn")
        self._handles: dict[int, _WorkerHandle] = {}
        self._pending: dict[int, _PendingRequest] = {}
        self._request_ids = itertools.count()
        self._inflight = [0] * plan.replica_count
        self._state = LifecycleState.CREATED
        self._failure: RuntimeError | None = None
        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = threading.Event()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._atexit_registered = False

    @property
    def state(self) -> LifecycleState:
        with self._state_lock:
            return self._state

    @property
    def failure(self) -> RuntimeError | None:
        with self._state_lock:
            return self._failure

    @property
    def healthy(self) -> tuple[bool, ...]:
        """Whether each replica can accept another request.

        Failure handling is fail-fast: any worker death fails the whole
        supervisor, so all replicas report the same health.
        """
        with self._state_lock:
            running = self._state == LifecycleState.RUNNING
        return (running,) * self.replica_count

    @property
    def inflight(self) -> tuple[int, ...]:
        with self._state_lock:
            return tuple(self._inflight)

    @property
    def replica_count(self) -> int:
        return self.plan.replica_count

    @property
    def worker_metadata(self) -> dict[int, dict[str, Any]]:
        with self._state_lock:
            return {
                worker_id: dict(handle.metadata)
                for worker_id, handle in self._handles.items()
            }

    def _worker_environment(self, placement: WorkerPlacement) -> dict[str, str]:
        replica = self.plan.workers_for_replica(placement.replica_id)
        root_node = self.plan.nodes[replica[0].node_rank]
        # A single-node replica rendezvous locally at the supervisor's
        # master_addr; a multi-node replica carries its root node's address in
        # the plan (DeploymentPlan.build refuses one without it).
        master_addr = root_node.address or self.config.master_addr
        values = dict(self.config.extra_env)
        # Only torchrun-style rendezvous variables travel through the
        # environment. No CUDA_VISIBLE_DEVICES: every worker keeps the parent's
        # full device visibility and binds its placement device index instead,
        # so device numbering agrees across processes (which is what lets CUDA
        # tensors travel between them as IPC handles). LOCAL_RANK carries the
        # device index (a backstop for any stray bare-"cuda" resolve), which
        # is why no LOCAL_WORLD_SIZE is set: with an arbitrary device list the
        # two cannot both keep torchrun's meaning. Placement identity reaches
        # the worker as the WorkerPlacement object; node topology is probed at
        # runtime by phyai.parallel.init.
        values.update(
            {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(self.config.base_port + placement.replica_id),
                "RANK": str(placement.replica_rank),
                "WORLD_SIZE": str(self.plan.replica_world_size),
                "LOCAL_RANK": str(placement.device_index),
            }
        )
        return values

    def start(self) -> None:
        with self._state_lock:
            if self._state != LifecycleState.CREATED:
                raise EngineUnavailableError(
                    f"cannot start WorkerSupervisor in state {self._state}."
                )
            self._state = LifecycleState.STARTING

        started: list[_WorkerHandle] = []
        try:
            for placement in self._placements:
                parent_connection, child_connection = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_worker_main,
                    name=f"phyai-worker-{placement.worker_id}",
                    args=(
                        placement,
                        child_connection,
                        self.factory,
                        self._worker_environment(placement),
                    ),
                )
                handle = _WorkerHandle(placement, process, parent_connection)
                try:
                    process.start()
                except BaseException:
                    child_connection.close()
                    parent_connection.close()
                    raise
                else:
                    child_connection.close()
                self._handles[placement.worker_id] = handle
                started.append(handle)

            self._wait_until_ready()
        except BaseException as error:
            self._shutdown_workers(force=True, handles=started)
            with self._state_lock:
                self._state = LifecycleState.FAILED
                self._failure = EngineUnavailableError(
                    f"worker process startup failed: {type(error).__name__}: {error}"
                )
                failure = self._failure
            # Same type the routing layer sees on every other backend death.
            raise failure from error

        with self._state_lock:
            self._state = LifecycleState.RUNNING
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="phyai-worker-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        connection_handles = {
            handle.connection: handle for handle in self._handles.values()
        }
        sentinel_handles = {
            handle.process.sentinel: handle for handle in self._handles.values()
        }
        watched = [*connection_handles, *sentinel_handles]

        while any(handle.startup_state != "ready" for handle in self._handles.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                states = {
                    worker_id: handle.startup_state
                    for worker_id, handle in self._handles.items()
                }
                raise TimeoutError(
                    f"workers did not become ready within "
                    f"{self.config.startup_timeout_s}s: {states}."
                )
            ready = wait(watched, timeout=min(remaining, 1.0))
            for watched_object in ready:
                handle = connection_handles.get(watched_object)
                if handle is None:
                    continue
                try:
                    message = handle.connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"worker {handle.placement.worker_id} closed its startup pipe."
                    ) from error
                self._handle_startup_message(handle, message)

            for sentinel, handle in sentinel_handles.items():
                if sentinel in ready or handle.process.exitcode is not None:
                    raise RuntimeError(
                        f"worker {handle.placement.worker_id} exited during startup "
                        f"with code {handle.process.exitcode}."
                    )

    def _handle_startup_message(self, handle: _WorkerHandle, message: Any) -> None:
        placement = handle.placement
        if isinstance(message, WorkerError):
            raise RuntimeError(
                f"worker {message.worker_id} failed during {message.phase}: "
                f"{message.error}\n{message.traceback}"
            )
        if isinstance(message, WorkerHello):
            if handle.startup_state != "new":
                raise RuntimeError(
                    f"worker {placement.worker_id} sent duplicate HELLO."
                )
            expected = (
                placement.worker_id,
                placement.replica_id,
                placement.replica_rank,
            )
            actual = (
                message.worker_id,
                message.replica_id,
                message.replica_rank,
            )
            if actual != expected:
                raise RuntimeError(
                    f"worker identity mismatch: expected {expected!r}, got {actual!r}."
                )
            if message.pid != handle.process.pid:
                raise RuntimeError(
                    f"worker {placement.worker_id} reported pid {message.pid}, "
                    f"expected {handle.process.pid}."
                )
            handle.startup_state = "hello"
            return
        if isinstance(message, WorkerReady):
            if handle.startup_state != "hello":
                raise RuntimeError(
                    f"worker {placement.worker_id} sent READY before HELLO."
                )
            if message.worker_id != placement.worker_id:
                raise RuntimeError(
                    f"READY came from unexpected worker {message.worker_id}."
                )
            handle.metadata = dict(message.metadata)
            handle.startup_state = "ready"
            return
        raise RuntimeError(
            f"worker {placement.worker_id} sent unexpected startup message "
            f"{type(message).__name__}."
        )

    def submit(self, replica_id: int, payload: Any) -> Future[Any]:
        with self._state_lock:
            if self._state != LifecycleState.RUNNING:
                raise EngineUnavailableError(
                    f"cannot submit to WorkerSupervisor in state {self._state}."
                )
            replica = self.plan.workers_for_replica(replica_id)
            missing = [
                placement.worker_id
                for placement in replica
                if placement.worker_id not in self._handles
            ]
            if missing:
                raise RuntimeError(
                    f"replica {replica_id} has workers {missing} outside this "
                    "supervisor."
                )
            request_id = next(self._request_ids)
            future: Future[Any] = Future()
            future.set_running_or_notify_cancel()
            deadline = (
                None
                if self.config.execution_timeout_s is None
                else time.monotonic() + self.config.execution_timeout_s
            )
            output_worker_id = next(
                placement.worker_id for placement in replica if placement.is_output_rank
            )
            self._pending[request_id] = _PendingRequest(
                future=future,
                replica_id=replica_id,
                remaining_workers={placement.worker_id for placement in replica},
                output_worker_id=output_worker_id,
                deadline=deadline,
            )
            self._inflight[replica_id] += 1

        command = ExecuteRequest(request_id, payload)
        try:
            with self._send_lock:
                for placement in replica:
                    self._handles[placement.worker_id].connection.send(command)
        except Exception as error:
            self._mark_failed(
                RuntimeError(
                    f"failed to send request {request_id} to replica {replica_id}: "
                    f"{error}"
                ),
            )
        return future

    def execute(
        self, replica_id: int, payload: Any, timeout: float | None = None
    ) -> Any:
        return self.submit(replica_id, payload).result(timeout=timeout)

    def _monitor_loop(self) -> None:
        connection_handles = {
            handle.connection: handle for handle in self._handles.values()
        }
        sentinel_handles = {
            handle.process.sentinel: handle for handle in self._handles.values()
        }
        while not self._monitor_stop.is_set():
            watched = [*connection_handles, *sentinel_handles]
            if not watched:
                return
            try:
                ready = wait(watched, timeout=0.2)
            except (OSError, ValueError):
                # close() may have closed a connection under us; that is the
                # normal end of the loop, anything else is a real failure.
                if self.state in (LifecycleState.STOPPING, LifecycleState.STOPPED):
                    return
                raise
            for watched_object in ready:
                handle = connection_handles.get(watched_object)
                if handle is None:
                    continue
                try:
                    message = handle.connection.recv()
                except (EOFError, OSError):
                    if self.state not in (
                        LifecycleState.STOPPING,
                        LifecycleState.STOPPED,
                    ):
                        self._mark_failed(
                            RuntimeError(
                                f"worker {handle.placement.worker_id} closed its "
                                "control connection."
                            ),
                        )
                    connection_handles.pop(handle.connection, None)
                    sentinel_handles.pop(handle.process.sentinel, None)
                    continue
                except BaseException as error:
                    self._mark_failed(
                        RuntimeError(
                            f"failed to receive a message from worker "
                            f"{handle.placement.worker_id}: "
                            f"{type(error).__name__}: {error}"
                        ),
                    )
                    connection_handles.pop(handle.connection, None)
                    sentinel_handles.pop(handle.process.sentinel, None)
                    continue
                self._handle_runtime_message(handle, message)

            if self.state not in (LifecycleState.STOPPING, LifecycleState.STOPPED):
                for sentinel, handle in sentinel_handles.items():
                    if sentinel in ready or handle.process.exitcode is not None:
                        self._mark_failed(
                            RuntimeError(
                                f"worker {handle.placement.worker_id} exited "
                                f"unexpectedly with code {handle.process.exitcode}."
                            ),
                        )
                        break
                self._check_request_deadlines()

    def _handle_runtime_message(self, handle: _WorkerHandle, message: Any) -> None:
        message_worker_id = getattr(message, "worker_id", handle.placement.worker_id)
        if message_worker_id != handle.placement.worker_id:
            self._mark_failed(
                RuntimeError(
                    f"worker connection {handle.placement.worker_id} reported worker "
                    f"{message_worker_id}."
                ),
            )
            return
        if isinstance(message, WorkerResult):
            if self.state in (LifecycleState.STOPPING, LifecycleState.STOPPED):
                return
            self._record_result(message)
            return
        if isinstance(message, WorkerError):
            self._mark_failed(
                RuntimeError(
                    f"worker {message.worker_id} failed during {message.phase}: "
                    f"{message.error}\n{message.traceback}"
                ),
            )
            return
        if isinstance(message, WorkerStopped):
            if self.state not in (LifecycleState.STOPPING, LifecycleState.STOPPED):
                self._mark_failed(
                    RuntimeError(f"worker {message.worker_id} stopped unexpectedly."),
                )
                return
            return
        self._mark_failed(
            RuntimeError(
                f"worker {handle.placement.worker_id} sent unexpected runtime "
                f"message {type(message).__name__}."
            ),
        )
        return

    def _record_result(self, message: WorkerResult) -> None:
        completion: tuple[Future[Any], Any] | None = None
        error: RuntimeError | None = None
        with self._state_lock:
            pending = self._pending.get(message.request_id)
            if pending is None:
                error = RuntimeError(
                    f"worker {message.worker_id} returned unknown request "
                    f"{message.request_id}."
                )
            elif message.worker_id not in pending.remaining_workers:
                error = RuntimeError(
                    f"worker {message.worker_id} returned duplicate request "
                    f"{message.request_id}."
                )
            elif (
                placement := next(
                    (
                        item
                        for item in self.plan.workers
                        if item.worker_id == message.worker_id
                    ),
                    None,
                )
            ) is None or placement.replica_id != pending.replica_id:
                error = RuntimeError(
                    f"worker {message.worker_id} returned request "
                    f"{message.request_id} for the wrong replica."
                )
            else:
                pending.remaining_workers.remove(message.worker_id)
                if message.worker_id == pending.output_worker_id:
                    pending.output = message.payload
                if not pending.remaining_workers:
                    if pending.output is _MISSING:
                        error = RuntimeError(
                            f"request {message.request_id} completed without output "
                            f"worker {pending.output_worker_id}."
                        )
                    else:
                        self._pending.pop(message.request_id)
                        self._inflight[pending.replica_id] -= 1
                        completion = (pending.future, pending.output)
        if error is not None:
            self._mark_failed(error)
        elif completion is not None and not completion[0].cancelled():
            completion[0].set_result(completion[1])

    def _check_request_deadlines(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            expired = [
                request_id
                for request_id, pending in self._pending.items()
                if pending.deadline is not None and pending.deadline <= now
            ]
        if not expired:
            return
        self._mark_failed(
            TimeoutError(f"worker request {expired[0]} exceeded execution timeout.")
        )

    @staticmethod
    def _as_runtime_error(error: BaseException) -> RuntimeError:
        # Every backend-death path funnels through here so pending futures and
        # subsequent submits consistently see EngineUnavailableError.
        if isinstance(error, EngineUnavailableError):
            return error
        return EngineUnavailableError(str(error))

    def _mark_failed(self, error: BaseException) -> None:
        runtime_error = self._as_runtime_error(error)
        pending: list[Future[Any]] = []
        with self._state_lock:
            if self._state in (LifecycleState.FAILED, LifecycleState.STOPPED):
                return
            self._state = LifecycleState.FAILED
            self._failure = runtime_error
            pending = [item.future for item in self._pending.values()]
            self._pending.clear()
            self._inflight = [0] * self.plan.replica_count
        for future in pending:
            if not future.done():
                future.set_exception(runtime_error)
        logger.error("worker supervisor failed: %s", runtime_error)
        self._monitor_stop.set()
        self._shutdown_workers(force=True)

    def close(self) -> None:
        with self._state_lock:
            if self._state in (LifecycleState.STOPPED, LifecycleState.CREATED):
                self._state = LifecycleState.STOPPED
                self._unregister_atexit()
                return
            failed = self._state == LifecycleState.FAILED
            if not failed:
                self._state = LifecycleState.STOPPING
            pending = [item.future for item in self._pending.values()]
            self._pending.clear()
            self._inflight = [0] * self.plan.replica_count

        close_error = EngineUnavailableError(
            "WorkerSupervisor closed before request completion."
        )
        for future in pending:
            if not future.done():
                future.set_exception(close_error)

        self._monitor_stop.set()
        monitor = self._monitor_thread
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=1.0)
        self._shutdown_workers(force=failed)
        with self._state_lock:
            self._state = (
                LifecycleState.FAILED
                if self._failure is not None
                else LifecycleState.STOPPED
            )
        self._unregister_atexit()

    def _unregister_atexit(self) -> None:
        with self._state_lock:
            if not self._atexit_registered:
                return
            self._atexit_registered = False
        atexit.unregister(self.close)

    def _shutdown_workers(
        self,
        *,
        force: bool,
        handles: list[_WorkerHandle] | None = None,
    ) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                owns_shutdown = False
            else:
                self._shutdown_started = True
                owns_shutdown = True
        if not owns_shutdown:
            # The owner's worst case is shutdown_timeout_s + 2 s of terminate
            # grace + one join per straggler; wait at least that long.
            self._shutdown_complete.wait(
                self.config.shutdown_timeout_s + 5.0 + len(self._handles)
            )
            return
        try:
            self._shutdown_workers_once(
                list(self._handles.values()) if handles is None else handles,
                force=force,
            )
        finally:
            self._shutdown_complete.set()

    def _shutdown_workers_once(
        self, handles: list[_WorkerHandle], *, force: bool
    ) -> None:
        shutdown_failures: list[str] = []
        if not force:
            with self._send_lock:
                for handle in handles:
                    if handle.process.is_alive():
                        try:
                            handle.connection.send(ShutdownRequest())
                        except (BrokenPipeError, EOFError, OSError, ValueError):
                            pass
            deadline = time.monotonic() + self.config.shutdown_timeout_s
            for handle in handles:
                handle.process.join(max(0.0, deadline - time.monotonic()))
            for handle in handles:
                if handle.process.is_alive():
                    shutdown_failures.append(
                        f"worker {handle.placement.worker_id} did not stop within "
                        f"{self.config.shutdown_timeout_s}s"
                    )
                elif handle.process.exitcode not in (None, 0):
                    shutdown_failures.append(
                        f"worker {handle.placement.worker_id} exited with code "
                        f"{handle.process.exitcode}"
                    )

        for handle in handles:
            if handle.process.is_alive():
                handle.process.terminate()
        deadline = time.monotonic() + min(self.config.shutdown_timeout_s, 2.0)
        for handle in handles:
            if handle.process.pid is None:
                continue
            handle.process.join(max(0.0, deadline - time.monotonic()))
        for handle in handles:
            if handle.process.is_alive():
                handle.process.kill()
        for handle in handles:
            if handle.process.pid is not None:
                handle.process.join(timeout=1.0)
            if handle.process.is_alive():
                shutdown_failures.append(
                    f"worker {handle.placement.worker_id} survived termination"
                )
        for handle in handles:
            try:
                handle.connection.close()
            except OSError:
                pass
        if shutdown_failures and not force:
            self._record_shutdown_failure("; ".join(shutdown_failures))

    def _record_shutdown_failure(self, detail: str) -> None:
        error = RuntimeError(f"worker process shutdown failed: {detail}.")
        with self._state_lock:
            if self._failure is None:
                self._failure = error
        logger.error("%s", error)

    def __enter__(self) -> "WorkerSupervisor":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "LifecycleState",
    "WorkerSupervisor",
    "WorkerSupervisorConfig",
    "WorkerFactorySpec",
    "WorkerRuntime",
]
