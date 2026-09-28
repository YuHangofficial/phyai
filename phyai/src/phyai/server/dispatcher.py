"""Request routing across independent model replicas."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Sequence

from phyai.server.executor import ReplicaExecutor
from phyai.server.lifecycle import EngineUnavailableError


class RequestDispatcher:
    """Least-loaded routing over independent complete replicas."""

    def __init__(self, executors: Sequence[ReplicaExecutor]) -> None:
        normalized = tuple(executors)
        if not normalized:
            raise ValueError("request dispatcher requires at least one executor.")
        ids = tuple(executor.replica_id for executor in normalized)
        if len(set(ids)) != len(ids):
            raise ValueError(f"executor replica ids must be unique, got {ids!r}.")
        self.executors = normalized
        self._lock = threading.Lock()
        self._setup_lock = threading.Lock()
        self._cursor = 0
        self._closed = False
        self._started = False

    @property
    def replica_count(self) -> int:
        return len(self.executors)

    @property
    def core(self) -> Any | None:
        """Return the in-process core for a single inline/external replica."""
        if len(self.executors) != 1:
            return None
        return getattr(self.executors[0], "core", None)

    def setup(self) -> None:
        # Serialize setup with close and with another setup caller.  This is
        # important for managers whose start operation is intentionally not
        # re-entrant (for example a spawn-based worker supervisor).
        with self._setup_lock:
            with self._lock:
                if self._closed:
                    raise EngineUnavailableError(
                        "cannot set up a closed request dispatcher."
                    )
                if self._started:
                    return
            try:
                for executor in self.executors:
                    executor.start()
            except BaseException:
                # A failed startup leaves a dispatcher with no reliable
                # partial state. Close every executor (including views that
                # had not reached ``start`` yet) so shared managers release
                # their references and callers cannot accidentally retry a
                # half-created deployment.
                with self._lock:
                    self._closed = True
                for executor in reversed(self.executors):
                    try:
                        executor.close()
                    except BaseException:
                        # Preserve the startup exception; the executor's own
                        # lifecycle object records any cleanup failure.
                        continue
                raise
            with self._lock:
                self._started = True

    def _select(self) -> ReplicaExecutor:
        candidates = [executor for executor in self.executors if executor.healthy]
        if not candidates:
            detail = ", ".join(
                f"replica {executor.replica_id}" for executor in self.executors
            )
            raise EngineUnavailableError(
                f"no healthy replicas are available ({detail})."
            )
        minimum = min(executor.inflight for executor in candidates)
        eligible = {
            executor.replica_id
            for executor in candidates
            if executor.inflight == minimum
        }
        with self._lock:
            # Round-robin breaks least-loaded ties so equal-load replicas
            # share work instead of the first one absorbing every request.
            for offset in range(len(self.executors)):
                index = (self._cursor + offset) % len(self.executors)
                executor = self.executors[index]
                if executor.replica_id in eligible:
                    self._cursor = (index + 1) % len(self.executors)
                    return executor
        raise AssertionError("healthy replica selection produced no candidate.")

    def step(self, request: Any) -> Any:
        with self._lock:
            if self._closed:
                raise EngineUnavailableError(
                    "cannot execute on a closed request dispatcher."
                )
        return self._select().step(request)

    def submit(self, request: Any) -> Future[Any]:
        with self._lock:
            closed = self._closed
        if closed:
            failed: Future[Any] = Future()
            failed.set_exception(
                EngineUnavailableError("cannot execute on a closed request dispatcher.")
            )
            return failed
        try:
            return self._select().submit(request)
        except BaseException as error:
            failed = Future()
            failed.set_exception(error)
            return failed

    def close(self) -> None:
        with self._setup_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            close_error: BaseException | None = None
            for executor in reversed(self.executors):
                try:
                    executor.close()
                except BaseException as error:
                    # Continue closing every replica, then preserve the first
                    # cleanup error for the caller.
                    if close_error is None:
                        close_error = error
            if close_error is not None:
                raise close_error

    def __enter__(self) -> "RequestDispatcher":
        self.setup()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["RequestDispatcher"]
