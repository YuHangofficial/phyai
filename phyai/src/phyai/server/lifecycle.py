"""Lifecycle vocabulary shared by executors and worker supervisors.

This module is a deliberate leaf (stdlib only) so ``executor`` and
``worker_supervisor`` can both import it without depending on each other:
the executor side needs the supervisor's state enum to judge replica health,
the supervisor side needs the executor's unavailability error to fail
pending futures.
"""

from __future__ import annotations

from concurrent.futures import Future
from enum import Enum
from typing import Any, Protocol


class EngineUnavailableError(RuntimeError):
    """The engine backend can no longer serve requests.

    Raised -- or set on pending futures -- when worker processes died, the
    backend failed, or the engine was closed, so a routing layer can tell
    "restart or route elsewhere" apart from "fix the request". On the inline
    and external executors a request-local failure (a model plugin rejecting
    its payload) propagates as the plugin's own exception type, never as this
    one. Managed workers are fail-fast: an exception inside a worker ends the
    worker group and pending futures see this type, because the ranks of a
    replica share collectives and cannot resume mid-request.
    """


class LifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class SupervisorProtocol(Protocol):
    """What managed replica executors need from a worker supervisor.

    ``healthy`` and ``inflight`` are indexed by replica id; a supervisor owns
    every replica of one local worker group.
    """

    @property
    def state(self) -> LifecycleState: ...

    @property
    def healthy(self) -> tuple[bool, ...]: ...

    @property
    def inflight(self) -> tuple[int, ...]: ...

    def start(self) -> None: ...

    def submit(self, replica_id: int, payload: Any) -> Future[Any]: ...

    def close(self) -> None: ...


__all__ = ["EngineUnavailableError", "LifecycleState", "SupervisorProtocol"]
