"""Pickle-safe control messages shared by process supervisors and workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkerHello:
    worker_id: int
    replica_id: int
    replica_rank: int
    pid: int


@dataclass(frozen=True, slots=True)
class WorkerReady:
    worker_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecuteRequest:
    request_id: int
    payload: Any


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    reason: str = "shutdown"


@dataclass(frozen=True, slots=True)
class WorkerResult:
    worker_id: int
    request_id: int
    payload: Any = None


@dataclass(frozen=True, slots=True)
class WorkerError:
    worker_id: int
    phase: str
    error: str
    traceback: str
    request_id: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerStopped:
    worker_id: int


__all__ = [
    "ExecuteRequest",
    "ShutdownRequest",
    "WorkerError",
    "WorkerHello",
    "WorkerReady",
    "WorkerResult",
    "WorkerStopped",
]
