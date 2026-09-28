"""Logging for phyai: rank-aware loggers with the ceremony removed.

Call sites use plain :mod:`logging` methods plus a few extras:

    from phyai.utils.logging import get_logger

    logger = get_logger(__name__)

    logger.info("built comm on axis %s", axis)      # every rank
    logger.info_rank0("engine ready in %.2fs", t)   # rank 0 only
    logger.warning_once("falling back to %s", name) # first occurrence only

Neither the logger nor the level is ever passed as an argument. That is the
whole point: the previous API (``this_rank_log(logger, logging.INFO, msg,
...)``) put two mandatory arguments in front of every message, which pushed
one-line logs past the formatter's column limit and expanded 51 of 63 call
sites into 4-6 line blocks.

Where the rank label comes from
-------------------------------
Not from the call site. :func:`install_rank_label` chains a
:class:`logging.LogRecord` factory that stamps ``record.rank`` on every
record created anywhere in the process, and
:data:`DEFAULT_LOG_FORMAT` prints it. Consequences worth knowing:

* "log on every rank, labelled" needs no helper at all — plain
  ``logger.info(...)`` is already labelled. So there is no ``all_ranks_log``.
* Records from third-party libraries get labelled too, without touching them.
* The label is **empty in a single-process run**, so ordinary single-GPU logs
  carry no rank noise; ``[rank 3/4] `` appears only when it means something.
  Under a launcher it comes from ``RANK`` / ``WORLD_SIZE`` until the process
  group is up, so the pre-``init_process_group`` lines are labelled too.
* The rank is resolved when a record is *created*, which is after
  :meth:`logging.Logger.isEnabledFor` has already dropped filtered levels —
  so a DEBUG-heavy path pays nothing.

A record factory rather than a :class:`logging.Filter`: a filter on a logger
is not inherited by its children, and a filter on a handler only covers that
one handler, which misses every record when the embedding application owns
the handlers.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from types import MethodType
from typing import Any, Callable, Hashable, cast

import torch.distributed as dist

from phyai.env import envs


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(rank)s%(name)s: %(message)s"


# --------------------------------------------------------------------------- #
# Rank label                                                                  #
# --------------------------------------------------------------------------- #

#: Cached label. Only ever set from the process group, because rank and world
#: size are fixed for the life of a process from that point on. The launcher-env
#: fallback below is deliberately NOT cached: it is a stand-in until the real
#: group exists, and the group is authoritative.
_rank_label_cache: str | None = None


def rank_label() -> str:
    """``"[rank 3/4] "`` when this process is one of several, ``""`` otherwise.

    Resolved from the live process group when there is one, and otherwise from
    the launcher's ``RANK`` / ``WORLD_SIZE``. The env fallback matters more
    than it looks: under ``torchrun`` several startup lines are emitted before
    ``init_process_group`` runs, and without the fallback
    every rank writes them unlabelled and interleaved, which is precisely
    when a reader most needs to know who said what.

    Stays empty for a genuinely single-process run, so ordinary single-GPU
    logs carry no rank noise. The trailing space is part of the label so the
    format string can concatenate it (``%(rank)s%(name)s``) and collapse to
    nothing.
    """
    global _rank_label_cache
    if _rank_label_cache is not None:
        return _rank_label_cache
    if dist.is_available() and dist.is_initialized():
        _rank_label_cache = f"[rank {dist.get_rank()}/{dist.get_world_size()}] "
        return _rank_label_cache
    rank, world = os.environ.get("RANK"), os.environ.get("WORLD_SIZE")
    if rank is not None and world is not None and world != "1":
        return f"[rank {rank}/{world}] "
    return ""


def reset_rank_label_cache() -> None:
    """Forget the cached label. For tests, and after tearing down a group."""
    global _rank_label_cache
    _rank_label_cache = None


_rank_factory_installed = False


def install_rank_label() -> bool:
    """Stamp ``record.rank`` on every :class:`logging.LogRecord` created.

    Returns ``True`` if this call installed the factory, ``False`` if it was
    already in place. Idempotent, and chains: whatever factory was set
    before is called first, so a custom factory installed by the embedding
    application keeps working.
    """
    global _rank_factory_installed
    if _rank_factory_installed:
        return False
    previous: Callable[..., logging.LogRecord] = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.rank = rank_label()
        return record

    logging.setLogRecordFactory(factory)
    _rank_factory_installed = True
    return True


def configure_logging(level: int | str | None = None) -> bool:
    """Install the rank label, and a stderr handler if nothing else has one.

    Args:
        level: level to set. ``None`` reads ``PHYAI_LOG_LEVEL`` (a name
            like ``INFO`` or a number), defaulting to ``INFO``.

    Returns:
        ``True`` if a handler was installed, ``False`` if one already
        existed and phyai left the handler configuration alone. The rank
        label is installed either way.

    The handler guard is the point. An application that has configured
    logging — a server with structured output, a test harness capturing
    records — owns that decision, and a library that adds a second handler
    on top produces duplicated lines. But the common case for phyai is a
    script that configured nothing at all, where staying silent discards its
    startup and stage logs.
    """
    install_rank_label()
    if level is None:
        raw = envs.PHYAI_LOG_LEVEL.get() or "INFO"
        level = int(raw) if raw.isdigit() else raw.upper()
    root = logging.getLogger()
    if root.handlers:
        # Someone else owns logging; only make sure phyai's records are not
        # filtered out before reaching their handler. Their format decides
        # whether the rank label is shown -- the attribute is there either way.
        if root.level > logging.INFO or root.level == logging.NOTSET:
            logging.getLogger("phyai").setLevel(level)
        return False
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
    return True


# --------------------------------------------------------------------------- #
# Rank-gated and de-duplicating logger methods                                #
# --------------------------------------------------------------------------- #


def _is_rank(rank: int) -> bool:
    """``True`` when this process is ``rank``, or when there is no group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == rank
    return True


@lru_cache
def _log_once(logger: logging.Logger, level: int, msg: str, *args: Hashable) -> None:
    # stacklevel=3: caller -> the public *_once method -> here -> Logger.log,
    # so %(filename)s / %(lineno)d point at the original call site.
    logger.log(level, msg, *args, stacklevel=3)


class PhyaiLogger(logging.Logger):
    """Type surface for the logger :func:`get_logger` returns.

    The methods are patched onto the logger *instance*, not installed via
    :func:`logging.setLoggerClass`. That choice is deliberate:
    ``setLoggerClass`` is process-global, so if two libraries both call it
    the last one wins and the other's methods silently vanish. phyai shares
    a process with triton, flashinfer and apache-tvm-ffi. (vLLM patches
    instances for the same reason.) This class exists so type checkers and
    IDEs know the methods are there.
    """

    def log_rank0(
        self, level: int, msg: str, *args: Any, rank: int = 0, **kwargs: Any
    ) -> None:
        """Log at a caller-computed ``level``, on one rank only.

        The escape hatch for a level held in a variable. Prefer the fixed-level
        methods below.
        """
        if not _is_rank(rank):
            return
        kwargs.setdefault("stacklevel", 2)
        self.log(level, msg, *args, **kwargs)

    def debug_rank0(self, msg: str, *args: Any, rank: int = 0, **kwargs: Any) -> None:
        """As :meth:`~logging.Logger.debug`, but only on ``rank``."""
        if not _is_rank(rank):
            return
        kwargs.setdefault("stacklevel", 2)
        self.debug(msg, *args, **kwargs)

    def info_rank0(self, msg: str, *args: Any, rank: int = 0, **kwargs: Any) -> None:
        """As :meth:`~logging.Logger.info`, but only on ``rank``."""
        if not _is_rank(rank):
            return
        kwargs.setdefault("stacklevel", 2)
        self.info(msg, *args, **kwargs)

    def warning_rank0(self, msg: str, *args: Any, rank: int = 0, **kwargs: Any) -> None:
        """As :meth:`~logging.Logger.warning`, but only on ``rank``."""
        if not _is_rank(rank):
            return
        kwargs.setdefault("stacklevel", 2)
        self.warning(msg, *args, **kwargs)

    def error_rank0(self, msg: str, *args: Any, rank: int = 0, **kwargs: Any) -> None:
        """As :meth:`~logging.Logger.error`, but only on ``rank``."""
        if not _is_rank(rank):
            return
        kwargs.setdefault("stacklevel", 2)
        self.error(msg, *args, **kwargs)

    def debug_once(self, msg: str, *args: Hashable) -> None:
        """As :meth:`~logging.Logger.debug`; repeats are dropped."""
        _log_once(self, logging.DEBUG, msg, *args)

    def info_once(self, msg: str, *args: Hashable) -> None:
        """As :meth:`~logging.Logger.info`; repeats are dropped."""
        _log_once(self, logging.INFO, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """As :meth:`~logging.Logger.warning`; repeats are dropped.

        De-duplication is by ``(logger, message, *args)``, so a warning
        parameterized by layer name still fires once per layer. Use this for
        anything on a per-request or per-layer path — a kernel fallback, a
        quantization downgrade — where the same line would otherwise bury
        the log.
        """
        _log_once(self, logging.WARNING, msg, *args)


_METHODS_TO_PATCH: dict[str, Any] = {
    "log_rank0": PhyaiLogger.log_rank0,
    "debug_rank0": PhyaiLogger.debug_rank0,
    "info_rank0": PhyaiLogger.info_rank0,
    "warning_rank0": PhyaiLogger.warning_rank0,
    "error_rank0": PhyaiLogger.error_rank0,
    "debug_once": PhyaiLogger.debug_once,
    "info_once": PhyaiLogger.info_once,
    "warning_once": PhyaiLogger.warning_once,
}


def get_logger(name: str) -> PhyaiLogger:
    """Return the logger for ``name`` with phyai's extra methods attached.

    Use in place of ``logging.getLogger(__name__)`` at module scope.
    Idempotent, and because :func:`logging.getLogger` returns a per-name
    singleton, patching here means *every* reference to that logger gains
    the methods — including one captured before this call.
    """
    logger = logging.getLogger(name)
    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))
    return cast(PhyaiLogger, logger)


def as_phyai_logger(logger: logging.Logger) -> PhyaiLogger:
    """Adopt a plain :class:`logging.Logger` handed in from outside.

    For the few APIs that accept a caller-supplied logger and need the extra
    methods on it. Same singleton identity, so the caller's object is patched
    in place.
    """
    return get_logger(logger.name)


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "PhyaiLogger",
    "as_phyai_logger",
    "configure_logging",
    "get_logger",
    "install_rank_label",
    "rank_label",
    "reset_rank_label_cache",
]
