"""Unit tests for phyai.utils.logging.

The module mutates process-global logging state (the LogRecord factory, the
root handler), so every test restores what it touched.
"""

from __future__ import annotations

import logging

import pytest

from phyai.utils import logging as plog
from phyai.utils.logging import (
    as_phyai_logger,
    get_logger,
    install_rank_label,
    rank_label,
    reset_rank_label_cache,
)


@pytest.fixture(autouse=True)
def _restore_logging_globals(monkeypatch):
    # Clear the launcher env so the "no label" assertions do not depend on how
    # the suite itself was launched; tests that want it set it explicitly.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    factory = logging.getLogRecordFactory()
    installed = plog._rank_factory_installed
    # configure_logging() sets a level and handlers on the "phyai" logger and
    # the root; a leak would filter DEBUG records out of every later test.
    saved = {
        name: (logger.level, list(logger.handlers), logger.propagate)
        for name, logger in ((n, logging.getLogger(n)) for n in ("", "phyai"))
    }
    reset_rank_label_cache()
    yield
    logging.setLogRecordFactory(factory)
    plog._rank_factory_installed = installed
    for name, (level, handlers, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate
    reset_rank_label_cache()
    plog._log_once.cache_clear()


@pytest.fixture
def as_rank(monkeypatch):
    """Pretend this process is rank ``r`` of ``w``."""

    def _apply(r: int, w: int = 4):
        monkeypatch.setattr(plog.dist, "is_available", lambda: True)
        monkeypatch.setattr(plog.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(plog.dist, "get_rank", lambda: r)
        monkeypatch.setattr(plog.dist, "get_world_size", lambda: w)
        reset_rank_label_cache()

    return _apply


EXTRA_METHODS = (
    "log_rank0",
    "debug_rank0",
    "info_rank0",
    "warning_rank0",
    "error_rank0",
    "debug_once",
    "info_once",
    "warning_once",
)


def test_get_logger_patches_the_shared_singleton():
    """Instance patching works on the stdlib singleton, so an already-held
    reference gains the methods too, and a plain logger can be adopted."""
    captured = logging.getLogger("phyai.test.captured")
    assert not hasattr(captured, "info_rank0")
    logger = get_logger("phyai.test.captured")
    assert logger is captured is get_logger("phyai.test.captured")
    for name in EXTRA_METHODS:
        assert callable(getattr(logger, name)), name
    plain = logging.getLogger("phyai.test.adopt")
    assert as_phyai_logger(plain) is plain and callable(plain.info_rank0)


# --------------------------------------------------------------------------- #
# rank_label                                                                  #
# --------------------------------------------------------------------------- #


def test_label_is_empty_for_single_process_runs_and_read_from_the_launcher_env(
    monkeypatch,
):
    """Under torchrun, early startup logs precede distributed initialization;
    a single process must not carry rank noise on every line."""
    assert rank_label() == ""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    reset_rank_label_cache()
    assert rank_label() == "[rank 3/4] "
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    reset_rank_label_cache()
    assert rank_label() == ""


def test_only_the_distributed_label_is_cached_and_the_group_beats_the_env(
    as_rank, monkeypatch
):
    """The env is a stand-in and an early empty label must not stick; once the
    real group exists it is authoritative and its answer is cached."""
    assert rank_label() == ""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    reset_rank_label_cache()
    assert rank_label() == "[rank 3/4] "
    as_rank(1, 2)
    assert rank_label() == "[rank 1/2] "

    def _boom():  # pragma: no cover - must not be reached
        raise AssertionError("rank should be cached after the first resolve")

    monkeypatch.setattr(plog.dist, "get_rank", _boom)
    assert rank_label() == "[rank 1/2] "
    reset_rank_label_cache()
    with pytest.raises(AssertionError):
        rank_label()


# --------------------------------------------------------------------------- #
# install_rank_label                                                          #
# --------------------------------------------------------------------------- #


def test_install_is_idempotent_and_chains_the_previous_factory():
    """A factory installed by the embedding application must keep working."""
    previous = logging.getLogRecordFactory()

    def custom(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.custom_marker = "kept"
        return record

    logging.setLogRecordFactory(custom)
    plog._rank_factory_installed = False
    assert install_rank_label() is True
    assert install_rank_label() is False
    record = logging.getLogRecordFactory()("n", logging.INFO, "p", 1, "msg", None, None)
    assert record.custom_marker == "kept" and hasattr(record, "rank")


def test_every_record_gets_a_rank_attribute_and_the_default_format_renders_it(
    as_rank, caplog
):
    """Including records from loggers that never went through get_logger."""
    plog._rank_factory_installed = False
    install_rank_label()
    with caplog.at_level(logging.INFO):
        logging.getLogger("third.party.untouched").info("hello")
    assert caplog.records[0].rank == ""
    as_rank(1, 2)
    record = logging.getLogRecordFactory()(
        "phyai.x", logging.INFO, "p", 1, "hello", None, None
    )
    assert "[rank 1/2] phyai.x: hello" in logging.Formatter(
        plog.DEFAULT_LOG_FORMAT
    ).format(record)
    # pytest owns the root handler, so this is the branch CI actually takes.
    plog._rank_factory_installed = False
    plog.configure_logging()
    assert hasattr(
        logging.getLogRecordFactory()("n", logging.INFO, "p", 1, "m", None, None),
        "rank",
    )


# --------------------------------------------------------------------------- #
# rank gating and *_once                                                      #
# --------------------------------------------------------------------------- #


def test_rank0_methods_emit_on_the_selected_rank_only(as_rank, caplog):
    as_rank(1, 4)
    logger = get_logger("phyai.test.gate")
    with caplog.at_level(logging.DEBUG):
        logger.debug_rank0("no")
        logger.info_rank0("no")
        logger.warning_rank0("no")
        logger.error_rank0("no")
        logger.log_rank0(logging.ERROR, "no")
        logger.info_rank0("on one", rank=1)
    assert [r.getMessage() for r in caplog.records] == ["on one"]
    caplog.clear()
    as_rank(0, 4)
    with caplog.at_level(logging.INFO):
        logger.info_rank0("visible")
    assert [r.getMessage() for r in caplog.records] == ["visible"]


def test_rank0_methods_keep_levels_and_lazy_formatting_without_a_process_group(caplog):
    logger = get_logger("phyai.test.levels")
    with caplog.at_level(logging.DEBUG):
        logger.debug_rank0("d")
        logger.info_rank0("x=%d y=%s", 3, "a")
        logger.warning_rank0("w")
        logger.error_rank0("e")
        logger.log_rank0(logging.CRITICAL, "c")
    assert [r.levelno for r in caplog.records] == [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    assert caplog.records[1].getMessage() == "x=3 y=a"


def test_once_drops_repeats_per_logger_and_per_arguments_at_the_right_level(caplog):
    a = get_logger("phyai.test.once.a")
    b = get_logger("phyai.test.once.b")
    with caplog.at_level(logging.DEBUG):
        a.warning_once("falling back to %s", "sdpa")
        a.warning_once("falling back to %s", "sdpa")
        a.warning_once("falling back to %s", "torch")
        b.warning_once("falling back to %s", "sdpa")  # a different logger
        a.debug_once("d")
        a.info_once("i")
    assert [r.getMessage() for r in caplog.records] == [
        "falling back to sdpa",
        "falling back to torch",
        "falling back to sdpa",
        "d",
        "i",
    ]
    assert [r.levelno for r in caplog.records[-3:]] == [
        logging.WARNING,
        logging.DEBUG,
        logging.INFO,
    ]
