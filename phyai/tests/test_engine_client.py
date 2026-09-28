"""Deployment selection and inline dispatcher behavior."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from phyai.server.deployment import DeploymentConfig, choose_executor_mode
from phyai.server.dispatcher import RequestDispatcher
from phyai.server.executor import EngineUnavailableError, InlineExecutor
from phyai.server.local import _resolve_local_devices


def _clear_rank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_auto_mode_is_inline_only_for_one_rank_one_replica_no_placement(monkeypatch):
    _clear_rank_env(monkeypatch)
    assert choose_executor_mode(1, DeploymentConfig()) == "inline"
    assert choose_executor_mode(2, DeploymentConfig()) == "local"
    assert choose_executor_mode(1, DeploymentConfig(replica_count=2)) == "local"
    assert choose_executor_mode(1, DeploymentConfig(devices=("3",))) == "local"


def test_external_mode_reuses_launcher_but_managed_mode_rejects_it(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert choose_executor_mode(2, DeploymentConfig.external()) == "external"
    with pytest.raises(ValueError, match="launcher is already active"):
        choose_executor_mode(2, DeploymentConfig())


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    (("0", None, "provided together"), ("bad", "2", "integer")),
)
def test_partial_or_invalid_rank_environment_is_rejected(
    monkeypatch,
    rank,
    world_size,
    message,
):
    monkeypatch.setenv("RANK", rank)
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(ValueError, match=message):
        choose_executor_mode(1, DeploymentConfig())


def test_explicit_local_devices_override_cuda_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    # Explicit entries are logical indices into this process's visible set,
    # so a shell-level mask changes what they mean but never their values.
    assert _resolve_local_devices((2, "3"), 2, "cuda") == (2, 3)
    assert _resolve_local_devices((), 2, "cpu") == (0, 1)


def test_deployment_configuration_rejects_ambiguous_fields():
    with pytest.raises(ValueError, match="exactly one replica"):
        DeploymentConfig(mode="external", replica_count=2)


def test_public_lazy_exports_all_resolve():
    # Lazy __getattr__ maps fail at access time, not import time; force every
    # advertised name to resolve so a stale key cannot hide until runtime.
    import phyai
    import phyai.server

    for name in phyai.__all__:
        assert getattr(phyai, name) is not None
    for name in phyai.server.__all__:
        assert getattr(phyai.server, name) is not None


class _Core:
    def __init__(self, _args, **_kwargs):
        self.args = type("Args", (), {"plugin": "test"})()
        self.closed = 0
        self.thread_ids: list[int] = []

    def step(self, request):
        self.thread_ids.append(threading.get_ident())
        return request

    def close(self):
        self.closed += 1


class _BlockingCore(_Core):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def step(self, request):
        self.calls.append(request)
        self.started.set()
        self.release.wait(timeout=2)
        return super().step(request)


class _RecoverableCore(_Core):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.fail_next = True

    def step(self, request):
        if self.fail_next:
            self.fail_next = False
            raise ValueError("request rejected")
        return super().step(request)


class _LazyExecutor:
    """Fake replica encoding the lazy-start contract: CREATED is healthy."""

    mode = "fake"
    rank_count = 1

    def __init__(self) -> None:
        self.replica_id = 0
        self.started = False
        self.failed = False
        self.start_calls = 0
        self.closed = False
        self._inflight = 0

    def start(self) -> None:
        if self.closed:
            raise EngineUnavailableError("closed")
        if not self.started:
            self.start_calls += 1
            self.started = True

    def step(self, request):
        return self.submit(request).result()

    def submit(self, request):
        self.start()
        future: Future[object] = Future()
        self._inflight += 1
        future.add_done_callback(lambda _done: setattr(self, "_inflight", 0))
        future.set_result(request)
        return future

    @property
    def healthy(self) -> bool:
        return not self.closed and not self.failed

    @property
    def inflight(self) -> int:
        return self._inflight

    def close(self) -> None:
        self.closed = True


def _client(core_type):
    executor = InlineExecutor(core_type, object())
    return RequestDispatcher((executor,)), executor


class _ConfiguredCore(_Core):
    """Core exposing an EngineConfig-shaped ``config.device.target``."""

    def __init__(self, _args, **_kwargs):
        super().__init__(_args, **_kwargs)
        device = type("Device", (), {"target": "cuda"})()
        self.config = type("Config", (), {"device": device})()


def test_inline_executor_binds_a_bare_cuda_target(monkeypatch):
    import torch

    pinned: list[object] = []
    monkeypatch.setattr(torch.cuda, "set_device", pinned.append)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    dispatcher, _executor = _client(_ConfiguredCore)
    dispatcher.setup()

    # The default DeviceConfig target is an index-less "cuda"; binding must
    # resolve it (device 0 without a launcher) instead of handing it to torch.
    assert dispatcher.step("request") == "request"
    assert pinned == [torch.device("cuda", 0)]
    dispatcher.close()


def test_inline_step_uses_the_caller_thread_and_starts_no_helper():
    dispatcher, executor = _client(_Core)
    caller = threading.get_ident()

    assert executor._worker is None
    dispatcher.setup()
    assert dispatcher.step("request") == "request"
    assert executor.core.thread_ids == [caller]
    assert executor._worker is None
    assert executor.inflight == 0

    dispatcher.close()
    dispatcher.close()
    assert executor.core.closed == 1


def test_dispatcher_starts_a_created_executor_exactly_once():
    executor = _LazyExecutor()
    dispatcher = RequestDispatcher((executor,))

    # Lazily on the first request, and setup() is idempotent afterwards.
    assert dispatcher.submit("request").result(timeout=1) == "request"
    dispatcher.setup()
    dispatcher.setup()
    assert executor.start_calls == 1
    dispatcher.close()


def test_inline_submit_lazily_starts_one_worker_and_serializes_step():
    dispatcher, executor = _client(_BlockingCore)
    dispatcher.setup()
    asynchronous = dispatcher.submit("async")
    assert executor.core.started.wait(timeout=1)
    synchronous: list[str] = []
    thread = threading.Thread(
        target=lambda: synchronous.append(dispatcher.step("sync"))
    )
    thread.start()
    time.sleep(0.05)

    assert executor.core.calls == ["async"]
    assert executor.inflight == 2
    executor.core.release.set()
    assert asynchronous.result(timeout=1) == "async"
    thread.join(timeout=1)
    assert synchronous == ["sync"]
    assert executor.core.calls == ["async", "sync"]
    dispatcher.close()


def test_inline_request_error_does_not_poison_the_replica():
    dispatcher, executor = _client(_RecoverableCore)
    dispatcher.setup()

    with pytest.raises(ValueError, match="request rejected") as first_error:
        dispatcher.step("bad")
    # A request-local failure is the plugin's own exception type, never the
    # backend-dead marker.
    assert not isinstance(first_error.value, EngineUnavailableError)
    assert executor.healthy
    assert dispatcher.step("good") == "good"

    executor.core.fail_next = True
    failed = dispatcher.submit("bad-again")
    with pytest.raises(ValueError, match="request rejected"):
        failed.result(timeout=1)
    assert executor.healthy
    dispatcher.close()


def test_inline_submit_cancel_cancels_queued_work():
    dispatcher, executor = _client(_BlockingCore)
    dispatcher.setup()
    active = dispatcher.submit("active")
    assert executor.core.started.wait(timeout=1)
    queued = dispatcher.submit("queued")

    assert queued.cancel()
    assert queued.cancelled()
    executor.core.release.set()
    assert active.result(timeout=1) == "active"
    # The cancelled request never reaches the core.
    assert executor.core.calls == ["active"]
    dispatcher.close()


def test_closed_dispatcher_raises_engine_unavailable():
    dispatcher, _executor = _client(_Core)
    dispatcher.setup()
    dispatcher.close()

    with pytest.raises(EngineUnavailableError):
        dispatcher.step("request")
    failed = dispatcher.submit("request")
    with pytest.raises(EngineUnavailableError):
        failed.result(timeout=1)


def test_unhealthy_replicas_raise_engine_unavailable():
    executor = _LazyExecutor()
    dispatcher = RequestDispatcher((executor,))
    dispatcher.setup()
    executor.failed = True

    with pytest.raises(EngineUnavailableError, match="no healthy replicas"):
        dispatcher.step("request")
    dispatcher.close()
